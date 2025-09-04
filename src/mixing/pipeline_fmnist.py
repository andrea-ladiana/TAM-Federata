# -*- coding: utf-8 -*-
"""
Exp-06 (single-only) — pipeline FMNIST (dataset strutturato).

Questa pipeline riprende lo schema di `pipeline_core` ma opera su immagini già
fornite come array NumPy (niente dipendenze esterne). Le immagini sono binarizzate
in {±1} e appiattite in vettori di dimensione N. Gli archetipi ξ_true (K,N)
sono ottenuti per classe con `sign(mean_per_class)` (opzione "medoid" non richiede
dipendenze: omessa per semplicità, si può aggiungere in seguito).

Input principale:
  - (X, y): immagini e label (interi) già caricati dal chiamante
  - classes: tripletta di classi FMNIST da usare (K=3)
  - pis: mixing schedule (T,K)
Il resto è identico: stima J_unsup (single), blend w, propagate, eigen_cut,
disentangling TAM, allineamento, metriche, Hopfield round-wise.

Riuso moduli dalla codebase (vedi codebase.txt).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any, List

import json
import math
import os
import numpy as np

# --- riuso dalla codebase (vedi codebase.txt) ---
from src.unsup.config import HyperParams  # :contentReference[oaicite:18]{index=18}
from src.unsup.data import new_round_single, compute_round_coverage, count_exposures  # :contentReference[oaicite:19]{index=19}
from src.unsup.estimators import build_unsup_J_single, blend_with_memory  # :contentReference[oaicite:20]{index=20}
from src.unsup.functions import propagate_J, estimate_K_eff_from_J  # :contentReference[oaicite:21]{index=21}
from src.unsup.dynamics import eigen_cut, dis_check  # :contentReference[oaicite:22]{index=22}
from src.unsup.hopfield_eval import run_or_load_hopfield_eval  # :contentReference[oaicite:23]{index=23}
from .control import (
    compute_drift_signals,
    update_w_threshold,
    update_w_sigmoid,
    update_w_pctrl,
)
from .metrics import lag_and_amplitude, simplex_embed_2d


# ----------------------------
# utilità I/O
# ----------------------------
def _ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, indent=2))


# ----------------------------
# helper: binarizzazione e archetipi
# ----------------------------
def _binarize_pm1(X: np.ndarray, thresh: Optional[float] = None) -> np.ndarray:
    """
    Binarizza immagini in {±1}. Se thresh è None, usa la mediana globale di X.
    X può essere (N_img, H, W) o (N_img, D).
    """
    Xf = X.astype(np.float32)
    if Xf.ndim == 3:
        n, h, w = Xf.shape
        Xf = Xf.reshape(n, h * w)
    if thresh is None:
        thresh = float(np.median(Xf))
    return np.where(Xf > thresh, 1.0, -1.0).astype(np.float32)


def _make_xi_true_from_classes(X_pm1: np.ndarray, y: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """
    Costruisce archetipi ξ_true (K,N) come sign(mean_per_class) sulle immagini binarizzate.
    """
    K = len(classes)
    xi_true = []
    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            raise ValueError(f"Nessuna immagine per classe {c}.")
        m = np.mean(X_pm1[idx], axis=0)  # (N,)
        xi_true.append(np.where(m >= 0.0, 1, -1).astype(int))
    return np.stack(xi_true, axis=0)  # (K, N)


# ----------------------------
# helper: dataset round-wise con schedule pis
# ----------------------------
def _build_eta_from_images(
    X_pm1: np.ndarray, y: np.ndarray, classes: Sequence[int],
    pis: np.ndarray, M_total: int, L: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera ETA, labels scegliendo immagini reali dalle classi richieste secondo pis(t).
    Ritorna:
      ETA    : (L, T, M_c, N)
      labels : (L, T, M_c)
    """
    T, K = pis.shape
    N = X_pm1.shape[1]
    M_c = int(math.ceil(M_total / float(L * T)))
    ETA = np.zeros((L, T, M_c, N), dtype=np.float32)
    labels = np.zeros((L, T, M_c), dtype=np.int32)

    # indici per classe (nella stessa order di 'classes')
    idx_by_class = [np.where(y == c)[0] for c in classes]
    for k, idx in enumerate(idx_by_class):
        if idx.size == 0:
            raise ValueError(f"Nessuna immagine disponibile per la classe {classes[k]}.")

    for t in range(T):
        pi_t = np.asarray(pis[t], dtype=float)
        pi_t = np.maximum(pi_t, 0.0)
        if pi_t.sum() <= 0:
            pi_t = np.ones(K, dtype=float) / float(K)
        else:
            pi_t = pi_t / pi_t.sum()

        for l in range(L):
            # campiona classi secondo pi_t
            ks = rng.choice(K, size=M_c, p=pi_t).astype(int)  # (M_c,)
            mus = ks  # mappa 0..K-1 direttamente alle classi scelte
            labels[l, t] = mus.astype(np.int32)
            # per ogni esempio, scegli immagine casuale della classe corrispondente
            for m in range(M_c):
                k = int(ks[m])
                pool = idx_by_class[k]
                j = int(rng.integers(low=0, high=pool.size))
                ETA[l, t, m] = X_pm1[pool[j]]
    return ETA, labels


# ----------------------------
# helper: allineamento greedy + pi_hat
# ----------------------------
def _align_greedy_sign(xi_r: np.ndarray, xi_true: np.ndarray):
    if xi_r.size == 0:
        return xi_r, {}
    R, N = xi_r.shape
    K = xi_true.shape[0]
    S = np.abs(xi_r @ xi_true.T) / float(N)
    S = S.copy()
    used_r, used_mu = set(), set()
    match: Dict[int, int] = {}
    while len(used_r) < min(R, K):
        r, mu = np.unravel_index(np.argmax(S, axis=None), S.shape)
        r, mu = int(r), int(mu)
        if r in used_r or mu in used_mu:
            S[r, mu] = -np.inf
            continue
        if float(np.dot(xi_r[r], xi_true[mu])) < 0.0:
            xi_r[r] = -xi_r[r]
        match[r] = mu
        used_r.add(r); used_mu.add(mu)
        S[r, :] = -np.inf; S[:, mu] = -np.inf
    return xi_r, match


def _estimate_pi_hat_from_examples(xi_ref: np.ndarray, E_t: np.ndarray) -> np.ndarray:
    L, M_c, N = E_t.shape
    S = xi_ref
    if S.ndim != 2:
        raise ValueError("xi_ref deve essere (S,N).")
    X = E_t.reshape(L * M_c, N)
    Ov = X @ S.T
    mu_hat = np.argmax(Ov, axis=1)
    K = S.shape[0]
    counts = np.bincount(mu_hat, minlength=K).astype(float)
    if counts.sum() <= 0:
        return np.ones(K) / float(K)
    return counts / counts.sum()


# ----------------------------
# main: run_seed_fmnist
# ----------------------------
def run_seed_fmnist(
    hp: HyperParams,
    seed: int,
    *,
    outdir: str,
    X: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
    pis: Optional[np.ndarray] = None,           # (T,K)
    binarize_thresh: Optional[float] = None,    # None => mediana globale
    eval_hopfield_every: int = 1,               # 1=ogni round, 0=off, n>1 => ogni n
    # --- controllo w ---
    w_policy: str = "pctrl",
    w_init: Optional[float] = None,
    w_min: float = 0.05,
    w_max: float = 0.95,
    alpha_w: float = 0.3,
    a_drift: float = 0.5,
    b_mismatch: float = 1.0,
    # policy A
    theta_low: float = 0.05,
    theta_high: float = 0.15,
    delta_up: float = 0.10,
    delta_down: float = 0.05,
    # policy B
    theta_mid: float = 0.12,
    beta: float = 10.0,
    # policy C
    lag_target: float = 0.3,
    lag_window: int = 8,
    kp: float = 0.8,
    ki: float = 0.0,
    kd: float = 0.0,
    gate_drift_theta: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Esegue Exp-06 (single-only) su FMNIST (3 classi).

    Parametri
    ---------
    hp, seed, outdir : come in pipeline_core
    X, y : immagini (N_img, H, W) o (N_img, D) e label interi
    classes : tripletta di classi da usare come archetipi (K deve essere 3)
    pis : mixing schedule (T,K); se None => uniforme
    binarize_thresh : soglia globale per la binarizzazione {±1} (None => mediana)
    eval_hopfield_every : frequenza valutazioni Hopfield

    Returns
    -------
    summary : dict
    """
    if len(classes) != hp.K:
        raise ValueError(f"hp.K={hp.K} ma classes ha lunghezza {len(classes)}.")

    rng = np.random.default_rng(int(hp.seed_base + seed))
    out = _ensure_dir(outdir)
    _save_json(out / "hyperparams.json", asdict(hp))
    _save_json(out / "classes.json", {"classes": list(map(int, classes))})

    # binarizzazione in {±1}
    X_pm1 = _binarize_pm1(X, thresh=binarize_thresh)

    # archetipi per classe (sign(mean_per_class))
    xi_true = _make_xi_true_from_classes(X_pm1, y, classes)
    np.save(out / "xi_true.npy", xi_true.astype(np.int8))

    # mixing schedule
    T = int(hp.n_batch)
    if pis is None:
        pis = np.ones((T, hp.K), dtype=float) / float(hp.K)
    np.save(out / "pis.npy", pis.astype(np.float32))

    # costruisci ETA round-wise da immagini reali
    ETA, labels = _build_eta_from_images(
        X_pm1=X_pm1, y=y, classes=classes,
        pis=pis, M_total=hp.M_total, L=hp.L, rng=rng
    )
    np.save(out / "ETA.shape.npy", np.array(ETA.shape, dtype=int))
    np.save(out / "labels.npy", labels.astype(np.int16))

    exposure = count_exposures(labels, K=hp.K)  # :contentReference[oaicite:24]{index=24}
    np.save(out / "exposure_counts.npy", exposure.astype(np.int32))

    xi_ref_prev: Optional[np.ndarray] = None
    # controllo w
    w_curr: float = float(hp.w if w_init is None else w_init)
    w_series: List[float] = []
    drift_series: List[Tuple[float, float, float]] = []
    pi_data_hist: List[np.ndarray] = []
    pi_mem_hist: List[np.ndarray] = []
    lag_abs_hist: List[float] = []
    pi_data_prev: Optional[np.ndarray] = None
    pi_mem_prev: Optional[np.ndarray] = None
    all_rounds = []
    for t in range(T):
        rdir = _ensure_dir(out / f"round_{t:03d}")
        E_t = new_round_single(ETA, t)  # (L, M_c, N)  :contentReference[oaicite:25]{index=25}
        cov_t = float(compute_round_coverage(labels[:, t, :], K=hp.K))  # :contentReference[oaicite:26]{index=26}

        # stima unsup + blend
        J_unsup, M_eff = build_unsup_J_single(E_t, K=hp.K)  # :contentReference[oaicite:27]{index=27}
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref_prev, w=float(np.clip(w_curr, 0.0, 1.0)))  # :contentReference[oaicite:28]{index=28}
        np.save(rdir / "J_unsup.npy", J_unsup.astype(np.float32))
        np.save(rdir / "J_rec.npy", J_rec.astype(np.float32))

        # propagazione + cut spettrale
        J_KS = propagate_J(J_rec, iters=hp.prop.iters, eps=hp.prop.eps, tol=1e-8)  # :contentReference[oaicite:29]{index=29}
        np.save(rdir / "J_KS.npy", J_KS.astype(np.float32))
        vals_sel, V = eigen_cut(J_KS, tau=hp.spec.tau)  # :contentReference[oaicite:30]{index=30}
        np.save(rdir / "eigs_sel.npy", vals_sel.astype(np.float32))
        np.save(rdir / "V_sel.npy", V.astype(np.float32))

        # disentangling TAM + pruning
        xi_r, m = dis_check(
            V, K=hp.K, L=hp.L,
            J_rec=J_rec, JKS_iter=J_KS,
            xi_true=xi_true,
            tam=hp.tam, spec=hp.spec,
            show_progress=hp.use_tqdm,
        )  # :contentReference[oaicite:31]{index=31}
        np.save(rdir / "xi_r.npy", xi_r.astype(np.int8))
        np.save(rdir / "mag_pruning.npy", m.astype(np.float32))

        # allineamento greedy + update memoria
        xi_aligned, match = _align_greedy_sign(xi_r.copy(), xi_true)
        xi_ref_prev = xi_aligned.copy() if xi_aligned.size else xi_ref_prev
        np.save(rdir / "xi_aligned.npy", xi_aligned.astype(np.int8))
        _save_json(rdir / "match.json", {str(k): int(v) for k, v in match.items()})

        # stima K_eff
        K_eff, keep_mask, info = estimate_K_eff_from_J(
            J_KS, method=hp.estimate_keff_method, M_eff=M_eff
        )  # :contentReference[oaicite:32]{index=32}

        # stima pi_hat e TV vs pi_t
        try:
            # Per evitare mismatch di shape quando il pruning rimuove pattern,
            # usa xi_true (K,N) come base per la classificazione dei campioni.
            pi_hat = _estimate_pi_hat_from_examples(xi_ref=xi_true, E_t=E_t)
        except Exception:
            pi_hat = np.ones(hp.K) / float(hp.K)
        pi_t = pis[t] / float(np.sum(pis[t])) if np.sum(pis[t]) > 0 else np.ones(hp.K) / float(hp.K)
        TV_t = 0.5 * float(np.sum(np.abs(pi_hat - pi_t)))

        metrics_t = {
            "coverage": cov_t,
            "K_eff": int(K_eff),
            "M_eff": int(M_eff),
            "n_eigs_sel": int(V.shape[0]),
            "TV_pi": TV_t,
            "pi_hat": pi_hat.tolist(),
            "pi_hat_data": pi_hat.tolist(),
            "pi_true": pi_t.tolist(),
        }
        all_rounds.append(metrics_t)

        # valutazione Hopfield (post-hoc) + estrazione pi_hat_retrieval
        pi_hat_retrieval_vec = None

        # valutazione Hopfield (post-hoc)
        if eval_hopfield_every and ((t % int(eval_hopfield_every)) == 0):
            # Let run_or_load_hopfield_eval create the directory
            hop_dir = rdir / "hopfield"
            try:
                results_h = run_or_load_hopfield_eval(
                    output_dir=str(hop_dir),
                    J_server=J_rec,
                    xi_true=xi_true,
                    exposure_counts=exposure,
                    beta=3.0, updates=30, reps_per_archetype=32, start_overlap=0.3,
                    force_run=True, save=True, stochastic=True,
                )  # :contentReference[oaicite:33]{index=33}
            except Exception:
                results_h = None

            # Estrai magnetizzazioni by-mu
            m_arr = None
            if isinstance(results_h, dict):
                for k in ("magnetization_by_mu", "m_by_mu", "mag_by_mu"):
                    if k in results_h:
                        try:
                            m_arr = np.asarray(results_h[k])
                            break
                        except Exception:
                            m_arr = None
            if m_arr is None:
                npz_path = hop_dir / "magnetization_by_mu.npz"
                if npz_path.exists():
                    try:
                        loaded = np.load(npz_path)
                        m_keys = sorted([k for k in loaded.files if k.startswith("m_")], key=lambda s: int(s.split("_",1)[1]))
                        if m_keys:
                            cols = []
                            for mk in m_keys:
                                v = np.asarray(loaded[mk], dtype=float)
                                if v.ndim == 1:
                                    cols.append(float(v.mean()))
                                else:
                                    cols.append(float(v.reshape(-1).mean()))
                            m_arr = np.asarray(cols, dtype=float)  # (K,)
                    except Exception:
                        m_arr = None
            if m_arr is not None and m_arr.size > 0:
                if m_arr.ndim == 1:
                    m_mu = m_arr.astype(float)
                else:
                    axes = tuple(range(1, m_arr.ndim))
                    m_mu = np.mean(m_arr, axis=axes).astype(float)
                eps = 1e-6
                wvec = np.maximum(m_mu, eps)
                den = float(wvec.sum()) if float(wvec.sum()) > 0 else 1.0
                pi_hat_retrieval_vec = (wvec / den)

        # fallback retrieval = precedente
        if pi_hat_retrieval_vec is None:
            try:
                prev = rdir.parent / f"round_{t-1:03d}" / "metrics.json"
                prev_metrics = json.loads(prev.read_text()) if (t > 0 and prev.exists()) else None
                if isinstance(prev_metrics, dict) and isinstance(prev_metrics.get("pi_hat_retrieval", None), list):
                    arr = np.asarray(prev_metrics["pi_hat_retrieval"], dtype=float)
                    if arr.shape == (hp.K,):
                        pi_hat_retrieval_vec = arr
            except Exception:
                pass

        metrics_t["pi_hat_retrieval"] = (
            pi_hat_retrieval_vec.tolist() if isinstance(pi_hat_retrieval_vec, np.ndarray) else None
        )

        # --- segnali di drift & controllo w ---
        try:
            drift = compute_drift_signals(
                pi_data_t=pi_hat,                 # stima data-driven dal round corrente
                pi_data_tm1=pi_data_prev,
                pi_mem_tm1=pi_mem_prev,
                a=float(a_drift), b=float(b_mismatch),
            )
            D_t, M_t, S_t = float(drift["D_t"]), float(drift["M_t"]), float(drift["S_t"])
        except Exception:
            D_t = M_t = S_t = 0.0

        # lag |phi| (se K=3)
        lag_abs_rad = None
        try:
            pi_mem_curr = None
            if isinstance(pi_hat_retrieval_vec, np.ndarray) and pi_hat_retrieval_vec.shape == (hp.K,):
                pi_mem_curr = pi_hat_retrieval_vec
            elif isinstance(pi_mem_prev, np.ndarray):
                pi_mem_curr = pi_mem_prev

            if pi_mem_curr is not None:
                pi_data_hist.append(np.asarray(pi_hat, dtype=float))
                pi_mem_hist.append(np.asarray(pi_mem_curr, dtype=float))
                W = int(max(1, lag_window))
                if len(pi_data_hist) >= 2 and hp.K == 3:
                    d_win = np.stack(pi_data_hist[-W:], axis=0)
                    m_win = np.stack(pi_mem_hist[-W:], axis=0)
                    la = lag_and_amplitude(d_win, m_win)
                    lag_abs_rad = float(abs(la.get("lag_radians", 0.0)))
                elif hp.K == 3 and pi_mem_curr is not None:
                    xyd = simplex_embed_2d(np.asarray(pi_hat, dtype=float))
                    xym = simplex_embed_2d(np.asarray(pi_mem_curr, dtype=float))
                    th_d = float(np.arctan2(xyd[1], xyd[0]))
                    th_m = float(np.arctan2(xym[1], xym[0]))
                    dphi = (th_d - th_m + np.pi) % (2.0 * np.pi) - np.pi
                    lag_abs_rad = float(abs(dphi))
        except Exception:
            lag_abs_rad = None

        if lag_abs_rad is not None:
            lag_abs_hist.append(float(lag_abs_rad))
        lag_series = np.asarray(lag_abs_hist[-int(max(1, lag_window)):], dtype=float)

        # aggiorna w per round successivo
        w_next = float(w_curr)
        pol = (w_policy or "pctrl").lower().strip()
        try:
            if pol == "fixed":
                w_next = float(hp.w if w_init is None else w_init)
            elif pol == "threshold":
                w_next = update_w_threshold(
                    w_prev=float(w_curr), D_t=D_t, M_t=M_t, S_t=S_t,
                    w_min=float(w_min), w_max=float(w_max),
                    theta_low=float(theta_low), theta_high=float(theta_high),
                    delta_up=float(delta_up), delta_down=float(delta_down),
                    alpha_w=float(alpha_w),
                )
            elif pol == "sigmoid":
                w_next = update_w_sigmoid(
                    w_prev=float(w_curr), D_t=D_t, M_t=M_t, S_t=S_t,
                    w_min=float(w_min), w_max=float(w_max),
                    theta_mid=float(theta_mid), beta=float(beta),
                    alpha_w=float(alpha_w),
                )
            else:
                w_next = update_w_pctrl(
                    w_prev=float(w_curr),
                    lag_series_radians=lag_series,
                    lag_target=float(lag_target),
                    w_min=float(w_min), w_max=float(w_max),
                    kp=float(kp), ki=float(ki), kd=float(kd),
                    alpha_w=float(alpha_w),
                    gate_S_t=gate_drift_theta if gate_drift_theta is not None else None,
                    S_t=S_t,
                )
        except Exception:
            w_next = float(np.clip(w_curr, float(w_min), float(w_max)))

        metrics_t["w"] = float(w_next)
        metrics_t["D_t"] = float(D_t)
        metrics_t["M_t"] = float(M_t)
        metrics_t["S_t"] = float(S_t)
        if lag_abs_rad is not None:
            metrics_t["lag_abs_rad"] = float(lag_abs_rad)
        metrics_t["controller"] = {
            "policy": pol,
            "params": {
                "alpha_w": float(alpha_w),
                "w_min": float(w_min),
                "w_max": float(w_max),
                "a": float(a_drift),
                "b": float(b_mismatch),
                "theta_low": float(theta_low),
                "theta_high": float(theta_high),
                "delta_up": float(delta_up),
                "delta_down": float(delta_down),
                "theta_mid": float(theta_mid),
                "beta": float(beta),
                "lag_target": float(lag_target),
                "lag_window": int(lag_window),
                "kp": float(kp), "ki": float(ki), "kd": float(kd),
                "gate_drift_theta": None if gate_drift_theta is None else float(gate_drift_theta),
            }
        }

        _save_json(rdir / "metrics.json", metrics_t)

        # persist serie
        w_series.append(float(w_next))
        drift_series.append((float(D_t), float(M_t), float(S_t)))
        results_dir = _ensure_dir(out / "results")
        try:
            np.save(results_dir / "w_series.npy", np.asarray(w_series, dtype=np.float32))
            np.save(results_dir / "drift_series.npy", np.asarray(drift_series, dtype=np.float32))
        except Exception:
            pass

        # aggiorna stato
        w_curr = float(w_next)
        pi_data_prev = np.asarray(pi_hat, dtype=float)
        if isinstance(pi_hat_retrieval_vec, np.ndarray):
            pi_mem_prev = np.asarray(pi_hat_retrieval_vec, dtype=float)

    summary = {
        "outdir": str(out),
        "seed": int(seed),
        "hp": asdict(hp),
        "classes": list(map(int, classes)),
        "rounds": all_rounds,
        "exposure_counts": exposure.tolist(),
    }
    _save_json(out / "summary.json", summary)
    return summary
