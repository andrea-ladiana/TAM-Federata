# -*- coding: utf-8 -*-
"""
Exp-06 (single-only) — pipeline core (dataset non strutturato).

Questo modulo implementa un motore round-by-round che:
  1) genera (o riceve) un dataset sintetico binario in {±1}, con mixing schedule pis (T,K)
  2) per ogni round t:
        - stima J_unsup(t) (single) e blend con memoria ebraica (J_hebb(xi_prev)) pesata da w
        - propaga J -> J_KS (pseudo-inversa iterativa)
        - taglio spettrale + disentangling TAM -> xi_r(t)
        - allinea i candidati a xi_true (greedy senza SciPy) e aggiorna xi_ref (per il blend successivo)
        - calcola metriche base (coverage, K_eff MP/shuffle, pi_hat, TV)
        - valuta magnetizzazione Hopfield (post-hoc) su J_rec(t)
  3) salva tutti gli artefatti in outdir/round_{t}/

Requisiti: riuso dei moduli presenti in codebase.txt:
- config.HyperParams, TAMParams, SpectralParams
- data.new_round_single, compute_round_coverage, count_exposures
- estimators.build_unsup_J_single, blend_with_memory
- functions.propagate_J, estimate_K_eff_from_J
- dynamics.eigen_cut, dis_check
- hopfield_eval.run_or_load_hopfield_eval

Nota: la costruzione del dataset supporta un mixing esplicito via `pis` (T,K).
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
from src.unsup.config import HyperParams  # :contentReference[oaicite:1]{index=1}
from src.unsup.data import new_round_single, compute_round_coverage, count_exposures  # :contentReference[oaicite:2]{index=2}
from src.unsup.estimators import build_unsup_J_single, blend_with_memory  # :contentReference[oaicite:3]{index=3}
from src.unsup.functions import propagate_J, estimate_K_eff_from_J, gen_patterns  # :contentReference[oaicite:4]{index=4}
from src.unsup.dynamics import dis_check  # :contentReference[oaicite:5]{index=5}
from src.unsup.spectrum import estimate_keff as _estimate_keff_spectrum
from src.unsup.hopfield_eval import run_or_load_hopfield_eval  # :contentReference[oaicite:6]{index=6}
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
# helper: allineamento greedy
# ----------------------------
def _align_greedy_sign(xi_r: np.ndarray, xi_true: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Allinea i candidati xi_r ai veri archetipi xi_true massimizzando l'overlap assoluto
    con una procedura greedy (evita dipendenze SciPy). Applica anche il flip di segno.

    Returns
    -------
    xi_aligned : (R, N)
    match      : dict idx_r -> mu_true
    """
    if xi_r.size == 0:
        return xi_r, {}
    R, N = xi_r.shape
    K = xi_true.shape[0]
    S = np.abs(xi_r @ xi_true.T) / float(N)  # (R, K)
    S = S.copy()
    used_r = set()
    used_mu = set()
    match: Dict[int, int] = {}

    while len(used_r) < min(R, K):
        # trova il massimo residuo
        idx = np.unravel_index(np.argmax(S, axis=None), S.shape)
        r, mu = int(idx[0]), int(idx[1])
        if r in used_r or mu in used_mu:
            # invalida la cella e continua
            S[r, mu] = -np.inf
            continue
        # firma: se <xi_r, xi_true[mu]> < 0 => flip segno
        if float(np.dot(xi_r[r], xi_true[mu])) < 0.0:
            xi_r[r] = -xi_r[r]
        match[r] = mu
        used_r.add(r)
        used_mu.add(mu)
        # invalida r e mu
        S[r, :] = -np.inf
        S[:, mu] = -np.inf

    return xi_r, match


# ----------------------------
# helper: stima pi_hat da esempi
# ----------------------------
def _estimate_pi_hat_from_examples(xi_ref: np.ndarray, E_t: np.ndarray) -> np.ndarray:
    """
    Classifica ogni esempio in E_t sul set di riferimenti xi_ref per overlapping massimo
    e restituisce le frequenze normalizzate (pi_hat) su K archetipi.

    E_t : (L, M_c, N)  ;  xi_ref : (S, N) con S>=K (useremo solo i migliori K se S>K)
    """
    L, M_c, N = E_t.shape
    S = xi_ref.shape[0]
    # se S > K, prendiamo i K primi (l'ordine è già 'promosso' dal pruning/greedy)
    K = min(S,  max(1, S))
    ref = xi_ref[:K]  # (K, N)
    X = E_t.reshape(L * M_c, N)  # (LM, N)
    # overlaps (LM, K)
    Ov = X @ ref.T
    mu_hat = np.argmax(Ov, axis=1)
    counts = np.bincount(mu_hat, minlength=K).astype(float)
    if counts.sum() <= 0:
        return np.ones(K, dtype=float) / float(K)
    return counts / counts.sum()


# ----------------------------
# helper: generatore ETA con schedule pis
# ----------------------------
def _gen_dataset_with_schedule(
    xi_true: np.ndarray,
    pis: np.ndarray,   # (T, K)
    M_total: int,
    r_ex: float,
    L: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera ETA, labels in SINGLE-mode rispettando la mixing-schedule pis per round.

    Returns
    -------
    ETA    : (L, T, M_c, N) in float32
    labels : (L, T, M_c)    in int32
    """
    T, K = pis.shape
    N = xi_true.shape[1]
    M_c = int(math.ceil(M_total / float(L * T)))
    ETA = np.zeros((L, T, M_c, N), dtype=np.float32)
    labels = np.zeros((L, T, M_c), dtype=np.int32)
    p_keep = 0.5 * (1.0 + float(r_ex))

    for t in range(T):
        pi_t = np.asarray(pis[t], dtype=float)
        pi_t = np.maximum(pi_t, 0.0)
        if pi_t.sum() <= 0:
            pi_t = np.ones(K, dtype=float) / float(K)
        else:
            pi_t = pi_t / pi_t.sum()

        for l in range(L):
            mus = rng.choice(K, size=M_c, replace=True, p=pi_t).astype(int)
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0).astype(np.float32)
            xi_sel = xi_true[mus].astype(np.float32)  # (M_c, N)
            ETA[l, t] = (chi * xi_sel).astype(np.float32)
            labels[l, t] = mus.astype(np.int32)

    return ETA, labels


# ----------------------------
# main: run_seed_synth
# ----------------------------
def run_seed_synth(
    hp: HyperParams,
    seed: int,
    *,
    outdir: str,
    pis: Optional[np.ndarray] = None,           # (T,K) mixing schedule; se None => uniforme
    xi_true: Optional[np.ndarray] = None,       # (K,N); se None => gen_patterns
    eval_hopfield_every: int = 1,               # 1=ogni round, 0=disabilitato, n>1 = ogni n round
    # --- controllo w ---
    w_policy: str = "pctrl",                    # {fixed, threshold, sigmoid, pctrl}
    w_init: Optional[float] = None,             # default: hp.w se None
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
    Esegue Exp-06 (single-only) su dataset sintetico.

    Parametri
    ---------
    hp : HyperParams
    seed : int
    outdir : directory dove salvare artefatti
    pis : opzionale (T,K)  mixing schedule
    xi_true : opzionale (K,N) archetipi binari
    eval_hopfield_every : frequenza valutazioni Hopfield (1=ogni round)

    Returns
    -------
    summary : dict con percorsi, metriche globali e set di artefatti creati.
    """
    rng = np.random.default_rng(int(hp.seed_base + seed))
    out = _ensure_dir(outdir)
    _save_json(out / "hyperparams.json", asdict(hp))

    # archetipi veri
    if xi_true is None:
        xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=int)  # (K,N)  :contentReference[oaicite:7]{index=7}
    else:
        assert xi_true.shape == (hp.K, hp.N), "xi_true deve essere (K,N) coerente con hp."

    # mixing schedule
    T = int(hp.n_batch)
    if pis is None:
        pis = np.ones((T, hp.K), dtype=float) / float(hp.K)

    # dataset round-wise
    ETA, labels = _gen_dataset_with_schedule(
        xi_true=xi_true,
        pis=pis,
        M_total=hp.M_total,
        r_ex=hp.r_ex,
        L=hp.L,
        rng=rng,
    )
    np.save(out / "xi_true.npy", xi_true.astype(np.int8))
    np.save(out / "ETA.shape.npy", np.array(ETA.shape, dtype=int))
    np.save(out / "labels.npy", labels.astype(np.int16))
    np.save(out / "pis.npy", pis.astype(np.float32))

    # esposizioni globali (per correlazione Hopfield)
    exposure = count_exposures(labels, K=hp.K)  # :contentReference[oaicite:8]{index=8}
    np.save(out / "exposure_counts.npy", exposure.astype(np.int32))

    # stato per il blend
    xi_ref_prev: Optional[np.ndarray] = None
    # controllo w: stato corrente e serie per logging
    w_curr: float = float(hp.w if w_init is None else w_init)
    w_series: List[float] = []
    drift_series: List[Tuple[float, float, float]] = []  # (D_t, M_t, S_t)
    # sequenze per lag |phi| (solo K=3)
    pi_data_hist: List[np.ndarray] = []
    pi_mem_hist: List[np.ndarray] = []
    lag_abs_hist: List[float] = []
    # memorizza per drift
    pi_data_prev: Optional[np.ndarray] = None
    pi_mem_prev: Optional[np.ndarray] = None

    all_rounds = []
    for t in range(T):
        rdir = _ensure_dir(out / f"round_{t:03d}")
        E_t = new_round_single(ETA, t)  # (L, M_c, N)  :contentReference[oaicite:9]{index=9}
        cov_t = float(compute_round_coverage(labels[:, t, :], K=hp.K))  # :contentReference[oaicite:10]{index=10}

        # stima unsup + blend single
        J_unsup, M_eff = build_unsup_J_single(E_t, K=hp.K)  # :contentReference[oaicite:11]{index=11}
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref_prev, w=float(np.clip(w_curr, 0.0, 1.0)))  # :contentReference[oaicite:12]{index=12}
        np.save(rdir / "J_unsup.npy", J_unsup.astype(np.float32))
        np.save(rdir / "J_rec.npy", J_rec.astype(np.float32))

        # propagazione + cut spettrale
        J_KS = propagate_J(J_rec, iters=hp.prop.iters, eps=hp.prop.eps, tol=1e-8)  # :contentReference[oaicite:13]{index=13}
        np.save(rdir / "J_KS.npy", J_KS.astype(np.float32))

        # Selezione autovettori per TAM: preferisci K_eff (shuffle/MP),
        # con fallback a top-K se la soglia fissa restituirebbe < K vettori.
        try:
            # 1) eigendecomp ordinata (decrescente)
            evals, evecs = np.linalg.eigh((J_KS + J_KS.T) * 0.5)
            order = np.argsort(evals)[::-1]
            evals_desc = evals[order]
            evecs_desc = evecs[:, order]

            # 2) stima K_eff coerente con i report
            K_eff_est, keep_mask_keff, _info = _estimate_keff_spectrum(J_KS, method=hp.estimate_keff_method, M_eff=M_eff)
            keep_mask_keff = np.asarray(keep_mask_keff, dtype=bool).reshape(-1)
            keff_threshold = float(_info.get("threshold", np.nan)) if isinstance(_info, dict) else float("nan")
            # 2b) maschera da soglia fissa tau (per robustezza/back-compat)
            keep_mask_tau = evals_desc > float(hp.spec.tau)

            # 3) costruisci maschera finale: almeno K autovettori
            K_target = int(max(1, hp.K))
            if keep_mask_keff.size != evals_desc.size:
                # fallback di sicurezza su top-K
                keep_idx = np.arange(min(K_target, evals_desc.size))
            else:
                union_mask = np.logical_or(keep_mask_keff, keep_mask_tau)
                n_union = int(np.sum(union_mask))
                if n_union >= K_target:
                    keep_idx = np.where(union_mask)[0]
                else:
                    # integra top‑K per garantire almeno K componenti
                    base = list(np.where(union_mask)[0])
                    extra = [i for i in range(evals_desc.size) if i not in base][:max(0, K_target - n_union)]
                    keep_idx = np.array(base + extra, dtype=int)
            # Limita a massimo K componenti (in ordine di importanza spettrale)
            if keep_idx.size > K_target:
                keep_idx = keep_idx[:K_target]

            V = evecs_desc[:, keep_idx].T.astype(np.float32)
            vals_sel = evals_desc[keep_idx].astype(np.float32)
            eig_sel_info = {
                "K_eff_est": int(K_eff_est),
                "keff_threshold": keff_threshold,
                "tau": float(hp.spec.tau),
                "n_keff_mask": int(np.sum(keep_mask_keff)) if keep_mask_keff.size == evals_desc.size else None,
                "n_tau_mask": int(np.sum(keep_mask_tau)),
                "n_final": int(V.shape[0]),
            }
        except Exception:
            # Ultimo fallback: usa tutti gli autovettori (potrebbe essere costoso ma robusto)
            evals, evecs = np.linalg.eigh((J_KS + J_KS.T) * 0.5)
            order = np.argsort(evals)[::-1]
            evals_desc = evals[order]
            evecs_desc = evecs[:, order]
            keep_idx = np.arange(min(int(hp.K), evals_desc.size))
            V = evecs_desc[:, keep_idx].T.astype(np.float32)
            vals_sel = evals_desc[keep_idx].astype(np.float32)
            eig_sel_info = {
                "K_eff_est": None,
                "keff_threshold": None,
                "tau": float(hp.spec.tau),
                "n_keff_mask": None,
                "n_tau_mask": int(np.sum(evals_desc > float(hp.spec.tau))),
                "n_final": int(V.shape[0]),
            }

        np.save(rdir / "eigs_sel.npy", vals_sel)
        np.save(rdir / "V_sel.npy", V)

        # disentangling TAM + pruning
        xi_r, m = dis_check(
            V, K=hp.K, L=hp.L,
            J_rec=J_rec, JKS_iter=J_KS,
            xi_true=xi_true,
            tam=hp.tam, spec=hp.spec,
            show_progress=hp.use_tqdm,
        )  # :contentReference[oaicite:15]{index=15}
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
        )  # :contentReference[oaicite:16]{index=16}

        # stima pi_hat e TV vs pi_t
        try:
            # Use the ground-truth references `xi_true` to estimate pi_hat so the
            # returned vector always has length K (hp.K). Using `xi_aligned` can
            # produce a shorter vector when pruning/disentangling removes some
            # references, which breaks downstream code that expects shape (K,).
            pi_hat = _estimate_pi_hat_from_examples(xi_ref=xi_true, E_t=E_t)
        except Exception:
            pi_hat = np.ones(hp.K) / float(hp.K)
        pi_t = pis[t] / float(np.sum(pis[t])) if np.sum(pis[t]) > 0 else np.ones(hp.K) / float(hp.K)
        TV_t = 0.5 * float(np.sum(np.abs(pi_hat - pi_t)))

        # metriche round (iniziali)
        metrics_t = {
            "coverage": cov_t,
            "K_eff": int(K_eff),
            "M_eff": int(M_eff),
            "n_eigs_sel": int(V.shape[0]),
            "n_candidates": int(xi_r.shape[0]),
            "eig_selection": eig_sel_info,
            "TV_pi": TV_t,
            # Mantieni la stima data-driven e anche sotto alias esplicito
            "pi_hat": pi_hat.tolist(),
            "pi_hat_data": pi_hat.tolist(),
            "pi_true": pi_t.tolist(),
        }
        all_rounds.append(metrics_t)

        # valutazione Hopfield (facoltativa) e costruzione di pi_hat_retrieval
        pi_hat_retrieval_vec = None
        if eval_hopfield_every and ((t % int(eval_hopfield_every)) == 0):
            # Don't pre-create the directory; let run_or_load_hopfield_eval decide
            hop_dir = rdir / "hopfield"
            try:
                results_h = run_or_load_hopfield_eval(
                    output_dir=str(hop_dir),
                    J_server=J_rec,
                    xi_true=xi_true,
                    exposure_counts=exposure,
                    beta=3.0, updates=30, reps_per_archetype=32, start_overlap=0.3,
                    force_run=True, save=True, stochastic=True,
                )  # :contentReference[oaicite:17]{index=17}
            except Exception:
                results_h = None

            # Se il runner non restituisce un dict, prova a leggere un JSON salvato su disco
            if not isinstance(results_h, dict):
                try:
                    json_files = sorted([p for p in hop_dir.glob("*.json") if p.is_file()],
                                        key=lambda p: p.stat().st_mtime, reverse=True)
                    for jp in json_files:
                        try:
                            obj = json.loads(jp.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        if any(k in obj for k in ("magnetization_by_mu", "m_by_mu", "mag_by_mu")):
                            results_h = obj
                            break
                except Exception:
                    pass

            # Estrai magnetizzazioni by-mu e mappa su simplesso
            m_arr = None
            if isinstance(results_h, dict):
                for k in ("magnetization_by_mu", "m_by_mu", "mag_by_mu"):
                    if k in results_h:
                        try:
                            m_arr = np.asarray(results_h[k])
                            break
                        except Exception:
                            m_arr = None
            # Fallback: file NPZ salvato dal runner
            if m_arr is None:
                npz_path = hop_dir / "magnetization_by_mu.npz"
                if npz_path.exists():
                    try:
                        loaded = np.load(npz_path)
                        # chiavi attese: m_0, m_1, ...
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

        # Se Hopfield non è stato eseguito in questo round, prova a riusare il valore precedente (continuità)
        if pi_hat_retrieval_vec is None:
            try:
                prev = rdir.parent / f"round_{t-1:03d}" / "metrics.json"
                prev_metrics = json.loads(prev.read_text()) if (t > 0 and prev.exists()) else None
            except Exception:
                prev_metrics = None
            if isinstance(prev_metrics, dict) and isinstance(prev_metrics.get("pi_hat_retrieval", None), list):
                try:
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
                pi_data_t=pi_t,
                pi_data_tm1=pi_data_prev,
                pi_mem_tm1=pi_mem_prev,
                a=float(a_drift), b=float(b_mismatch),
            )
            D_t, M_t, S_t = float(drift["D_t"]), float(drift["M_t"]), float(drift["S_t"])
        except Exception:
            D_t = M_t = S_t = 0.0

        # lag |phi| su finestra scorrevole (K=3)
        lag_abs_rad = None
        try:
            # pi_mem corrente: se disponibile la retrieval, altrimenti mantieni prev
            pi_mem_curr = None
            if isinstance(pi_hat_retrieval_vec, np.ndarray) and pi_hat_retrieval_vec.shape == (hp.K,):
                pi_mem_curr = pi_hat_retrieval_vec
            elif isinstance(pi_mem_prev, np.ndarray):
                pi_mem_curr = pi_mem_prev

            if pi_mem_curr is not None:
                pi_data_hist.append(np.asarray(pi_t, dtype=float))
                pi_mem_hist.append(np.asarray(pi_mem_curr, dtype=float))
                # finestra ultimi W punti
                W = int(max(1, lag_window))
                if len(pi_data_hist) >= 2 and hp.K == 3:
                    d_win = np.stack(pi_data_hist[-W:], axis=0)
                    m_win = np.stack(pi_mem_hist[-W:], axis=0)
                    la = lag_and_amplitude(d_win, m_win)
                    lag_abs_rad = float(abs(la.get("lag_radians", 0.0)))
                elif hp.K == 3 and pi_mem_curr is not None:
                    # stima istantanea via differenza di fase su embed 2D
                    xyd = simplex_embed_2d(np.asarray(pi_t, dtype=float))
                    xym = simplex_embed_2d(np.asarray(pi_mem_curr, dtype=float))
                    th_d = float(np.arctan2(xyd[1], xyd[0]))
                    th_m = float(np.arctan2(xym[1], xym[0]))
                    dphi = float(th_d - th_m)
                    # porta in [-pi, pi]
                    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
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
            else:  # pctrl (default)
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
            # in caso di problemi, mantieni w corrente
            w_next = float(np.clip(w_curr, float(w_min), float(w_max)))

        # logging aggiuntivo
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

        # persist serie round-wise
        w_series.append(float(w_next))
        drift_series.append((float(D_t), float(M_t), float(S_t)))
        results_dir = _ensure_dir(out / "results")
        try:
            np.save(results_dir / "w_series.npy", np.asarray(w_series, dtype=np.float32))
            np.save(results_dir / "drift_series.npy", np.asarray(drift_series, dtype=np.float32))
        except Exception:
            pass

        # aggiorna stato per round successivo
        w_curr = float(w_next)
        pi_data_prev = np.asarray(pi_t, dtype=float)
        if isinstance(pi_hat_retrieval_vec, np.ndarray):
            pi_mem_prev = np.asarray(pi_hat_retrieval_vec, dtype=float)
        # altrimenti lascia il precedente

    summary = {
        "outdir": str(out),
        "seed": int(seed),
        "hp": asdict(hp),
        "rounds": all_rounds,
        "exposure_counts": exposure.tolist(),
    }
    _save_json(out / "summary.json", summary)
    return summary
