
# -*- coding: utf-8 -*-
"""
exp07_mnist_runner.py — Driver Exp-07 (single-only) su dati STRUTTURATI (MNIST-like).

Caratteristiche principali
-------------------------
• Seleziona 3 classi (raccomandato per Δ₂) o in generale K = K_old + K_new.
• Binarizza le immagini in {±1}, costruisce i prototipi di classe ξ_true = sign(mean).
• Scheduler della novità (rampa) e sampling SINGLE per round con visibilità parziale dei nuovi
  archetipi fra i client (new_visibility_frac).
• Per ogni round: J_unsup → blend con memoria ebraica → propagazione → spectral cut →
  TAM + disentangling → allineamento candidati → π̂_t e TV(π̂,π).
• Salva per-round e top-level esattamente come il runner unstructured, così reporting/plots funzionano.

NOTA: Questo script NON scarica MNIST. Attende un file .npz con 'X' e 'y' (X in (M,H,W) o (M,N)).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np

# -----------------------------
# Import dalla codebase esistente
# -----------------------------
# Hyper-params e strutture
try:
    from src.unsup.config import HyperParams, TAMParams, SpectralParams, PropagationParams  # type: ignore
except Exception:  # fallback minimale per esecuzione isolata
    @dataclass
    class TAMParams:
        beta_T: float = 2.5
        lam: float = 0.2
        h_in: float = 0.1
        updates: int = 50
        noise_scale: float = 0.3
    @dataclass
    class SpectralParams:
        tau: float = 0.5
        rho: float = 0.6
        qthr: float = 0.4
    @dataclass
    class PropagationParams:
        iters: int = 20
    @dataclass
    class HyperParams:
        L: int = 3
        K: int = 3
        N: int = 784
        n_batch: int = 24
        M_total: int = 2400
        r_ex: float = 0.0   # NON usato per dati reali; lasciato per compatibilità
        w: float = 0.0
        tam: TAMParams = TAMParams()
        prop: PropagationParams = PropagationParams()
        spec: SpectralParams = SpectralParams()
        estimate_keff_method: str = "shuffle"
        ema_alpha: float = 0.0
        use_tqdm: bool = False
        mode: str = "single"

# Funzioni core (riuso dalla codebase single-mode)
try:
    from src.unsup.single_mode import (  # type: ignore
        build_unsup_J_single,
        blend_with_memory,
        propagate_J,
        eigen_cut,
        dis_check,
        estimate_keff as estimate_keff_wrapped,
    )
except Exception:
    # fallback di emergenza: interfacce compatibili richieste dal runner
    from src.unsup.functions import build_unsup_J_single, blend_with_memory, propagate_J, eigen_cut, dis_check  # type: ignore
    def estimate_keff_wrapped(J, method="shuffle", **kwargs):
        from src.unsup.functions import estimate_K_eff_from_J  # type: ignore
        Keff, evals, info = estimate_K_eff_from_J(J, method=method, **kwargs)
        return Keff, evals, info

# Metriche mixing (riuso)
try:
    from src.mixing.metrics import tv_distance, estimate_pi_hat_from_examples  # type: ignore
except Exception:
    def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
        return 0.5 * float(np.sum(np.abs(p - q)))
    def estimate_pi_hat_from_examples(xi_ref: np.ndarray, E_t: np.ndarray) -> np.ndarray:
        L, M_c, N = E_t.shape
        K = xi_ref.shape[0]
        X = E_t.reshape(L * M_c, N)
        Ov = X @ xi_ref[:K].T
        mu_hat = np.argmax(Ov, axis=1)
        counts = np.bincount(mu_hat, minlength=K).astype(float)
        return counts / (counts.sum() + 1e-9)

# Utility MNIST locali (riuso del modulo creato in precedenza)
try:
    from .mnist_utils import binarize_pm1, build_class_prototypes, select_classes  # type: ignore
except Exception:
    # Fallback minimi (se eseguito fuori package)
    def binarize_pm1(X: np.ndarray, *, threshold: Optional[float] = None) -> np.ndarray:
        Xf = X.astype(np.float32)
        if Xf.ndim == 3:
            M,H,W = Xf.shape; Xf = Xf.reshape(M, H*W)
        thr = np.median(Xf, axis=0, keepdims=True) if threshold is None else float(threshold)
        return np.where(Xf >= thr, 1.0, -1.0).astype(np.float32)
    def build_class_prototypes(X_pm1: np.ndarray, y: np.ndarray, classes: Sequence[int]) -> np.ndarray:
        cls = list(classes); K = len(cls); N = X_pm1.shape[1]
        xi = np.zeros((K, N), dtype=np.float32)
        for i,c in enumerate(cls):
            idx = np.where(y == c)[0]
            mu = X_pm1[idx].mean(axis=0, keepdims=True)
            xi[i] = np.where(mu >= 0.0, 1.0, -1.0)
        return xi
    def select_classes(X: np.ndarray, y: np.ndarray, classes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(y, np.array(classes, dtype=int))
        return X[mask], y[mask]

# Scheduler della novità (riuso; fallback se assente)
try:
    from .novelty import novelty_schedule  # type: ignore
except Exception:
    def novelty_schedule(T: int, K_old: int, K_new: int, t_intro: int, ramp_len: int,
                         alpha_max: float = 1.0, new_visibility_frac: float = 1.0) -> np.ndarray:
        """
        Rampa lineare: massa "nuovi" = α(t) ∈ [0, alpha_max]; vecchi = 1-α(t).
        Uniforme all'interno dei blocchi old/new. Restituisce (T, K).
        """
        K = int(K_old + K_new)
        P = np.zeros((T, K), dtype=np.float32)
        for t in range(T):
            if t < t_intro:
                a = 0.0
            elif t >= t_intro + max(1, ramp_len):
                a = alpha_max
            else:
                a = alpha_max * (t - t_intro + 1) / float(max(1, ramp_len))
            mass_old = max(0.0, 1.0 - a); mass_new = max(0.0, a)
            if K_old > 0:
                P[t, :K_old] = mass_old / float(K_old)
            if K_new > 0:
                P[t, K_old:] = mass_new / float(K_new)
            s = P[t].sum()
            if s <= 0:
                P[t] = 1.0 / float(K)
            else:
                P[t] /= s
        return P

# -----------------------------
# Helper di filesystem
# -----------------------------
def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _save_json(p: Union[str, Path], obj: Dict[str, Any]) -> None:
    Path(p).write_text(json.dumps(obj, indent=2))

def _round_dir(run_dir: Path, t: int) -> Path:
    rd = run_dir / f"round_{t:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd

# -----------------------------
# Allineamento candidati → true (Hungarian + flip di segno)
# -----------------------------
def _align_candidates_to_true(xi_cand: np.ndarray, xi_true: np.ndarray, K: int) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    if xi_cand.size == 0:
        return xi_true.copy().astype(np.int8)
    Ka, N = xi_cand.shape
    Kt, Nt = xi_true.shape
    if Nt != N:
        raise ValueError("Dimensioni incompatibili tra xi_cand e xi_true.")
    M = np.abs(xi_cand @ xi_true.T) / float(N)
    cost = 1.0 - M
    rI, cI = linear_sum_assignment(cost)
    order = np.argsort(M[rI, cI])[::-1]
    rI = rI[order]; cI = cI[order]
    take = min(K, min(Ka, Kt))
    xi_aligned = np.zeros((K, N), dtype=np.int8)
    used = 0
    for rr, cc in zip(rI[:take], cI[:take]):
        v = xi_cand[int(rr)]
        sgn = 1.0 if float(v @ xi_true[int(cc)]) >= 0.0 else -1.0
        xi_aligned[used] = np.where(v >= 0, 1, -1).astype(np.int8) * int(sgn)
        used += 1
    for j in range(used, K):
        xi_aligned[j] = xi_true[j]
    return xi_aligned

# -----------------------------
# Sampling SINGLE con visibilità parziale (MNIST arrays)
# -----------------------------
def _sample_round_mnist_with_visibility(
    X_pm1: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    pi_t: np.ndarray,
    *,
    L: int,
    M_c: int,
    K_old: int,
    new_visibility_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distribuisce esempi reali secondo π_t in SINGLE-mode, ma limitando la presenza
    delle classi nuove (μ >= K_old) soltanto ad una frazione di client per round.
    Ritorna E_t ∈ R^{L×M_c×N} e y_t ∈ {0,...,K-1}^{L×M_c}.
    """
    K = len(classes)
    N = X_pm1.shape[1]
    # Pre-compute indices per class (in order of 'classes')
    idx_per = [np.where(y == c)[0] for c in classes]
    for arr in idx_per:
        if arr.size == 0:
            raise ValueError("Una delle classi selezionate non ha campioni.")

    # Which clients can see new classes
    vis_count = int(np.floor(L * max(0.0, min(1.0, float(new_visibility_frac)))))
    new_enabled = np.zeros(L, dtype=bool)
    if vis_count > 0:
        new_enabled[rng.choice(np.arange(L), size=vis_count, replace=False)] = True

    # Global class counts by π_t
    total = L * M_c
    pi_t = np.asarray(pi_t, dtype=float); pi_t = pi_t / (pi_t.sum() + 1e-9)
    counts = rng.multinomial(total, pi_t)  # (K,)

    # Allocate slots (class indices) respecting visibility
    per_client_slots: List[List[Optional[int]]] = [[None] * M_c for _ in range(L)]
    for k in range(K):
        n_k = int(counts[k])
        if n_k <= 0:
            continue
        if k >= K_old:
            allowed_clients = np.where(new_enabled)[0]
            if allowed_clients.size == 0:
                allowed_clients = np.arange(L)  # conservative fallback
        else:
            allowed_clients = np.arange(L)
        # gather available slots on allowed clients
        slots = [(l, m) for l in allowed_clients for m in range(M_c) if per_client_slots[l][m] is None]
        if not slots:
            continue
        pick = min(n_k, len(slots))
        chosen = rng.choice(np.arange(len(slots)), size=pick, replace=False)
        for idx in chosen:
            l, m = slots[int(idx)]
            per_client_slots[l][m] = k

    # Fill remaining None with old classes according to remaining counts
    remaining_counts = np.array([counts[k] - sum((np.array(per_client_slots[l]) == k).sum() for l in range(L)) for k in range(K)], dtype=int)
    remaining_counts[:K_old] += remaining_counts[K_old:]; remaining_counts[K_old:] = 0
    old_pool = []
    for k in range(K_old):
        old_pool += [k] * int(max(0, remaining_counts[k]))
    if not old_pool:
        old_pool = [int(k) for k in range(K_old)] * max(1, (L * M_c) // max(1, K_old))

    optr = 0
    for l in range(L):
        for m in range(M_c):
            if per_client_slots[l][m] is None:
                per_client_slots[l][m] = old_pool[optr % len(old_pool)]
                optr += 1

    # Draw concrete examples from the per-class pools (with/without replacement)
    # Pre-draw shuffled indices to avoid repetition biases
    drawn_per: List[np.ndarray] = []
    for k, arr in enumerate(idx_per):
        need_k = sum(1 for l in range(L) for m in range(M_c) if per_client_slots[l][m] == k)
        if need_k <= arr.size:
            choose = rng.choice(arr, size=need_k, replace=False)
        else:
            choose = rng.choice(arr, size=need_k, replace=True)
        drawn_per.append(choose)

    # Materialize E_t and y_t
    E_t = np.empty((L, M_c, N), dtype=np.float32)
    y_t = np.empty((L, M_c), dtype=np.int32)
    ptr = [0] * K
    for l in range(L):
        for m in range(M_c):
            k = int(per_client_slots[l][m])
            idx = drawn_per[k][ptr[k]]; ptr[k] += 1
            E_t[l, m] = X_pm1[idx]
            y_t[l, m] = k
    return E_t, y_t

# -----------------------------
# Funzione principale
# -----------------------------
def run_exp07_mnist(
    run_dir: Union[str, Path],
    hp: HyperParams,
    *,
    data_npz: Union[str, Path],
    classes: Sequence[int],
    K_old: int,
    t_intro: int,
    ramp_len: int,
    alpha_max: float = 1.0,
    new_visibility_frac: float = 1.0,
    seed: int = 0,
    save_mats: bool = True,
    save_spectrum: bool = True,
) -> Dict[str, Any]:
    """
    Esegue Exp-07 (single) su MNIST-like per una lista di classi (tipicamente K=3).
    Si attende un .npz con 'X' e 'y' (X ∈ R^{M×H×W} o R^{M×N}, y ∈ {0..9}).
    """
    run_dir = _ensure_dir(run_dir)
    rng = np.random.default_rng(seed)
    classes = list(map(int, classes))
    K = len(classes)
    assert hp.K == K, f"hp.K={hp.K} deve coincidere con len(classes)={K}"
    assert K_old < K and K_old > 0, "K_old deve essere in [1, K-1]"

    # Carica dataset
    data = np.load(data_npz)
    X, y = data["X"], data["y"]
    if X.ndim == 3:
        M,H,W = X.shape; N = H*W
    else:
        M,N = X.shape
    # Se necessario, aggiorna hp.N
    if hp.N != N:
        hp = HyperParams(**{**asdict(hp), "N": int(N)})
    # Selezione classi e binarizzazione
    Xc, yc = select_classes(X, y, classes)
    X_pm1 = binarize_pm1(Xc)
    xi_true = build_class_prototypes(X_pm1, yc, classes).astype(np.int8)  # (K,N)
    np.save(Path(run_dir) / "xi_true.npy", xi_true)
    # Mapping classi salvato a parte
    _save_json(Path(run_dir) / "classes.json", {"classes": classes, "K": K, "K_old": int(K_old)})

    # Schedule della novità
    pis = novelty_schedule(
        T=int(hp.n_batch),
        K_old=int(K_old),
        K_new=int(K - K_old),
        t_intro=int(t_intro),
        ramp_len=int(ramp_len),
        alpha_max=float(alpha_max),
        new_visibility_frac=float(new_visibility_frac),
    ).astype(np.float32)
    np.save(Path(run_dir) / "pis.npy", pis)

    # M_c da budget
    M_c = int(math.ceil(hp.M_total / float(hp.L * hp.n_batch)))

    # Memoria ebraica (ξ_ref allineati round-1)
    xi_ref: Optional[np.ndarray] = None

    # Exposure counts (per classe, in ordine di 'classes')
    exposure_counts = np.zeros(K, dtype=np.int64)

    for t in range(hp.n_batch):
        rd = _round_dir(Path(run_dir), t)

        # Dati del round con visibilità parziale dei nuovi
        E_t, y_t = _sample_round_mnist_with_visibility(
            X_pm1, yc, classes, pis[t],
            L=hp.L, M_c=M_c, K_old=int(K_old),
            new_visibility_frac=float(new_visibility_frac),
            rng=rng,
        )
        # Aggiorna exposure
        vals, cnts = np.unique(y_t, return_counts=True)
        exposure_counts[vals] += cnts

        # 1) J_unsup (single) — dai soli esempi del round
        J_unsup, M_eff = build_unsup_J_single(E_t, K=hp.K)

        # 2) Blend con memoria
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)

        # 3) Propagazione (pseudo-inversa / KS-like)
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)

        # 4) Cut spettrale e spettro
        try:
            from src.unsup.single_mode import spectral_cut  # type: ignore
            spec_out = spectral_cut(J_KS, tau=hp.spec.tau, return_info=True)
            if len(spec_out) == 3:
                V_sel, k_from_cut, info_spec = spec_out
            else:
                V_sel, k_from_cut = spec_out
                info_spec = {"evals": None}
        except Exception:
            ec = eigen_cut(J_KS, tau=hp.spec.tau, return_info=True)
            if len(ec) == 3:
                V_sel, k_from_cut, info_spec = ec
            else:
                V_sel, k_from_cut = ec
                info_spec = {"evals": None}

        # 5) K_eff (shuffle|mp)
        if hp.estimate_keff_method == "mp":
            K_eff, evals, info_keff = estimate_keff_wrapped(J_KS, method="mp", M_eff=M_eff)
        else:
            K_eff, evals, info_keff = estimate_keff_wrapped(J_KS, method="shuffle")
        if info_spec.get("evals", None) is None and evals is not None:
            info_spec["evals"] = evals

        # 6) TAM + disentangling, poi allineamento candidati → true
        xi_r, _m = dis_check(
            V=V_sel, K=hp.K, L=hp.L,
            J_rec=J_rec, JKS_iter=J_KS,
            xi_true=xi_true, tam=hp.tam, spec=hp.spec,
            show_progress=False, max_attempts=10,
        )
        xi_r = np.asarray(xi_r, dtype=float)
        xi_aligned = _align_candidates_to_true(xi_r, xi_true, K=hp.K)

        # 7) π̂_t dai soli esempi del round + TV(π̂,π)
        pi_hat = estimate_pi_hat_from_examples(xi_true, E_t)
        pi_true_t = pis[t] / (pis[t].sum() + 1e-9)
        TV_t = tv_distance(pi_hat, pi_true_t)

        # 8) Salvataggi per round
        metrics = {
            "round": int(t),
            "K_eff": int(K_eff),
            "TV_pi": float(TV_t),
            "pi_hat": pi_hat.tolist(),
            "pi_true": pi_true_t.tolist(),
            "keff_info": {
                "method": hp.estimate_keff_method,
                "eigvals": None if info_spec.get("evals", None) is None else np.asarray(info_spec["evals"], dtype=float).tolist(),
            },
            "shapes": {
                "V_sel": list(np.shape(V_sel)),
                "J_unsup": list(np.shape(J_unsup)),
                "J_rec": list(np.shape(J_rec)),
                "J_KS": list(np.shape(J_KS)),
                "xi_aligned": list(np.shape(xi_aligned)),
            },
        }
        _save_json(rd / "metrics.json", metrics)
        np.save(rd / "xi_aligned.npy", xi_aligned.astype(np.int8))
        if save_mats:
            np.save(rd / "J_unsup.npy", np.asarray(J_unsup, dtype=np.float32))
            np.save(rd / "J_rec.npy", np.asarray(J_rec, dtype=np.float32))
            np.save(rd / "J_KS.npy", np.asarray(J_KS, dtype=np.float32))
        if save_spectrum:
            np.save(rd / "V_sel.npy", np.asarray(V_sel, dtype=np.float32))
            if info_spec.get("evals", None) is not None:
                np.save(rd / "eigs_sel.npy", np.asarray(info_spec["evals"], dtype=np.float32))

        # 9) Memoria per il round successivo
        xi_ref = xi_aligned.copy()

    # Top-level salvataggi
    np.save(Path(run_dir) / "exposure_counts.npy", exposure_counts.astype(np.int64))

    meta = {
        "K": int(K),
        "K_old": int(K_old),
        "K_new": int(K - K_old),
        "t_intro": int(t_intro),
        "ramp_len": int(ramp_len),
        "alpha_max": float(alpha_max),
        "new_visibility_frac": float(new_visibility_frac),
        "hp": {
            "L": int(hp.L), "N": int(hp.N), "n_batch": int(hp.n_batch),
            "M_total": int(hp.M_total), "w": float(hp.w),
            "prop.iters": int(hp.prop.iters),
            "spec.tau": float(hp.spec.tau),
            "estimate_keff_method": str(hp.estimate_keff_method),
        },
        "classes": classes,
    }
    _save_json(Path(run_dir) / "meta_exp07_mnist.json", meta)

    return {
        "run_dir": str(run_dir),
        "K": int(K),
        "K_old": int(K_old),
        "K_new": int(K - K_old),
        "n_rounds": int(hp.n_batch),
        "paths": {
            "xi_true": str(Path(run_dir) / "xi_true.npy"),
            "pis": str(Path(run_dir) / "pis.npy"),
            "exposure_counts": str(Path(run_dir) / "exposure_counts.npy"),
            "meta": str(Path(run_dir) / "meta_exp07_mnist.json"),
            "classes": str(Path(run_dir) / "classes.json"),
        },
    }

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Directory di output della run")
    ap.add_argument("--data-npz", type=str, required=True, help="Percorso a file .npz con X, y")
    ap.add_argument("--classes", type=str, default="0,1,2", help="Lista di classi CSV (es. 0,1,2)")
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--K_old", type=int, default=2)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--M_total", type=int, default=2400)
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--keff", type=str, default="shuffle", choices=["shuffle", "mp"])
    ap.add_argument("--t_intro", type=int, default=8)
    ap.add_argument("--ramp_len", type=int, default=6)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--vis_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = [int(x) for x in args.classes.split(",") if x.strip() != ""]
    K = len(classes)

    # hp (N sarà ricavato dal dataset al volo)
    hp = HyperParams(
        L=args.L, K=K, N=784, n_batch=args.T, M_total=args.M_total,
        r_ex=0.0, w=args.w, estimate_keff_method=args.keff,
        prop=PropagationParams(iters=args.iters),
        spec=SpectralParams(tau=args.tau),
    )
    out = run_exp07_mnist(
        run_dir=args.out, hp=hp, data_npz=args.data_npz, classes=classes,
        K_old=args.K_old, t_intro=args.t_intro, ramp_len=args.ramp_len,
        alpha_max=args.alpha_max, new_visibility_frac=args.vis_frac,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
