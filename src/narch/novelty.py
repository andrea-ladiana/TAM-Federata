# -*- coding: utf-8 -*-
"""
novelty.py — Exp-07 (single-only) utilities for novelty tracking and Hopfield-based magnetisations.

This module REUSES your existing codebase (no "extend" logic anywhere) to:
  • Track K_eff(t), spectral gap at the boundary K_old, and detection latency.
  • Compute per-round Hopfield magnetisations using a HEBB matrix built from ξ_ref (aligned) —
    this satisfies the requirement "magnetizzazioni calcolate con rete di Hopfield con sinapsi
    inizializzate con matrice di Hebb opportuna".
  • Aggregate series and convenience helpers for downstream reporting.

It expects the standard round folders produced by your pipelines:
    run_dir/
      ├─ xi_true.npy
      ├─ pis.npy  (optional, for plotting π_true)
      ├─ round_000/
      │    ├─ metrics.json   (contains K_eff, TV_pi, keff_info.eigvals, pi_hat, pi_true, ...)
      │    ├─ xi_aligned.npy (aligned & sign-fixed candidates for that round)
      │    └─ J_KS.npy, eigs_sel.npy, V_sel.npy (optional)
      ├─ round_001/
      └─ ...

All functions operate in "single" setting only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import json
import numpy as np

# --- Reuse from the provided codebase ---
try:
    from src.unsup.hopfield_eval import run_or_load_hopfield_eval  # type: ignore
except Exception:  # pragma: no cover - allow import outside the project layout
    run_or_load_hopfield_eval = None  # will raise if actually used without codebase

# Prefer the versions in your "mixing.metrics" if present.
try:
    from .metrics import tv_distance, estimate_pi_hat_from_examples  # type: ignore
except Exception:  # pragma: no cover
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

# -----------------------------------------------------------------------------
# Small filesystem helpers
# -----------------------------------------------------------------------------
def _list_round_dirs(run_dir: str | Path) -> List[Path]:
    rdir = Path(run_dir)
    if not rdir.exists():
        return []
    ds = [p for p in rdir.iterdir() if p.is_dir() and p.name.startswith("round_")]
    def _key(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 10**9
    return sorted(ds, key=_key)

def _read_json(p: str | Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())

# -----------------------------------------------------------------------------
# Core math helpers
# -----------------------------------------------------------------------------
def hebb_J(xi: np.ndarray) -> np.ndarray:
    """
    Hebbian synaptic matrix J = (1/N) * ξᵀ ξ from binary archetypes ξ (S,N).
    """
    if xi.ndim != 2:
        raise ValueError("xi must be a 2D array (S, N).")
    xi_f = xi.astype(np.float32)
    N = xi_f.shape[1]
    return (xi_f.T @ xi_f) / float(N)

def spectral_gap_at_boundary(eigvals: np.ndarray, K_old: int, *, relative: bool = True) -> float:
    """
    gap = λ_{Kold-1} - λ_{Kold} on a DESCENDING-sorted spectrum.
    If relative=True: divide by |λ_{Kold-1}| to obtain a scale-free drop (cf. report §18).
    """
    ev = np.asarray(eigvals, dtype=float).ravel()
    if ev.size < (K_old + 1) or K_old <= 0:
        return float("nan")
    lam_km1 = float(ev[K_old - 1])
    lam_k = float(ev[K_old])
    gap = lam_km1 - lam_k
    if relative:
        denom = abs(lam_km1) if abs(lam_km1) > 1e-9 else 1.0
        return float(gap / denom)
    return float(gap)

def detect_novelty_round(keff_series: np.ndarray, K_old: int, *, detect_patience: int = 2) -> Optional[int]:
    """
    First t with K_eff(t) ≥ K_old+1 that persists for 'detect_patience' consecutive rounds.
    """
    ks = np.asarray(keff_series, dtype=int).ravel()
    target = K_old + 1
    run = 0
    for t, k in enumerate(ks):
        run = (run + 1) if (k >= target) else 0
        if run >= max(1, int(detect_patience)):
            return t
    return None

# -----------------------------------------------------------------------------
# Hopfield magnetisation (HEBB initialisation) per round
# -----------------------------------------------------------------------------
@dataclass
class HopfieldParams:
    beta: float = 3.0
    updates: int = 30
    reps_per_archetype: int = 32
    start_overlap: float = 0.3
    stochastic: bool = True
    frequency: int = 1   # evaluate every n rounds (1 = every round)

def _eval_hopfield_hebb_for_round(
    round_dir: Path,
    *,
    xi_true: np.ndarray,
    xi_ref_for_hebb: Optional[np.ndarray],
    exposure_counts: Optional[np.ndarray] = None,
    hp: HopfieldParams = HopfieldParams(),
) -> Optional[Dict[str, Any]]:
    """
    Build J_hebb = (1/N) ξ_refᵀ ξ_ref using per-round aligned references (if available),
    then run the standard hopfield runner with that J. Returns the runner's dict.
    """
    if run_or_load_hopfield_eval is None:
        raise RuntimeError("run_or_load_hopfield_eval not importable; ensure codebase is on PYTHONPATH.")
    # prefer aligned candidates; fallback to ground-truth archetypes if missing
    if xi_ref_for_hebb is None or xi_ref_for_hebb.size == 0:
        xi_ref_for_hebb = xi_true
    J_hebb = hebb_J(xi_ref_for_hebb)
    hop_dir = round_dir / "hopfield_hebb"
    hop_dir.mkdir(parents=True, exist_ok=True)
    res = run_or_load_hopfield_eval(
        output_dir=str(hop_dir),
        J_server=J_hebb,
        xi_true=xi_true,
        exposure_counts=exposure_counts,
        beta=float(hp.beta),
        updates=int(hp.updates),
        reps_per_archetype=int(hp.reps_per_archetype),
        start_overlap=float(hp.start_overlap),
        force_run=True,
        save=True,
        stochastic=bool(hp.stochastic),
    )
    if isinstance(res, dict):
        res.setdefault("_meta", {})
        res["_meta"]["round_dir"] = str(round_dir)
        res["_meta"]["mode"] = "hebb"
    return res if isinstance(res, dict) else None

# -----------------------------------------------------------------------------
# Aggregation across rounds
# -----------------------------------------------------------------------------
@dataclass
class SeriesResult:
    T: int
    K: int
    K_old: int
    keff: np.ndarray           # (T,)
    gap: np.ndarray            # (T,)
    TV: np.ndarray             # (T,)
    L1: np.ndarray             # (T,)
    m_old: np.ndarray          # (T,)  mean Mattis overlap on old block
    m_new: np.ndarray          # (T,)  mean Mattis overlap on new block
    pi_hat: Optional[np.ndarray] = None  # (T,K)
    pi_true: Optional[np.ndarray] = None # (T,K)
    t_detect: Optional[int] = None

def compute_series_over_run(
    run_dir: str | Path,
    *,
    K_old: int,
    hop: HopfieldParams = HopfieldParams(),
    detect_patience: int = 2,
) -> SeriesResult:
    """
    Scan run_dir/round_XXX and build time series for Exp-07 diagnostics.
    Magnetisations are computed via HEBB-Hopfield as requested.
    """
    rdir = Path(run_dir)
    xi_true_path = rdir / "xi_true.npy"
    if not xi_true_path.exists():
        raise FileNotFoundError(f"Missing xi_true.npy in {rdir}")
    xi_true = np.load(xi_true_path)        # (K,N)
    K, N = xi_true.shape

    # Optional: true schedule for plotting
    pis_path = rdir / "pis.npy"
    pis = np.load(pis_path) if pis_path.exists() else None
    if isinstance(pis, np.ndarray) and pis.ndim == 2 and pis.shape[1] == K:
        pi_true_series = pis / (pis.sum(axis=1, keepdims=True) + 1e-9)
    else:
        pi_true_series = None

    rounds = _list_round_dirs(rdir)
    T = len(rounds)
    if T == 0:
        raise RuntimeError(f"No round_* folders found under {rdir}")

    keff = np.full(T, np.nan, dtype=float)
    gap = np.full(T, np.nan, dtype=float)
    TV = np.full(T, np.nan, dtype=float)
    L1 = np.full(T, np.nan, dtype=float)
    m_old = np.full(T, np.nan, dtype=float)
    m_new = np.full(T, np.nan, dtype=float)
    pi_hat_series = np.full((T, K), np.nan, dtype=float)

    # Exposure counts for the Hopfield report (if present)
    exposure_path = rdir / "exposure_counts.npy"
    exposure_counts = np.load(exposure_path) if exposure_path.exists() else None

    for t, rd in enumerate(rounds):
        # ---------- metrics.json (K_eff, TV, eigenvalues, pi_hat, pi_true) ----------
        metrics_path = rd / "metrics.json"
        if metrics_path.exists():
            mj = _read_json(metrics_path)
            keff[t] = float(mj.get("K_eff", np.nan))
            TV[t] = float(mj.get("TV_pi", np.nan))
            if "pi_hat" in mj:
                v = np.asarray(mj["pi_hat"], dtype=float)
                if v.size == K:
                    pi_hat_series[t] = v / (v.sum() + 1e-9)
                    if pi_true_series is not None:
                        L1[t] = float(np.sum(np.abs(pi_hat_series[t] - pi_true_series[t])))
            # spectral gap at K_old
            eigvals = None
            if isinstance(mj.get("keff_info", {}), dict):
                ev = mj["keff_info"].get("eigvals", None)
                if isinstance(ev, list):
                    eigvals = np.asarray(ev, dtype=float)
            if eigvals is None:
                ev_path = rd / "eigs_sel.npy"
                if ev_path.exists():
                    try:
                        eigvals = np.load(ev_path)
                    except Exception:
                        eigvals = None
            if isinstance(eigvals, np.ndarray):
                ev_sorted = -np.sort(-eigvals.ravel())   # DESC
                gap[t] = spectral_gap_at_boundary(ev_sorted, int(K_old), relative=True)

        # ---------- magnetisations via HEBB-Hopfield ----------
        do_hop = (hop.frequency <= 1) or ((t % int(hop.frequency)) == 0)
        if do_hop:
            xi_aligned_path = rd / "xi_aligned.npy"
            xi_ref_for_hebb = np.load(xi_aligned_path) if xi_aligned_path.exists() else None
            res = _eval_hopfield_hebb_for_round(
                rd, xi_true=xi_true, xi_ref_for_hebb=xi_ref_for_hebb,
                exposure_counts=exposure_counts, hp=hop
            )
            if isinstance(res, dict):
                m_mu = None
                for key in ("magnetization_by_mu", "m_by_mu", "mag_by_mu"):
                    if key in res:
                        m_mu = np.asarray(res[key], dtype=float).ravel()
                        break
                if isinstance(m_mu, np.ndarray) and m_mu.size == K:
                    if K_old > 0:
                        m_old[t] = float(np.mean(m_mu[:K_old]))
                    if K_old < K:
                        m_new[t] = float(np.mean(m_mu[K_old:]))

    # novelty detection from K_eff
    t_detect = detect_novelty_round(keff, int(K_old), detect_patience=detect_patience)

    return SeriesResult(
        T=T, K=K, K_old=int(K_old),
        keff=keff, gap=gap, TV=TV, L1=L1,
        m_old=m_old, m_new=m_new,
        pi_hat=(pi_hat_series if np.isfinite(pi_hat_series).any() else None),
        pi_true=pi_true_series,
        t_detect=t_detect,
    )
