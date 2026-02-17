# -*- coding: utf-8 -*-
"""
Federated simulation loop with per-client adaptive weight.

Pipeline (each round t):
    1. Each client generates a local batch and computes J_local_c
    2. Compute adaptive w_c(t) from sign-agreement entropy (t > 0)
    3. Blend: J_c = w_c · J_local_c + (1 − w_c) · J_server_prev
    4. Server aggregates: J_s = (1/L) Σ_c J_c
    5. Server: propagate_J → eigen_cut → dis_check → reconstructed archetypes
    6. Server: J_server = Hebb_J(ξ_reconstructed)  →  broadcast to clients
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

# ── local imports (from same package) ──
from .adaptive_w import adaptive_w_step

# ── project imports (require project root on sys.path) ──
from src.unsup.functions import gen_patterns, unsupervised_J, Hebb_J, propagate_J
from src.unsup.spectrum import eigen_cut
from src.unsup.dynamics import dis_check
from src.unsup.config import TAMParams, SpectralParams
from src.unsup.metrics import retrieval_mean_hungarian


# ──────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All parameters for the client-aware-w experiment."""
    N: int = 200              # neurons
    K: int = 3                # archetypes
    L: int = 5                # clients  (L-1 good + 1 attacker)
    T: int = 20               # rounds
    M_per: int = 200          # examples per client per round
    r_good: float = 0.8       # signal quality for good clients
    r_bad: float = 0.0        # signal quality for attacker (pure noise)
    attacker_idx: int = 4     # which client is the attacker
    alpha_ema: float = 0.5    # EMA coefficient for w smoothing
    prop_iters: int = 50       # propagation iterations
    seed: int = 42

    # TAM / spectral (permissive values — needed for the eigenvalue
    # scale produced by unsupervised_J / Hebb_J normalisation)
    tam: TAMParams = field(default_factory=lambda: TAMParams(
        beta_T=4.5, lam=0.1, h_in=0.1, updates=30,
        noise_scale=0.3, min_scale=0.02, anneal=True, schedule="linear",
    ))
    spec: SpectralParams = field(default_factory=lambda: SpectralParams(
        tau=0.10, rho=0.1, qthr=0.8,
    ))


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _generate_batch(
    xi_true: np.ndarray,
    M: int,
    r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate M noisy examples from K archetypes (uniform mixture).

    Each example: η = χ ⊙ ξ_μ  with  P(χ_i = +1) = (1+r)/2.
    When r = 0 this produces pure symmetric noise.
    """
    K, N = xi_true.shape
    mus = rng.integers(0, K, size=M)
    probs = rng.uniform(size=(M, N))
    chi = np.where(probs <= 0.5 * (1.0 + r), 1.0, -1.0).astype(np.float32)
    return (chi * xi_true[mus]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# Main simulation
# ──────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Container for all simulation outputs."""
    w_history: np.ndarray          # (L, T)  adaptive weights
    w_raw_history: np.ndarray      # (L, T)  raw (pre-EMA) weights
    H_history: np.ndarray          # (L, T)  sign-agreement entropy
    p_history: np.ndarray          # (L, T)  sign-agreement fraction
    retrieval_history: np.ndarray  # (T,)    server-side mean retrieval
    mag_history: np.ndarray        # (K, T)  per-archetype magnetisation
    xi_true: np.ndarray            # (K, N)  ground-truth archetypes
    config: SimConfig = field(default_factory=SimConfig)


def run_simulation(cfg: Optional[SimConfig] = None) -> SimResult:
    """
    Execute the full federated simulation and return recorded trajectories.
    """
    if cfg is None:
        cfg = SimConfig()

    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)  # gen_patterns uses legacy API

    # ── Generate true archetypes ──
    xi_true = np.array(gen_patterns(cfg.N, cfg.K), dtype=np.float32)

    # ── Storage ──
    L, T, K, N = cfg.L, cfg.T, cfg.K, cfg.N
    w_hist = np.ones((L, T))
    w_raw_hist = np.ones((L, T))
    H_hist = np.zeros((L, T))
    p_hist = np.ones((L, T))
    retr_hist = np.zeros(T)
    mag_hist = np.zeros((K, T))

    # ── Server state ──
    xi_reconstructed: Optional[np.ndarray] = None
    J_server_prev: Optional[np.ndarray] = None

    for t in range(T):
        # ── 1. Each client generates local batch & Hebbian correlator ──
        J_local = np.zeros((L, N, N), dtype=np.float32)
        for c in range(L):
            r_c = cfg.r_bad if c == cfg.attacker_idx else cfg.r_good
            eta_c = _generate_batch(xi_true, cfg.M_per, r_c, rng)
            J_local[c] = unsupervised_J(eta_c, cfg.M_per)

        # ── 2. Compute adaptive w_c(t) ──
        if t == 0:
            # First round: no prior → w = 1 for all
            w_hist[:, 0] = 1.0
            w_raw_hist[:, 0] = 1.0
        else:
            for c in range(L):
                res = adaptive_w_step(
                    J_server=J_server_prev,
                    J_local=J_local[c],
                    w_prev=w_hist[c, t - 1],
                    alpha_ema=cfg.alpha_ema,
                    r_ref=cfg.r_good,
                )
                w_hist[c, t] = res["w"]
                w_raw_hist[c, t] = res["w_raw"]
                H_hist[c, t] = res["H_AB"]
                p_hist[c, t] = res["p"]

        # ── 3. Client blending ──
        J_blended = np.zeros((L, N, N), dtype=np.float32)
        for c in range(L):
            wc = w_hist[c, t]
            if J_server_prev is not None:
                J_blended[c] = wc * J_local[c] + (1.0 - wc) * J_server_prev
            else:
                J_blended[c] = J_local[c]

        # ── 4. Server aggregation ──
        J_server = np.mean(J_blended, axis=0)

        # ── 5. Server spectral factorisation ──
        J_KS = np.asarray(
            propagate_J(J_server, J_real=-1, verbose=False, iters=cfg.prop_iters),
            dtype=np.float32,
        )
        V, k_eff, _info = eigen_cut(J_KS, tau=cfg.spec.tau, return_info=True)

        if V.shape[0] >= 1:
            xi_r, m_vec = dis_check(
                V=V, K=K, L=L,
                J_rec=J_server, JKS_iter=J_KS,
                xi_true=xi_true,
                tam=cfg.tam, spec=cfg.spec,
                show_progress=False,
            )
            if xi_r.shape[0] > 0:
                xi_bin = np.where(xi_r >= 0, 1, -1).astype(np.float32)
                # Keep at most K reconstructed archetypes (matching
                # single_round.py pipeline). Extra candidates add noise
                # to J_server_prev and degrade sign-agreement fidelity.
                xi_reconstructed = xi_bin[: min(K, xi_bin.shape[0])]
            else:
                xi_reconstructed = None
        else:
            xi_reconstructed = None

        # ── 6. Build server reconstruction operator for next round ──
        if xi_reconstructed is not None:
            J_server_prev = Hebb_J(xi_reconstructed)
            # Retrieval
            retr = retrieval_mean_hungarian(
                xi_reconstructed.astype(int), xi_true.astype(int)
            )
            retr_hist[t] = retr
            # Per-archetype magnetisation
            M_overlap = np.abs(xi_reconstructed.astype(int) @ xi_true.astype(int).T) / N
            for k in range(K):
                mag_hist[k, t] = float(M_overlap[:, k].max()) if M_overlap.shape[0] > 0 else 0.0
        else:
            J_server_prev = J_server  # fallback: use raw aggregated J
            retr_hist[t] = 0.0

        # ── Progress ──
        w_good = np.mean(w_hist[:cfg.attacker_idx, t])
        w_att = w_hist[cfg.attacker_idx, t]
        p_good = np.mean(p_hist[:cfg.attacker_idx, t]) if t > 0 else 1.0
        p_att = p_hist[cfg.attacker_idx, t] if t > 0 else 1.0
        print(
            f"  Round {t:2d}/{T-1}  |  K_eff={k_eff}  "
            f"retr={retr_hist[t]:.3f}  "
            f"p_good={p_good:.3f}  p_att={p_att:.3f}  "
            f"w_good={w_good:.3f}  w_att={w_att:.3f}"
        )

    return SimResult(
        w_history=w_hist,
        w_raw_history=w_raw_hist,
        H_history=H_hist,
        p_history=p_hist,
        retrieval_history=retr_hist,
        mag_history=mag_hist,
        xi_true=xi_true,
        config=cfg,
    )


# ──────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ──────────────────────────────────────────────────────────────────

@dataclass
class MultiSeedResult:
    """Aggregated statistics over S independent seeds."""
    # Panel A: w trajectories — shape (T,)
    w_good_mean: np.ndarray
    w_good_se: np.ndarray
    w_att_mean: np.ndarray
    w_att_se: np.ndarray
    # Panel B: retrieval — shape (T,)
    retr_mean: np.ndarray
    retr_se: np.ndarray
    # Panel B: per-archetype mag — shape (K, T)
    mag_mean: np.ndarray
    mag_se: np.ndarray
    # Metadata
    n_seeds: int
    config: SimConfig = field(default_factory=SimConfig)


def run_multi_seed(
    cfg: SimConfig,
    n_seeds: int = 20,
    seed_base: int = 0,
    verbose: bool = True,
) -> MultiSeedResult:
    """
    Run *n_seeds* independent simulations and return aggregated
    mean ± standard-error statistics.
    """
    T = cfg.T
    K = cfg.K
    L = cfg.L
    att = cfg.attacker_idx

    # Accumulators: per-seed scalars  (n_seeds, T)
    all_w_good = np.zeros((n_seeds, T))
    all_w_att = np.zeros((n_seeds, T))
    all_retr = np.zeros((n_seeds, T))
    all_mag = np.zeros((n_seeds, K, T))

    for s in range(n_seeds):
        seed_s = seed_base + s
        cfg_s = SimConfig(
            N=cfg.N, K=cfg.K, L=cfg.L, T=cfg.T,
            M_per=cfg.M_per, r_good=cfg.r_good, r_bad=cfg.r_bad,
            attacker_idx=cfg.attacker_idx,
            alpha_ema=cfg.alpha_ema, prop_iters=cfg.prop_iters,
            seed=seed_s,
            tam=cfg.tam, spec=cfg.spec,
        )
        if verbose:
            print(f"\n--- Seed {s+1}/{n_seeds}  (seed={seed_s}) ---")
        res = run_simulation(cfg_s)

        # w_good = mean over good clients for this seed
        good_mask = np.ones(L, dtype=bool)
        good_mask[att] = False
        all_w_good[s] = res.w_history[good_mask].mean(axis=0)   # (T,)
        all_w_att[s] = res.w_history[att]                        # (T,)
        all_retr[s] = res.retrieval_history                      # (T,)
        all_mag[s] = res.mag_history                             # (K, T)

    # Aggregate: mean & SEM
    def _mean_se(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        m = arr.mean(axis=axis)
        se = arr.std(axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
        return m, se

    w_good_m, w_good_se = _mean_se(all_w_good)
    w_att_m, w_att_se = _mean_se(all_w_att)
    retr_m, retr_se = _mean_se(all_retr)
    # mag: shape (n_seeds, K, T) → mean/se over seeds → (K, T)
    mag_m, mag_se = _mean_se(all_mag)

    return MultiSeedResult(
        w_good_mean=w_good_m, w_good_se=w_good_se,
        w_att_mean=w_att_m, w_att_se=w_att_se,
        retr_mean=retr_m, retr_se=retr_se,
        mag_mean=mag_m, mag_se=mag_se,
        n_seeds=n_seeds, config=cfg,
    )
