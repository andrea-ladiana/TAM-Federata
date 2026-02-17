# -*- coding: utf-8 -*-
"""
Client-aware adaptive weight  w_c(t)  based on normalised sign-agreement.

Metric  (cf. eq:p_sign_agreement in the manuscript)
------
p_c(t) = fraction of off-diagonal entries where
         sign(A^{t-1}_{ij}) == sign(J̃_c^{t}_{ij})

Under the generative model with quality parameter r, the theoretical
maximum sign agreement is  p₀ = (1 + r²) / 2.

Weight rule
-----------
    w_raw,c(t) = clip( (p_c − ½) / (p₀ − ½),  0,  1 )
    w_c(t)     = α · w_raw,c(t) + (1−α) · w_c(t−1)       [EMA]

Interpretation
--------------
p ≈ 0.5  →  local data is pure noise   (r ≈ 0)  →  w → 0  (trust server)
p ≈ p₀   →  data at full quality       (r = r_ref)  →  w → 1  (trust data)

This avoids the saturation issue of binary-entropy mapping near p ≈ 0.5,
giving clear separation between good and noisy clients.
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Core primitives
# ──────────────────────────────────────────────────────────────────

def sign_agreement(A: np.ndarray, B: np.ndarray) -> float:
    """Fraction of upper-triangular off-diagonal sign agreements."""
    N = A.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(np.mean(np.sign(A[iu]) == np.sign(B[iu])))


def binary_entropy(p: float) -> float:
    """Binary entropy h₂(p), for diagnostic logging."""
    p = float(np.clip(p, 1e-15, 1.0 - 1e-15))
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


# ──────────────────────────────────────────────────────────────────
# Per-client, per-round update
# ──────────────────────────────────────────────────────────────────

def adaptive_w_step(
    J_server: np.ndarray,
    J_local: np.ndarray,
    w_prev: float,
    alpha_ema: float = 0.5,
    r_ref: float = 0.8,
) -> dict:
    """
    Single-step update of the client-aware adaptive weight.

    Parameters
    ----------
    J_server  : (N, N)  server reconstruction operator from previous round.
    J_local   : (N, N)  client local Hebbian correlator from current round.
    w_prev    : previous round's w value for this client.
    alpha_ema : EMA smoothing coefficient (higher → faster adaptation).
    r_ref     : reference data-quality parameter for normalisation.

    Returns
    -------
    dict with keys  'w', 'w_raw', 'p', 'H_AB'.
    """
    p = sign_agreement(J_server, J_local)
    H = binary_entropy(p)

    # Normalised sign-agreement → w_raw ∈ [0, 1]
    p0 = 0.5 * (1.0 + r_ref ** 2)          # theoretical max agreement
    denom = p0 - 0.5
    if denom < 1e-12:
        w_raw = 0.0
    else:
        w_raw = (p - 0.5) / denom

    w_raw = float(np.clip(w_raw, 0.0, 1.0))
    w = alpha_ema * w_raw + (1.0 - alpha_ema) * w_prev

    return {
        "w": float(np.clip(w, 0.0, 1.0)),
        "w_raw": float(w_raw),
        "p": float(p),
        "H_AB": float(H),
    }
