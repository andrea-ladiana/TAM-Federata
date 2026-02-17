# -*- coding: utf-8 -*-
"""
Adaptive w computation based on sign-consistency entropy.

Implements the entropy-based method for determining the optimal blending weight w_t
as described in the theoretical appendix. The method computes the binary entropy of
sign agreement between:
  - A_t: Hebbian matrix of reconstructed archetypes from previous round
  - B_t: Hebbian matrix of current round data (examples)

The adaptive weight w_t balances stability (archetype-driven term) and plasticity
(data-driven term) by mapping the observed sign-consistency entropy H_AB to a
normalized score anchored by the intrinsic noise floor H_min(r) and capped at
maximum uncertainty H_max=1.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def _binary_entropy(p: float) -> float:
    """
    Compute binary entropy h_2(p) = -p log_2(p) - (1-p) log_2(1-p).
    
    Handles edge cases p=0 or p=1 gracefully (returns 0).
    
    Parameters
    ----------
    p : float in [0, 1]
        Probability value.
    
    Returns
    -------
    float
        Binary entropy in [0, 1].
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    # Use natural log and convert to log2 for numerical stability
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def compute_entropy_based_w(
    A_t: np.ndarray,
    B_t: np.ndarray,
    r_ex: float,
) -> Dict[str, Any]:
    """
    Compute adaptive weight w_t based on sign-consistency entropy between A_t and B_t.
    
    Algorithm
    ---------
    1. Compute sign matrices s^A and s^B for A_t and B_t
    2. Calculate sign agreement probability p (fraction of off-diagonal entries with same sign)
    3. Compute binary entropy H_AB = h_2(p)
    4. Compute minimum noise floor H_min = h_2((1 + r^2) / 2)
    5. Normalize: w_t = (H_AB - H_min) / (1 - H_min)
    
    Interpretation
    --------------
    - w_t ≈ 0: sign patterns are consistent => stable distribution, favor archetypes (A_t)
    - w_t ≈ 1: maximum uncertainty => distributional shift, favor fresh data (B_t)
    
    Parameters
    ----------
    A_t : np.ndarray (N, N)
        Hebbian matrix of reconstructed archetypes from previous round.
    B_t : np.ndarray (N, N)
        Hebbian matrix of current round data (examples).
    r_ex : float in [0, 1]
        Dataset quality parameter (average example-archetype correlation).
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'w': float, the computed adaptive weight in [0, 1]
        - 'H_AB': float, binary entropy of sign agreement
        - 'H_min': float, minimum entropy floor from noise
        - 'p': float, sign agreement probability
    
    Notes
    -----
    - A_t and B_t must have the same shape (N, N).
    - Only off-diagonal entries are considered for sign agreement (diagonal ignored).
    - Result is clipped to [0, 1] for robustness.
    
    References
    ----------
    See Appendix (Algorithmic Details for Adaptive w) in the paper.
    """
    A_t = np.asarray(A_t, dtype=np.float64)
    B_t = np.asarray(B_t, dtype=np.float64)
    
    if A_t.shape != B_t.shape:
        raise ValueError(f"A_t and B_t must have same shape; got {A_t.shape} vs {B_t.shape}")
    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
        raise ValueError(f"A_t and B_t must be square matrices; got shape {A_t.shape}")
    
    N = A_t.shape[0]
    r_ex = float(np.clip(r_ex, 0.0, 1.0))
    
    # Compute sign matrices
    s_A = np.sign(A_t)
    s_B = np.sign(B_t)
    
    # Extract off-diagonal indices (upper triangle, excluding diagonal)
    # We use only off-diagonal because diagonal is self-correlation and doesn't
    # carry cross-feature structure information
    iu = np.triu_indices(N, k=1)
    s_A_off = s_A[iu]
    s_B_off = s_B[iu]
    
    # Count sign agreements (ignoring zeros for robustness)
    # Agreement: both positive, both negative, or both zero
    agreement = (s_A_off == s_B_off)
    
    # Compute probability p: fraction of off-diagonal entries with same sign
    n_total = len(s_A_off)
    if n_total == 0:
        # Edge case: 1x1 matrix (no off-diagonal entries)
        # Return w=0 (favor stability when no cross-feature information)
        return {
            'w': 0.0,
            'H_AB': 0.0,
            'H_min': 0.0,
            'p': 1.0,
        }
    
    n_agree = int(np.sum(agreement))
    p = float(n_agree) / float(n_total)
    
    # Compute binary entropy of sign agreement
    H_AB = _binary_entropy(p)
    
    # Compute minimum entropy floor H_min(r)
    # From probabilistic argument: P(η_i η_j = ξ^μ_i ξ^μ_j) = (1 + r²) / 2
    p_min = (1.0 + r_ex**2) / 2.0
    H_min = _binary_entropy(p_min)
    
    # Maximum entropy is h_2(0.5) = 1.0 (when p = 0.5, maximum uncertainty)
    H_max = 1.0
    
    # Compute normalized adaptive weight
    # w_t = (H_AB - H_min) / (H_max - H_min) = (H_AB - H_min) / (1 - H_min)
    denominator = H_max - H_min
    if denominator <= 1e-12:
        # Edge case: H_min ≈ 1 (r ≈ 0, completely random data)
        # In this regime all data is noise; default to balanced blend
        w_t = 0.5
    else:
        w_t = (H_AB - H_min) / denominator
    
    # Clip to [0, 1] for robustness (numerical issues might push slightly outside)
    w_t = float(np.clip(w_t, 0.0, 1.0))
    
    return {
        'w': w_t,
        'H_AB': float(H_AB),
        'H_min': float(H_min),
        'p': float(p),
    }
