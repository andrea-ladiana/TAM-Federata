# -*- coding: utf-8 -*-
"""
Damped adaptive w computation based on sign-consistency entropy.

This module extends the entropy-based adaptive w method with temporal damping
mechanisms to prevent catastrophic forgetting and excessive rigidity caused by
abrupt changes in w_t.

Key improvements over adaptive_w.py:
  1. Exponential Moving Average (EMA) smoothing of w_t across rounds
  2. Rate limiting: maximum allowed change Δw per round
  3. Momentum-based updates: w_t evolves gradually toward target
  4. Adaptive damping: stronger smoothing when uncertainty is high

The goal is to achieve smooth transitions that balance:
  - Fast adaptation when distributional shifts are genuine and sustained
  - Slow changes to avoid catastrophic forgetting from transient noise
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


def _binary_entropy(p: float) -> float:
    """
    Compute binary entropy h_2(p) = -p log_2(p) - (1-p) log_2(1-p).
    
    Handles edge cases p=0 or p=1 gracefully (returns 0).
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def compute_entropy_based_w_raw(
    A_t: np.ndarray,
    B_t: np.ndarray,
    r_ex: float,
) -> Dict[str, Any]:
    """
    Compute raw (undamped) adaptive weight w_t based on sign-consistency entropy.
    
    This is the instantaneous target value before applying temporal smoothing.
    See adaptive_w.py for detailed algorithm description.
    
    Returns
    -------
    dict with keys: 'w_raw', 'H_AB', 'H_min', 'p'
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
    
    # Extract off-diagonal indices
    iu = np.triu_indices(N, k=1)
    s_A_off = s_A[iu]
    s_B_off = s_B[iu]
    
    # Count sign agreements
    agreement = (s_A_off == s_B_off)
    n_total = len(s_A_off)
    
    if n_total == 0:
        return {'w_raw': 0.0, 'H_AB': 0.0, 'H_min': 0.0, 'p': 1.0}
    
    n_agree = int(np.sum(agreement))
    p = float(n_agree) / float(n_total)
    
    # Compute entropies
    H_AB = _binary_entropy(p)
    p_min = (1.0 + r_ex**2) / 2.0
    H_min = _binary_entropy(p_min)
    H_max = 1.0
    
    # Compute raw adaptive weight
    denominator = H_max - H_min
    if denominator <= 1e-12:
        w_raw = 0.5
    else:
        w_raw = (H_AB - H_min) / denominator
    
    w_raw = float(np.clip(w_raw, 0.0, 1.0))
    
    return {
        'w_raw': w_raw,
        'H_AB': float(H_AB),
        'H_min': float(H_min),
        'p': float(p),
    }


def compute_entropy_based_w_damped(
    A_t: np.ndarray,
    B_t: np.ndarray,
    r_ex: float,
    w_prev: float,
    *,
    damping_mode: str = "ema",
    # EMA parameters
    alpha_ema: float = 0.3,
    # Rate limiting parameters
    max_delta_w: float = 0.15,
    # Momentum parameters
    momentum: float = 0.7,
    # Adaptive damping parameters
    adaptive_alpha: bool = False,
    alpha_min: float = 0.1,
    alpha_max: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute damped adaptive weight w_t with temporal smoothing.
    
    Damping Modes
    -------------
    1. "ema" (default): Exponential Moving Average
       w_t = α * w_raw + (1-α) * w_{t-1}
       
    2. "rate_limit": Hard clipping of change rate
       Δw = clip(w_raw - w_{t-1}, -max_delta_w, +max_delta_w)
       w_t = w_{t-1} + Δw
       
    3. "momentum": Momentum-based smooth transitions
       v_t = β * v_{t-1} + (1-β) * (w_raw - w_{t-1})
       w_t = w_{t-1} + v_t
       
    4. "adaptive_ema": EMA with adaptive alpha based on entropy uncertainty
       When H_AB is near H_max (high uncertainty), use stronger smoothing (lower α)
       When H_AB is near H_min (stable), allow faster adaptation (higher α)
    
    Parameters
    ----------
    A_t : np.ndarray (N, N)
        Hebbian matrix of reconstructed archetypes from previous round.
    B_t : np.ndarray (N, N)
        Hebbian matrix of current round data.
    r_ex : float in [0, 1]
        Dataset quality parameter.
    w_prev : float
        Previous round's w value (w_{t-1}) for temporal smoothing.
    damping_mode : str
        Smoothing strategy: "ema", "rate_limit", "momentum", "adaptive_ema"
    alpha_ema : float in [0, 1]
        EMA smoothing factor (higher = faster adaptation)
    max_delta_w : float
        Maximum allowed change in w per round (for rate_limit mode)
    momentum : float in [0, 1]
        Momentum coefficient (for momentum mode)
    adaptive_alpha : bool
        If True, modulate alpha_ema based on H_AB (for adaptive_ema mode)
    alpha_min, alpha_max : float
        Range for adaptive alpha modulation
    
    Returns
    -------
    dict
        - 'w': float, damped adaptive weight
        - 'w_raw': float, undamped target value
        - 'H_AB', 'H_min', 'p': entropy metrics
        - 'damping_info': dict with mode-specific details
    
    Notes
    -----
    Recommended configurations:
    - Aggressive damping (prevent catastrophic forgetting):
      damping_mode="ema", alpha_ema=0.2, or damping_mode="rate_limit", max_delta_w=0.10
    - Moderate damping (balanced):
      damping_mode="adaptive_ema", alpha_min=0.2, alpha_max=0.5
    - Light damping (fast adaptation):
      damping_mode="momentum", momentum=0.5
    """
    # Compute raw (undamped) target
    raw_result = compute_entropy_based_w_raw(A_t, B_t, r_ex)
    w_raw = raw_result['w_raw']
    H_AB = raw_result['H_AB']
    H_min = raw_result['H_min']
    p = raw_result['p']
    
    w_prev = float(np.clip(w_prev, 0.0, 1.0))
    damping_mode = str(damping_mode).lower().strip()
    
    # Apply damping strategy
    damping_info: Dict[str, Any] = {'mode': damping_mode}
    
    if damping_mode == "ema":
        # Simple Exponential Moving Average
        alpha = float(np.clip(alpha_ema, 0.0, 1.0))
        w_damped = alpha * w_raw + (1.0 - alpha) * w_prev
        damping_info['alpha'] = alpha
        
    elif damping_mode == "rate_limit":
        # Hard clip on maximum change per round
        delta_w = w_raw - w_prev
        delta_w_clipped = float(np.clip(delta_w, -max_delta_w, max_delta_w))
        w_damped = w_prev + delta_w_clipped
        damping_info['delta_w_raw'] = float(delta_w)
        damping_info['delta_w_clipped'] = delta_w_clipped
        damping_info['max_delta_w'] = max_delta_w
        
    elif damping_mode == "momentum":
        # Momentum-based: smooth velocity toward target
        # (Requires external state for velocity; here we approximate with single-step)
        beta = float(np.clip(momentum, 0.0, 1.0))
        # Simplified momentum without explicit velocity state:
        # v_t ≈ (1-β) * (w_raw - w_prev)
        # w_t = w_prev + v_t
        delta = (1.0 - beta) * (w_raw - w_prev)
        w_damped = w_prev + delta
        damping_info['beta'] = beta
        damping_info['delta'] = float(delta)
        
    elif damping_mode == "adaptive_ema":
        # Adaptive EMA: modulate alpha based on entropy uncertainty
        # When H_AB is high (near 1.0), distribution is very uncertain -> smooth more (low alpha)
        # When H_AB is low (near H_min), distribution is stable -> adapt faster (high alpha)
        H_max = 1.0
        if H_max - H_min > 1e-12:
            # Normalize H_AB to [0, 1] range within [H_min, H_max]
            uncertainty = (H_AB - H_min) / (H_max - H_min)
        else:
            uncertainty = 0.5
        
        # Map uncertainty: high uncertainty -> low alpha (more smoothing)
        alpha = alpha_max - uncertainty * (alpha_max - alpha_min)
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        
        w_damped = alpha * w_raw + (1.0 - alpha) * w_prev
        damping_info['alpha'] = alpha
        damping_info['uncertainty'] = float(uncertainty)
        damping_info['alpha_min'] = alpha_min
        damping_info['alpha_max'] = alpha_max
        
    else:
        # Fallback: no damping (same as original adaptive_w)
        w_damped = w_raw
        damping_info['mode'] = 'none'
    
    # Final safety clipping
    w_damped = float(np.clip(w_damped, 0.0, 1.0))
    
    return {
        'w': w_damped,
        'w_raw': w_raw,
        'H_AB': H_AB,
        'H_min': H_min,
        'p': p,
        'damping_info': damping_info,
    }
