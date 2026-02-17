# -*- coding: utf-8 -*-
"""
Adaptive w computation for Experiment 07 - Novelty Emergence.

Entropy-based adaptive weight with damping mechanisms optimized for
detecting and adapting to emerging archetypes in federated learning.

This is a specialized version of src.mixing.adaptive_w_damped tailored
for the novelty emergence scenario in exp07.
"""
from __future__ import annotations

from typing import Dict, Any
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
    J_memory: np.ndarray,
    J_current: np.ndarray,
    r_ex: float,
) -> Dict[str, Any]:
    """
    Compute raw (undamped) adaptive weight based on sign-consistency entropy.
    
    This measures the agreement between the sign structure of the memory
    coupling matrix and the current round's coupling matrix.
    
    Parameters
    ----------
    J_memory : np.ndarray (N, N)
        Hebbian matrix from reconstructed archetypes (previous round).
    J_current : np.ndarray (N, N)
        Hebbian matrix from current round data.
    r_ex : float in [0, 1]
        Dataset quality parameter (signal-to-noise ratio).
    
    Returns
    -------
    dict
        - 'w_raw': float, raw adaptive weight in [0, 1]
        - 'H_AB': float, observed sign-consistency entropy
        - 'H_min': float, minimum entropy (perfect agreement)
        - 'p': float, fraction of sign agreements
    
    Notes
    -----
    High entropy → distribution mismatch → higher w (trust new data more)
    Low entropy → distribution stable → lower w (trust memory more)
    """
    J_memory = np.asarray(J_memory, dtype=np.float64)
    J_current = np.asarray(J_current, dtype=np.float64)
    
    if J_memory.shape != J_current.shape:
        raise ValueError(
            f"J_memory and J_current must have same shape; "
            f"got {J_memory.shape} vs {J_current.shape}"
        )
    if J_memory.ndim != 2 or J_memory.shape[0] != J_memory.shape[1]:
        raise ValueError(
            f"Coupling matrices must be square; got shape {J_memory.shape}"
        )
    
    N = J_memory.shape[0]
    r_ex = float(np.clip(r_ex, 0.0, 1.0))
    
    # Compute sign matrices
    s_memory = np.sign(J_memory)
    s_current = np.sign(J_current)
    
    # Extract upper triangular off-diagonal indices
    iu = np.triu_indices(N, k=1)
    s_mem_off = s_memory[iu]
    s_cur_off = s_current[iu]
    
    # Count sign agreements
    agreement = (s_mem_off == s_cur_off)
    n_total = len(s_mem_off)
    
    if n_total == 0:
        return {'w_raw': 0.5, 'H_AB': 0.0, 'H_min': 0.0, 'p': 1.0}
    
    n_agree = int(np.sum(agreement))
    p = float(n_agree) / float(n_total)
    
    # Compute entropies
    H_AB = _binary_entropy(p)
    p_min = (1.0 + r_ex**2) / 2.0
    H_min = _binary_entropy(p_min)
    H_max = 1.0
    
    # Compute raw adaptive weight
    # Maps entropy from [H_min, H_max] to [0, 1]
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


def compute_adaptive_w(
    J_memory: np.ndarray,
    J_current: np.ndarray,
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
    # Adaptive EMA parameters
    alpha_min: float = 0.1,
    alpha_max: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute damped adaptive weight w_t with temporal smoothing.
    
    This is the main function for computing adaptive weights in exp07.
    It combines entropy-based detection with various damping strategies.
    
    Damping Modes
    -------------
    1. "ema": Exponential Moving Average
       w_t = α * w_raw + (1-α) * w_{t-1}
       Best for: Smooth gradual adaptation
       
    2. "rate_limit": Hard clipping of change rate
       Δw = clip(w_raw - w_{t-1}, -max_delta_w, +max_delta_w)
       w_t = w_{t-1} + Δw
       Best for: Preventing sudden jumps
       
    3. "momentum": Momentum-based smooth transitions
       delta = (1-β) * (w_raw - w_{t-1})
       w_t = w_{t-1} + delta
       Best for: Balanced responsiveness
       
    4. "adaptive_ema": EMA with adaptive alpha based on uncertainty
       α = α_max - uncertainty * (α_max - α_min)
       w_t = α * w_raw + (1-α) * w_{t-1}
       Best for: Context-aware adaptation
    
    Parameters
    ----------
    J_memory : np.ndarray (N, N)
        Coupling matrix from reconstructed archetypes.
    J_current : np.ndarray (N, N)
        Coupling matrix from current round data.
    r_ex : float in [0, 1]
        Dataset quality parameter.
    w_prev : float
        Previous round's w value for temporal smoothing.
    damping_mode : str
        Smoothing strategy (see above).
    alpha_ema : float in [0, 1]
        EMA smoothing factor (higher = faster adaptation).
    max_delta_w : float
        Maximum allowed change in w per round.
    momentum : float in [0, 1]
        Momentum coefficient.
    alpha_min, alpha_max : float
        Range for adaptive alpha modulation.
    
    Returns
    -------
    dict
        - 'w': float, final damped adaptive weight
        - 'w_raw': float, undamped target value
        - 'H_AB', 'H_min', 'p': entropy metrics
        - 'damping_info': dict with mode-specific details
    
    Examples
    --------
    >>> # Aggressive damping for catastrophic forgetting prevention
    >>> result = compute_adaptive_w(
    ...     J_mem, J_cur, r_ex=0.8, w_prev=0.7,
    ...     damping_mode="ema", alpha_ema=0.2
    ... )
    
    >>> # Balanced adaptation
    >>> result = compute_adaptive_w(
    ...     J_mem, J_cur, r_ex=0.8, w_prev=0.7,
    ...     damping_mode="adaptive_ema", alpha_min=0.2, alpha_max=0.5
    ... )
    """
    # Compute raw (undamped) target
    raw_result = compute_entropy_based_w_raw(J_memory, J_current, r_ex)
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
        beta = float(np.clip(momentum, 0.0, 1.0))
        delta = (1.0 - beta) * (w_raw - w_prev)
        w_damped = w_prev + delta
        damping_info['beta'] = beta
        damping_info['delta'] = float(delta)
        
    elif damping_mode == "adaptive_ema":
        # Adaptive EMA: modulate alpha based on entropy uncertainty
        # High H_AB (near 1.0) → high uncertainty → low alpha (more smoothing)
        # Low H_AB (near H_min) → stable → high alpha (faster adaptation)
        H_max = 1.0
        if H_max - H_min > 1e-12:
            uncertainty = (H_AB - H_min) / (H_max - H_min)
        else:
            uncertainty = 0.5
        
        # Map uncertainty: high uncertainty → low alpha
        alpha = alpha_max - uncertainty * (alpha_max - alpha_min)
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        
        w_damped = alpha * w_raw + (1.0 - alpha) * w_prev
        damping_info['alpha'] = alpha
        damping_info['uncertainty'] = float(uncertainty)
        damping_info['alpha_min'] = alpha_min
        damping_info['alpha_max'] = alpha_max
        
    else:
        # Fallback: no damping
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
