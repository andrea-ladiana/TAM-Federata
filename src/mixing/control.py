# -*- coding: utf-8 -*-
"""
Controllo adattivo del peso w (Exp-06 single-only).

API principali:
  - compute_drift_signals: calcola segnali di drift e mismatch (D_t, M_t, S_t)
  - update_w_threshold: policy A (isteresi su S_t)
  - update_w_sigmoid:   policy B (sigmoide S_t -> w)
  - update_w_pctrl:     policy C (PID sul lag di fase |phi|)

Note:
  - D_t = TV(pi_data_t, pi_data_{t-1}); M_t = TV(pi_mem_{t-1}, pi_data_t)
  - S_t = a*D_t + b*M_t
  - Lo smoothing è applicato SOLO al controllo su w (alpha_w), non ai segnali.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

from .metrics import tv_distance


def _normalize_pi(pi: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if pi is None:
        return None
    v = np.asarray(pi, dtype=float)
    s = float(v.sum())
    if s <= 0:
        return np.ones_like(v, dtype=float) / float(v.size)
    return v / s


def compute_drift_signals(
    pi_data_t: np.ndarray,
    pi_data_tm1: Optional[np.ndarray],
    pi_mem_tm1: Optional[np.ndarray],
    *,
    a: float = 0.5,
    b: float = 1.0,
) -> Dict[str, float]:
    """
    Restituisce: {'D_t': drift dati, 'M_t': mismatch memoria→dati, 'S_t': a*D_t + b*M_t}
    Se mancano tm1, usa 0 per il segnale mancante.
    """
    p_t = _normalize_pi(pi_data_t)
    p_tm1 = _normalize_pi(pi_data_tm1) if pi_data_tm1 is not None else None
    m_tm1 = _normalize_pi(pi_mem_tm1) if pi_mem_tm1 is not None else None

    if p_tm1 is None:
        D_t = 0.0
    else:
        D_t = float(tv_distance(p_t, p_tm1))

    if m_tm1 is None:
        M_t = 0.0
    else:
        M_t = float(tv_distance(m_tm1, p_t))

    S_t = float(a) * D_t + float(b) * M_t
    return {"D_t": float(D_t), "M_t": float(M_t), "S_t": float(S_t)}


def _smooth(prev: float, target: float, alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * float(prev) + alpha * float(target)


def update_w_threshold(
    w_prev: float, D_t: float, M_t: float, S_t: float, *,
    w_min: float, w_max: float,
    theta_low: float, theta_high: float,
    delta_up: float, delta_down: float,
    alpha_w: float = 0.3,
) -> float:
    """Policy A (isteresi). Ritorna w_t smussato.

    Convenzione: w più alto = più plasticità (più peso ai dati correnti).
    - Se S_t ≥ theta_high: aumenta w di delta_up
    - Se S_t ≤ theta_low:  diminuisce w di delta_down
    - Altrimenti: mantieni w_prev
    """
    w_star = float(w_prev)
    if float(S_t) >= float(theta_high):
        w_star = w_prev + float(delta_up)
    elif float(S_t) <= float(theta_low):
        w_star = w_prev - float(delta_down)
    # clip
    w_star = float(np.clip(w_star, float(w_min), float(w_max)))
    # smoothing
    return float(_smooth(w_prev, w_star, alpha=float(alpha_w)))


def update_w_sigmoid(
    w_prev: float, D_t: float, M_t: float, S_t: float, *,
    w_min: float, w_max: float,
    theta_mid: float, beta: float,
    alpha_w: float = 0.3,
) -> float:
    """Policy B (mappatura morbida S_t -> w). Ritorna w_t smussato.

    w_star = w_min + (w_max - w_min) * sigma(beta * (S_t - theta_mid))
    dove sigma(x) = 1 / (1 + exp(-x)).
    """
    s = 1.0 / (1.0 + float(np.exp(-float(beta) * (float(S_t) - float(theta_mid)))))
    w_star = float(w_min) + (float(w_max) - float(w_min)) * float(s)
    w_star = float(np.clip(w_star, float(w_min), float(w_max)))
    return float(_smooth(w_prev, w_star, alpha=float(alpha_w)))


def update_w_pctrl(
    w_prev: float,
    lag_series_radians: np.ndarray,
    lag_target: float,
    *,
    w_min: float, w_max: float,
    kp: float, ki: float = 0.0, kd: float = 0.0,
    alpha_w: float = 0.3,
    gate_S_t: Optional[float] = None,
    S_t: Optional[float] = None,
) -> float:
    """
    Policy C (controller sul lag). Calcola errore e = mean(lag_series) - lag_target,
    integra/deriva su finestra, poi w_star = clip(w_prev - kP*e - kI*sumE + kD*diffE).
    Applica smoothing: (1-alpha_w)*w_prev + alpha_w*w_star. Ritorna w_t.
    """
    series = np.asarray(lag_series_radians, dtype=float).reshape(-1)
    if series.size == 0:
        # nessuna informazione: mantieni w_prev
        return float(np.clip(float(w_prev), float(w_min), float(w_max)))

    e_curr = float(np.mean(series) - float(lag_target))
    sumE = float(np.sum(series - float(lag_target)))
    if series.size >= 2:
        # stima diff sull'ultimo passo (forward difference)
        e_prev = float(np.mean(series[:-1]) - float(lag_target))
        diffE = float(e_curr - e_prev)
    else:
        diffE = 0.0

    # legge standard con segni: meno proporzionale/integrale, più derivativo
    w_star = float(w_prev) - float(kp) * e_curr - float(ki) * sumE + float(kd) * diffE
    w_star = float(np.clip(w_star, float(w_min), float(w_max)))

    # Gating opzionale: se S_t è sotto soglia, non aumentare w
    if gate_S_t is not None and S_t is not None:
        try:
            if float(S_t) < float(gate_S_t):
                # blocca solo gli incrementi (plasticità in aumento)
                if w_star > float(w_prev):
                    w_star = float(w_prev)
        except Exception:
            pass

    return float(_smooth(w_prev, w_star, alpha=float(alpha_w)))

