# -*- coding: utf-8 -*-
"""
Generatori di mixing-schedule π_t sul simplesso (K archetipi) per Exp-06 (single-only).

Fornisce tre strategie robuste:
- cyclic(T, K, period, gamma, temp)         : traiettoria liscia e periodica, ideale per esperimenti "cyclic"
- piecewise_dirichlet(T, K, block, alpha)   : blocchi costanti campionati da Dirichlet
- random_walk(T, K, step_sigma, tv_max)     : random walk su logits con vincolo di TV step-wise

Utility:
- make_schedule(hp, kind=..., rng=None, **kwargs) -> (T, K) numpy array (float32)
- total_variation(p, q) : distanza TV = 0.5 * ||p - q||_1

Nota: restituisce sempre pesi positivi e normalizzati (somma=1) per ogni round.
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


# ----------------------------
# utilità
# ----------------------------
def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x / float(max(1e-8, temp))) - np.max(x)  # stabilità numerica
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        # fallback uniforme
        return np.ones_like(x) / float(x.size)
    return e / s


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV(p, q) = 0.5 * sum_k |p_k - q_k|"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


# ----------------------------
# scheduler: cyclic
# ----------------------------
def cyclic(
    T: int,
    K: int,
    *,
    period: int,
    gamma: float = 3.0,
    temp: float = 1.0,
    center_mix: float = 0.0,
) -> np.ndarray:
    """
    Traccia una traiettoria periodica usando K sinusoidi sfasate e softmax.

    Parametri
    ---------
    T : numero di round
    K : numero archetipi
    period : periodo (in round). Se T non è multiplo, la fase "riparte" ciclicamente.
    gamma : ampiezza dei logits (più grande = più vicino ai vertici)
    temp : temperatura softmax (più piccola = distribuzioni più appuntite)
    center_mix : blending con uniforme: π <- (1-center_mix)*π + center_mix*(1/K)

    Returns
    -------
    pis : (T, K) float32
    """
    if period <= 0:
        raise ValueError("period dev'essere > 0.")
    phases = 2.0 * np.pi * np.arange(K, dtype=float) / float(K)
    pis = np.zeros((T, K), dtype=np.float32)

    for t in range(T):
        theta = 2.0 * np.pi * (t % period) / float(period)
        logits = gamma * np.cos(theta + phases)
        p = _softmax(logits, temp=temp)
        if center_mix > 0.0:
            p = (1.0 - float(center_mix)) * p + float(center_mix) * (np.ones(K) / float(K))
        pis[t] = p.astype(np.float32)
    return pis


# ----------------------------
# scheduler: piecewise Dirichlet
# ----------------------------
def piecewise_dirichlet(
    T: int,
    K: int,
    *,
    block: int,
    alpha: Sequence[float] | float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Blocchi costanti (lunghezza 'block') con pesi campionati da Dirichlet(alpha).

    Se T non è multiplo di block, l'ultimo blocco viene troncato.
    alpha:
        - scalare => vettore [alpha]*K
        - sequenza => di lunghezza K
    """
    if block <= 0:
        raise ValueError("block dev'essere > 0.")
    rng = np.random.default_rng() if rng is None else rng
    alpha_vec = np.full((K,), float(alpha), dtype=float) if isinstance(alpha, (int, float)) else np.asarray(alpha, dtype=float)
    if alpha_vec.shape != (K,):
        raise ValueError("alpha deve essere scalare oppure una sequenza di lunghezza K.")

    pis = np.zeros((T, K), dtype=np.float32)
    t = 0
    while t < T:
        # campiona un vettore di mixing dal semplice
        p = rng.dirichlet(alpha_vec)
        # normalizzazione (robustezza numerica)
        if p.sum() <= 0:
            p = np.ones(K) / float(K)
        p = (p / p.sum()).astype(np.float32)

        end = min(T, t + block)
        pis[t:end] = p
        t = end
    return pis


# ----------------------------
# scheduler: random walk su logits
# ----------------------------
def random_walk(
    T: int,
    K: int,
    *,
    step_sigma: float = 0.7,
    tv_max: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random walk nei logits con softmax → π_t. Limita la TV step-wise a 'tv_max' (≈ ampiezza massima).

    Parametri
    ---------
    step_sigma : deviazione standard del passo gaussiano sui logits
    tv_max     : bound massimo per TV(π_t, π_{t-1}) per evitare salti irrealistici
    """
    rng = np.random.default_rng() if rng is None else rng
    logits = np.zeros((K,), dtype=float)  # parte uniforme
    pis = np.zeros((T, K), dtype=np.float32)
    pis[0] = np.ones((K,), dtype=np.float32) / float(K)

    for t in range(1, T):
        # proposta passo
        delta = rng.normal(loc=0.0, scale=step_sigma, size=(K,))
        # forza zero-mean per tenere il baricentro controllato
        delta -= delta.mean()
        logits_prop = logits + delta
        p_prop = _softmax(logits_prop)

        # enforce TV bound via backtracking se necessario
        tv = total_variation(p_prop, pis[t - 1])
        if tv > tv_max:
            scale = 1.0
            for _ in range(12):  # backtracking limitato
                scale *= 0.5
                logits_prop = logits + scale * delta
                p_prop = _softmax(logits_prop)
                tv = total_variation(p_prop, pis[t - 1])
                if tv <= tv_max:
                    break

        pis[t] = p_prop.astype(np.float32)
        logits = logits_prop
    return pis


# ----------------------------
# factory principale
# ----------------------------
def make_schedule(
    hp,
    *,
    kind: str = "cyclic",
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> np.ndarray:
    """
    Restituisce una matrice (T, K) con la mixing schedule desiderata.

    kind ∈ {'cyclic', 'piecewise_dirichlet', 'random_walk'}.
    Parametri specifici vedi le funzioni omonime.
    """
    T = int(hp.n_batch)
    K = int(hp.K)

    if kind == "cyclic":
        period = kwargs.get("period", max(3, T))
        gamma = kwargs.get("gamma", 3.0)
        temp = kwargs.get("temp", 1.0)
        center_mix = kwargs.get("center_mix", 0.0)
        return cyclic(T, K, period=period, gamma=gamma, temp=temp, center_mix=center_mix)

    if kind == "piecewise_dirichlet":
        block = kwargs.get("block", max(1, T // 6))
        alpha = kwargs.get("alpha", 1.0)
        return piecewise_dirichlet(T, K, block=block, alpha=alpha, rng=rng)

    if kind == "random_walk":
        step_sigma = kwargs.get("step_sigma", 0.7)
        tv_max = kwargs.get("tv_max", 0.35)
        return random_walk(T, K, step_sigma=step_sigma, tv_max=tv_max, rng=rng)

    raise ValueError(f"kind '{kind}' non riconosciuto.")
