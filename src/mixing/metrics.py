# -*- coding: utf-8 -*-
"""
Metriche e diagnostiche per Exp-06 (single-only).

Include:
- Stima π̂_t dai soli esempi E_t rispetto a un set di riferimenti ξ_ref
- Distanza di TV tra π_t e π̂_t
- K_eff wrapper (MP / shuffle) tramite funzione già definita nella codebase
- Embedding del simplesso Δ_2 (K=3) in R^2 e stima di lag/ampiezza per drift ciclici
- Indice di forgetting e misure di "equità" (varianza e Gini) sulle magnetizzazioni
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np

# riuso dalla codebase
from src.unsup.functions import estimate_K_eff_from_J  # noqa: F401


# ----------------------------
# utilità generiche
# ----------------------------
def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance TV(p,q) = 0.5 * ||p - q||_1."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini su vettori non negativi. Se tutti zero -> 0.
    Restituisce un valore in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0.0):
        return 0.0
    if np.any(x < 0):
        # shift to non-negative
        x = x - x.min()
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n
    return float(g)


# ----------------------------
# π̂_t stimato dagli esempi (classificatore su ξ_ref)
# ----------------------------
def estimate_pi_hat_from_examples(xi_ref: np.ndarray, E_t: np.ndarray) -> np.ndarray:
    """
    Classifica ogni esempio in E_t sul set di riferimenti xi_ref per overlapping massimo.
    Restituisce π̂_t = frequenze normalizzate. Robusto anche se xi_ref ha S>=K pattern: usa i primi K.

    Parametri
    ---------
    xi_ref : (S, N) pattern di riferimento (binarizzati in {±1})
    E_t    : (L, M_c, N) esempi del round t

    Returns
    -------
    pi_hat : (K,) float64
    """
    L, M_c, N = E_t.shape
    S = xi_ref.shape[0]
    K = min(S, max(1, S))  # in pratica S==K, ma restiamo robusti
    ref = xi_ref[:K]
    X = E_t.reshape(L * M_c, N)
    Ov = X @ ref.T
    mu_hat = np.argmax(Ov, axis=1)
    counts = np.bincount(mu_hat, minlength=K).astype(float)
    if counts.sum() <= 0:
        return np.ones(K, dtype=float) / float(K)
    return counts / counts.sum()


# ----------------------------
# K_eff wrapper (riuso codebase)
# ----------------------------
def keff_and_info(
    J_KS: np.ndarray,
    *,
    method: str = "shuffle",
    M_eff: Optional[int] = None,
    data_var: Optional[float] = None,
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """
    Calcola K_eff e informazioni accessorie riusando la funzione esistente nella codebase.
    """
    K_eff, keep_mask, info = estimate_K_eff_from_J(J_KS, method=method, M_eff=M_eff, data_var=data_var)
    return int(K_eff), np.asarray(keep_mask), dict(info)


# ----------------------------
# Embedding Δ2 -> R^2 e lag/ampiezza (K=3)
# ----------------------------
_TRI_VERTS = np.array([
    [1.0, 0.0],                         # v0
    [-0.5, np.sqrt(3.0) / 2.0],         # v1
    [-0.5, -np.sqrt(3.0) / 2.0],        # v2
], dtype=float)


def simplex_embed_2d(pi: np.ndarray) -> np.ndarray:
    """
    Proietta un vettore π ∈ Δ2 in R^2 usando i vertici di un triangolo equilatero.
    """
    if pi.shape[-1] != 3:
        raise ValueError("simplex_embed_2d richiede K=3.")
    return pi @ _TRI_VERTS  # (2,) per singolo, (T,2) per sequenza


def _angles_from_xy(xy: np.ndarray) -> np.ndarray:
    """Restituisce la sequenza di angoli θ(t) = atan2(y, x) in [-π, π]."""
    return np.arctan2(xy[..., 1], xy[..., 0])


def _radial_distance(xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xy, axis=-1)


def lag_and_amplitude(
    pi_true_seq: np.ndarray,
    pi_hat_seq: np.ndarray,
) -> Dict[str, float]:
    """
    Stima lag (in round e in radianti) e rapporto di ampiezza tra traiettorie cicliche su Δ2 (K=3).
    Metodo: embed in R^2, costruisci fasi θ, massimizza la correlazione tra e^{iθ_true} e e^{iθ_hat}.

    Returns
    -------
    dict con chiavi:
        'lag_rounds'   : int in [-T/2, T/2] circa (best shift)
        'lag_radians'  : float in [-π, π]
        'amp_ratio'    : float >= 0 (⟨r_hat⟩ / ⟨r_true⟩)
    """
    if pi_true_seq.shape != pi_hat_seq.shape:
        raise ValueError("Sequenze π_true e π_hat devono avere stessa shape (T,3).")
    T, K = pi_true_seq.shape
    if K != 3:
        raise ValueError("lag_and_amplitude supporta K=3.")

    xy_true = simplex_embed_2d(pi_true_seq)  # (T,2)
    xy_hat = simplex_embed_2d(pi_hat_seq)    # (T,2)

    theta_true = _angles_from_xy(xy_true)
    theta_hat = _angles_from_xy(xy_hat)

    # phasors
    z_true = np.exp(1j * theta_true)
    z_hat = np.exp(1j * theta_hat)

    # cross-correlation sui phasors complessi per stimare lo shift migliore
    best_tau = 0
    best_val = -np.inf
    for tau in range(-T + 1, T):
        # shift circolare
        z_hat_shift = np.roll(z_hat, shift=tau, axis=0)
        # usare la parte reale della correlazione (coerenza di fase)
        val = np.real(np.vdot(z_true, z_hat_shift))  # vdot = conj(z_true)·z_hat_shift
        if val > best_val:
            best_val = val
            best_tau = tau

    lag_rounds = int(best_tau)
    lag_radians = float(2.0 * np.pi * lag_rounds / float(T))

    # ampiezze radiali (distanza dal baricentro)
    r_true = _radial_distance(xy_true)
    r_hat = _radial_distance(xy_hat)
    # evitare divisioni per zero
    denom = float(np.mean(r_true)) if np.mean(r_true) > 1e-8 else 1.0
    amp_ratio = float(np.mean(r_hat) / denom)

    return {
        "lag_rounds": lag_rounds,
        "lag_radians": lag_radians,
        "amp_ratio": amp_ratio,
    }


# ----------------------------
# Forgetting & equity
# ----------------------------
def forgetting_index(
    m_by_mu_over_time: np.ndarray,  # shape (K, T)
    exposure_mask: Optional[np.ndarray] = None,  # bool (K, T) True=alta esposizione
) -> np.ndarray:
    """
    Per ciascun archetipo μ, definisce FI_μ = m_μ(t*) - m_μ(T-1),
    dove t* è l'ultimo round con "alta esposizione" (se noto), altrimenti t* = argmax_t m_μ(t).

    Returns
    -------
    FI : (K,) float64 (valori positivi indicano perdita rispetto all'ultimo stato "buono")
    """
    m = np.asarray(m_by_mu_over_time, dtype=float)
    if m.ndim != 2:
        raise ValueError("m_by_mu_over_time deve avere shape (K, T).")
    K, T = m.shape
    FI = np.zeros((K,), dtype=float)

    for mu in range(K):
        if exposure_mask is not None:
            mask = np.asarray(exposure_mask[mu], dtype=bool)
            if mask.size != T:
                raise ValueError("exposure_mask dimension mismatch.")
            idx = np.where(mask)[0]
            if idx.size > 0:
                t_star = int(idx.max())
            else:
                t_star = int(np.argmax(m[mu]))
        else:
            t_star = int(np.argmax(m[mu]))
        FI[mu] = float(m[mu, t_star] - m[mu, T - 1])
    return FI


def equity_measures(values: np.ndarray) -> Dict[str, float]:
    """
    Misure di equità tra archetipi, per un vettore di magnetizzazioni (K,) o una matrice (K, T):
    - varianza
    - Gini

    Se (K, T), restituisce le medie su T dei due indici.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        var = float(np.var(x))
        gini = float(gini_coefficient(np.clip(x, 0.0, None)))
        return {"variance": var, "gini": gini}
    elif x.ndim == 2:
        K, T = x.shape
        var_t = np.var(x, axis=0)             # (T,)
        gini_t = np.array([gini_coefficient(np.clip(x[:, t], 0.0, None)) for t in range(T)])
        return {"variance": float(np.mean(var_t)), "gini": float(np.mean(gini_t))}
    else:
        raise ValueError("values deve essere (K,) o (K, T).")


# ----------------------------
# Convenience: metriche round base
# ----------------------------
def compute_round_metrics(
    *,
    E_t: np.ndarray,                 # (L, M_c, N)
    J_KS: np.ndarray,                # (N, N)
    xi_ref_for_pi: np.ndarray,       # (S, N) ⇒ useremo i primi K come riferimenti
    pi_true_t: np.ndarray,           # (K,)
    method_keff: str = "shuffle",
    M_eff: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calcola metriche di base usate in Exp-06:
      - π̂_t stimata dagli esempi (classificazione su ξ_ref allineati)
      - TV(π_t, π̂_t)
      - K_eff su J_KS (wrapper codebase)
    """
    pi_hat_t = estimate_pi_hat_from_examples(xi_ref_for_pi, E_t)
    pi_true_t = np.asarray(pi_true_t, dtype=float)
    if pi_true_t.sum() <= 0:
        pi_true_t = np.ones_like(pi_hat_t) / float(pi_hat_t.size)
    else:
        pi_true_t = pi_true_t / pi_true_t.sum()

    TV_t = tv_distance(pi_true_t, pi_hat_t)
    K_eff, keep_mask, info = keff_and_info(J_KS, method=method_keff, M_eff=M_eff)

    eigvals = info.get("eigvals", None)
    return {
        "pi_hat": pi_hat_t.tolist(),
        "pi_true": pi_true_t.tolist(),
        "TV_pi": float(TV_t),
        "K_eff": int(K_eff),
        "keff_info": {
            "threshold": float(info.get("threshold", np.nan)),
            "method": str(info.get("method", method_keff)),
            "eigvals": eigvals.tolist() if isinstance(eigvals, np.ndarray) else None,
        },
    }
