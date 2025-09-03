# src/unsup/estimators.py
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np

from .functions import unsupervised_J, Hebb_J  # reuse existing stable primitives


__all__ = [
    "build_unsup_J_single",
    "blend_with_memory",
]


def _symmetrize(J: np.ndarray) -> np.ndarray:
    """Rende J esattamente simmetrica (tollerante a lievi asimmetrie numeriche)."""
    J = np.asarray(J, dtype=np.float32)
    return 0.5 * (J + J.T)


def build_unsup_J_single(ETA_t: np.ndarray, K: int) -> Tuple[np.ndarray, int]:
    """
    Stima J_unsup per il SOLO round t, mediando su layer/client.

    Parametri
    ---------
    ETA_t : (L, M_c, N)
        Esempi del round corrente (single-mode).
    K : int
        Numero di archetipi (usato per definire M_eff = max(1, M_c // K)).

    Returns
    -------
    J_unsup : (N, N) float32
        Media delle matrici unsupervised per layer.
    M_eff : int
        Effective sample size per-layer usato in unsupervised_J (per coerenza con stima K_eff MP).
    """
    if ETA_t.ndim != 3:
        raise ValueError("ETA_t deve avere shape (L, M_c, N).")
    L, M_c, N = ETA_t.shape
    if K <= 0:
        raise ValueError("K deve essere > 0.")
    M_eff = max(1, int(M_c // K))

    # Calcola J_unsup per ciascun layer e media
    Js = [unsupervised_J(np.asarray(ETA_t[l], dtype=np.float32), M_eff) for l in range(L)]
    J_unsup = np.sum(Js, axis=0) / float(L)

    return _symmetrize(J_unsup), M_eff


def blend_with_memory(
    J_unsup: np.ndarray,
    xi_prev: Optional[np.ndarray],
    w: float,
) -> np.ndarray:
    """
    Fonde la stima unsupervised del round corrente con la memoria Hebb del round precedente.

    Regole:
      - Se xi_prev è None -> ritorna J_unsup (primo round).
      - Altrimenti: J_rec = w * J_unsup + (1-w) * J_hebb_prev, con J_hebb_prev = Hebb_J(xi_prev).

    Parametri
    ---------
    J_unsup : (N, N)
        Stima appena ottenuta dal round corrente.
    xi_prev : (S, N) oppure None
        Pattern disentangled ottenuti fino al round precedente (S può essere ≠ K).
    w : float in [0, 1]
        Peso di blending (w alto = più fiducia nel dato corrente).

    Returns
    -------
    J_rec : (N, N)
        Matrice ricostruita post-blend, simmetrizzata.
    """
    J_unsup = _symmetrize(J_unsup)
    if xi_prev is None:
        return J_unsup

    xi_prev = np.asarray(xi_prev, dtype=np.float32)
    if xi_prev.ndim != 2:
        raise ValueError("xi_prev deve avere shape (S, N).")

    J_hebb_prev = Hebb_J(xi_prev)  # equivalente a unsupervised_J(xi_prev, M=1) per binario ±1
    J_rec = float(w) * J_unsup + float(1.0 - w) * J_hebb_prev
    return _symmetrize(J_rec)
