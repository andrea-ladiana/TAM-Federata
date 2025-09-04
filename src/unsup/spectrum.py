# src/unsup/spectrum.py
from __future__ import annotations
from typing import Tuple, Dict, Any

import numpy as np

from .functions import estimate_K_eff_from_J as _estimate_K_eff_from_J


__all__ = [
    "eigen_cut",
    "estimate_keff",
]


def _symmetrize(J: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(J, dtype=np.float32) + np.asarray(J, dtype=np.float32).T)


def eigen_cut(
    J: np.ndarray,
    tau: float = 0.5,
    return_info: bool = False,
) -> Tuple[np.ndarray, int] | Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Seleziona gli autovettori associati ad autovalori > tau (default 0.5),
    restituendoli come righe (k_eff, N), compatibile con `dis_check`.

    Parametri
    ---------
    J : (N, N)
        Matrice (leggermente asimmetrica tollerata; viene simmetrizzata).
    tau : float
        Soglia su autovalori reali.
    return_info : bool
        Se True, restituisce anche info diagnostiche (evals, mask).

    Returns
    -------
    V_sel : (k_eff, N)
        Autovettori selezionati (trasposti per coerenza con codice esistente).
    k_eff : int
        Numero di componenti selezionate.
    info : dict (opzionale)
        {'evals': evals_desc, 'keep_mask': mask_desc}
    """
    J_sym = _symmetrize(J)
    # Use symmetric eigendecomposition for speed and stability
    evals, evecs = np.linalg.eigh(J_sym)

    # Ordina per autovalore decrescente
    order = np.argsort(evals)[::-1]
    evals_desc = evals[order]
    evecs_desc = evecs[:, order]

    keep_mask = evals_desc > float(tau)
    V_sel = evecs_desc[:, keep_mask].T  # (k_eff, N)
    k_eff = int(V_sel.shape[0])

    if return_info:
        return V_sel, k_eff, {"evals": evals_desc, "keep_mask": keep_mask}
    return V_sel, k_eff


def estimate_keff(
    J: np.ndarray,
    method: str = "shuffle",
    **kwargs,
) -> Tuple[int, np.ndarray, dict]:
    """
    Wrapper “pass-through” per la stima di K_eff.

    Parametri
    ---------
    J : (N, N)
        Matrice (propagata o meno) su cui stimare K_eff.
    method : {'shuffle', 'mp'}
        Metodo sottostante.
    **kwargs :
        Parametri addizionali inoltrati a `functions.estimate_K_eff_from_J`,
        p.es. alpha, n_random, M_eff (necessario per 'mp'), data_var.

    Returns
    -------
    K_eff : int
    keep_mask : (N,) bool
        Maschera sugli autovalori ORDINATI in senso decrescente.
    info : dict
        Dizionario diagnostico dal metodo sottostante (evals, soglie, ecc.).
    """
    # La funzione sottostante esegue già l'ordinamento discendente degli autovalori.
    return _estimate_K_eff_from_J(np.asarray(J, dtype=np.float32), method=method, **kwargs)
