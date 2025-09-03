# -*- coding: utf-8 -*-
"""
Metriche core per exp_01 (single):

- Frobenius relativo tra J_KS e J*.
- Overlap/matching Ungherese per retrieval medio.
- Magnetizzazioni per-candidato.
- Coverage per round.
- Z-score robusto (median/MAD).
- Wrapper semplici per serie K_eff (se vuoi agganciarti a spectrum.py).

Formule e definizioni sono allineate alle sezioni *Metrics* e *Disentangling*
(del report): FRO, matching su costo 1-M, magnetizzazioni come max overlap, ecc.
"""
from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np


def frobenius_relative(J_hat: np.ndarray, J_star: np.ndarray, eps: float = 1e-9) -> float:
    """
    ||J_hat - J_star||_F / (||J_star||_F + eps)
    """
    num = np.linalg.norm(J_hat - J_star, ord="fro")
    den = np.linalg.norm(J_star, ord="fro") + float(eps)
    return float(num / den)


def overlap_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    M_ab = |<A_a, B_b>| / N, dove A.shape=(Ka, N), B.shape=(Kb, N).
    """
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    N = A.shape[1]
    return np.abs(A @ B.T) / float(N)


def magnetisations(xi_r: np.ndarray, xi_true: np.ndarray) -> np.ndarray:
    """
    m_a = max_b M_ab
    """
    M = overlap_matrix(xi_r, xi_true)
    return M.max(axis=1) if M.size else np.array([])


def retrieval_mean_hungarian(xi_est: np.ndarray, xi_true: np.ndarray) -> float:
    """
    Retrieval medio dopo matching ottimo (costo 1 - M) via algoritmo Ungherese.

    Se Ka != Kb, normalizza in modo conservativo come nel codice degli esperimenti:
    - se Ka < Kb: somma overlaps matched / Kb
    - altrimenti: media semplice degli overlaps matched
    """
    from scipy.optimize import linear_sum_assignment

    Ka, N = xi_est.shape
    Kb, N2 = xi_true.shape
    if N != N2:
        raise ValueError("Dimensioni incompatibili per overlap.")
    M = overlap_matrix(xi_est, xi_true)
    cost = 1.0 - M
    rI, cI = linear_sum_assignment(cost)
    overlaps = M[rI, cI]
    if Ka < Kb:
        return float(overlaps.sum() / Kb)
    return float(overlaps.mean())


def coverage_fraction(labels_seen: Iterable[int], K: int) -> float:
    """
    Coverage istantaneo di un round: frazione di archetipi distinti visti (∈ [0,1]).
    """
    S = set(int(x) for x in labels_seen)
    return float(len(S) / max(1, K))


def robust_z(values: Iterable[float]) -> List[float]:
    """
    Z-score robusto: (x - med) / (1.4826 * MAD)
    Restituisce una lista per compatibilità con pipelines di aggregazione.
    """
    x = np.asarray(list(values), dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    z = (x - med) / (1.4826 * mad)
    return list(z)


# --- (Opzionali) util per serie Keff: questi sono thin wrapper se vuoi usare spectrum.py ---

def keff_from_eigs(vals: np.ndarray, mp_edge: float) -> int:
    """
    Conta # { λ_i > mp_edge } (comodo se hai già calcolato edge MP altrove).
    """
    return int(np.count_nonzero(np.asarray(vals, float) > float(mp_edge)))
