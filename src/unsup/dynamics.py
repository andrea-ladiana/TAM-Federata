# -*- coding: utf-8 -*-
"""
Dinamiche e disentangling (single mode).

Espone:
- `new_round`: estrazione batch round-t in single-mode.
- `eigen_cut`: selezione autovettori informativi da J_KS.
- `init_candidates_from_eigs`: generazione σ(0) via mixture di segni.
- `disentangling`: dinamica TAM + pruning + magnetizzazioni.
- `dis_check`: wrapper che garantisce almeno K candidati (con fallback).

Dipendenze chiave:
- src.unsup.networks.TAM_Network
- src.unsup.functions.propagate_J (a monte, non qui)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np

# Import locali (nessuna dipendenza da TF)
from src.unsup.networks import TAM_Network
from src.unsup.config import TAMParams, SpectralParams


__all__ = [
    "new_round",
    "eigen_cut",
    "init_candidates_from_eigs",
    "disentangling",
    "dis_check",
]


def new_round(ETA: np.ndarray, t: int) -> np.ndarray:
    """
    Estrae il batch del *round t* in modalità single.

    Parameters
    ----------
    ETA : np.ndarray
        Tensore federato shape (L, T, M_c, N).
    t : int
        Indice round (0..T-1).

    Returns
    -------
    np.ndarray
        Batch shape (L, M_c, N) per il round t.
    """
    if ETA.ndim != 4:
        raise ValueError(f"ETA atteso di rank 4 (L,T,M_c,N); ricevuto shape={ETA.shape}")
    L, T, Mc, N = ETA.shape
    if not (0 <= t < T):
        raise IndexError(f"round t={t} fuori range [0, {T-1}]")
    return ETA[:, t, :, :]


def eigen_cut(JKS: np.ndarray, tau: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Taglio spettrale su J_KS: seleziona autovettori con autovalori > tau.

    Returns
    -------
    (vals_sel, vecs_sel)
      vals_sel : (K_eff,)
      vecs_sel : (K_eff, N)  # righe = autovettori trasposti per coerenza con generazione
    """
    if JKS.ndim != 2 or JKS.shape[0] != JKS.shape[1]:
        raise ValueError("JKS deve essere quadrata.")
    # Use symmetric eigendecomposition for speed and stability
    vals, vecs = np.linalg.eigh(JKS)
    mask = vals > float(tau)
    V = vecs[:, mask].T  # (K_eff, N)
    return vals[mask], V


def init_candidates_from_eigs(V: np.ndarray, L: int, s: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Genera stati iniziali σ(0) combinando gli autovettori informativi con
    pesi gaussiani e prendendo il segno.

    Parameters
    ----------
    V : np.ndarray
        Autovettori selezionati, shape (K_eff, N).
    L : int
        Numero layer/client (serve a tarare il numero di mix).
    s : int | None
        Numero di combinazioni. Di default segue la formula empirica del notebook:
        s = 10 * int(K_eff / L * log(K_eff / 0.01)).
    rng : np.random.Generator | None
        RNG opzionale per riproducibilità.

    Returns
    -------
    np.ndarray
        σ(0) shape (s, N), valori in {-1, +1}.
    """
    if V.ndim != 2:
        raise ValueError("V deve avere shape (K_eff, N).")
    K_eff, N = V.shape
    if K_eff == 0:
        # fallback: nessun autovettore informativo
        return np.empty((0, N), dtype=int)

    if s is None:
        s = max(1, 10 * int((K_eff / max(1, L)) * np.log(K_eff / 0.01)))

    rng = np.random.default_rng() if rng is None else rng
    W = rng.normal(0.0, 1.0, size=(s, K_eff))  # (s, K_eff)
    # Correct multiplication: mix weights W (s, K_eff) with eigenvectors V (K_eff, N)
    # to obtain Z of shape (s, N). Previous code used (V @ W.T).T which requires
    # N == K_eff and caused a ValueError when they differ.
    Z = W @ V  # (s, N)
    sigma0 = np.where(Z >= 0.0, 1, -1).astype(int)
    return sigma0


@dataclass
class _DisResult:
    xi_r: np.ndarray      # (K_tilde, N)
    magnetisations: np.ndarray  # (K_tilde,)


def _prune_and_score(xi_r: np.ndarray, JKS: np.ndarray, xi_true: np.ndarray, spec: SpectralParams) -> _DisResult:
    """
    Applica:
      1) allineamento spettrale: <ξ, JKS ξ>/N >= rho
      2) pruning per overlap mutuo: |<ξ_a, ξ_b>|/N <= qthr
    e calcola magnetizzazioni: m_a = max_b |<ξ_a, ξ_b* >|/N
    """
    if xi_r.size == 0:
        return _DisResult(xi_r=np.empty((0, JKS.shape[0]), dtype=int), magnetisations=np.array([]))

    N = JKS.shape[0]
    # 1) spectral alignment
    # quad[a,i] = sum_j JKS[i,j] * xi_r[a,j] = (xi_r @ JKS^T)[a,i]
    quad = (xi_r @ JKS.T)
    align = (np.einsum("ai,ai->a", xi_r, quad) / N)   # (K_tilde,)
    keep = align >= float(spec.rho)
    xi_r = xi_r[keep]
    if xi_r.size == 0:
        return _DisResult(xi_r=np.empty((0, N), dtype=int), magnetisations=np.array([]))

    # 2) mutual-overlap pruning (sopra diagonale)
    q = np.abs((xi_r @ xi_r.T) / N).astype(float)
    iu = np.triu_indices(q.shape[0], k=1)
    bad_pairs = q[iu] > float(spec.qthr)
    if np.any(bad_pairs):
        # Strategia semplice: rimuovi gli indici ripetuti (first-occur policy).
        A = q.copy()
        to_remove: set[int] = set()
        for i, j in zip(*iu):
            if A[i, j] > spec.qthr:
                # rimuovi j (arbitrario/greedy)
                to_remove.add(j)
        if to_remove:
            mask = np.ones(xi_r.shape[0], dtype=bool)
            mask[list(sorted(to_remove))] = False
            xi_r = xi_r[mask]

    # magnetizzazioni (max overlap con veri archetipi)
    M = np.abs(xi_r @ xi_true.T) / N  # (K_tilde, K)
    m = M.max(axis=1) if M.size else np.array([])
    return _DisResult(xi_r=xi_r, magnetisations=m)


def disentangling(
    V: np.ndarray,
    K: int,
    L: int,
    J_rec: np.ndarray,
    JKS_iter: np.ndarray,
    xi_true: np.ndarray,
    tam: TAMParams,
    spec: SpectralParams,
    xi_prev: np.ndarray | int = -10,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Esegue: init da autovettori -> dinamica TAM -> pruning -> magnetizzazioni.

    Parameters
    ----------
    V : (K_eff, N) autovettori selezionati da `eigen_cut`.
    J_rec : matrice di ricostruzione pre-propagazione (per `prepare` della TAM).
    JKS_iter : matrice J dopo propagazione (per check spettrale).
    xi_prev : se passato (array) concatena le ricostruzioni precedenti (fallback loop).

    Returns
    -------
    (xi_r, m)
      xi_r : (K_tilde, N) candidati finali
      m    : (K_tilde,) magnetizzazioni (max overlap vs archetipi veri)
    """
    N = J_rec.shape[0]
    sigma0 = init_candidates_from_eigs(V, L=L)
    if sigma0.size == 0:
        return np.empty((0, N), dtype=int), np.array([])

    # Replicazione sui layer e preparazione input TAM
    # σ shape attesa da TAM_Network: (s, L, N)
    sigma = np.repeat(sigma0[:, None, :], L, axis=1)

    net = TAM_Network()
    net.prepare(J_rec, L)
    net.dynamics(
        sigma,
        tam.beta_T,
        tam.lam,
        tam.h_in,
        updates=tam.updates,
        noise_scale=tam.noise_scale,
        min_scale=tam.min_scale,
        anneal=tam.anneal,
        schedule=tam.schedule,
        show_progress=show_progress,
        desc="TAM (single)",
    )

    if net.σ is None:
        return np.empty((0, N), dtype=int), np.array([])

    xi_new = np.reshape(np.asarray(net.σ), (sigma0.shape[0] * L, N)).astype(int)

    if isinstance(xi_prev, np.ndarray) and np.mean(xi_prev) != -10:
        xi_all = np.vstack([xi_new, xi_prev])
    else:
        xi_all = xi_new

    # Pruning + magnetizzazioni
    res = _prune_and_score(xi_all, JKS_iter, xi_true, spec)

    # Fallback se tutto rimosso: conserva primi max(1, min(K, K̃))
    if res.xi_r.shape[0] == 0:
        keep = max(1, min(K, xi_all.shape[0]))
        xi_fbk = xi_all[:keep]
        res = _prune_and_score(xi_fbk, JKS_iter, xi_true, spec)

    return res.xi_r, res.magnetisations


def dis_check(
    V: np.ndarray,
    K: int,
    L: int,
    J_rec: np.ndarray,
    JKS_iter: np.ndarray,
    xi_true: np.ndarray,
    tam: TAMParams,
    spec: SpectralParams,
    show_progress: bool = False,
    max_attempts: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper robusto: ripete il disentangling concatenando le soluzioni finché non
    ottiene almeno K candidati (oppure esaurisce i tentativi).

    Returns
    -------
    (xi_r, m)
    """
    xi_r, m = disentangling(
        V, K, L, J_rec, JKS_iter, xi_true, tam=tam, spec=spec, xi_prev=-10, show_progress=show_progress
    )
    attempts = 0
    while xi_r.shape[0] < K and attempts < max_attempts:
        attempts += 1
        xi_r, m = disentangling(
            V, K, L, J_rec, JKS_iter, xi_true, tam=tam, spec=spec, xi_prev=xi_r, show_progress=show_progress
        )
    return xi_r, m
