# -*- coding: utf-8 -*-
"""
Valutazione post-hoc con rete di Hopfield (single-mode).

Dato J_server (matrice hebbiana finale del server) e i veri archetipi ξ_true,
simuliamo la dinamica di Hopfield partendo da stati iniziali fortemente corrotti
di ciascun archetipo e misuriamo la magnetizzazione finale. Questo consente di
testare l'ipotesi "archetipi più esposti ⇒ retrieval migliore".

API principali
--------------
- corrupt_like_archetype(...)    : genera σ0 con overlap iniziale controllato.
- run_hopfield_test(...)         : esegue la dinamica Hopfield su più repliche.
- eval_retrieval_vs_exposure(..) : aggrega per archetipo e correla con esposizione.

Compatibilità
-------------
Fa uso di `Hopfield_Network` definita in `src.unsup.networks`. Se desideri
iniettare J_server direttamente (senza passare da prepare(η)), è sufficiente:
    net = Hopfield_Network()
    net.N = J_server.shape[0]
    net.J = J_server
e poi chiamare `net.dynamics(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

# Import locale
from src.unsup.networks import Hopfield_Network


__all__ = [
    "corrupt_like_archetype",
    "run_hopfield_test",
    "eval_retrieval_vs_exposure",
]


def corrupt_like_archetype(
    xi_true: np.ndarray,
    reps_per_archetype: int,
    start_overlap: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera stati iniziali σ0 per Hopfield come versioni corrotte degli archetipi.

    Parametri
    ---------
    xi_true : (K, N) in {±1}
    reps_per_archetype : int
        Numero di repliche per ciascun archetipo.
    start_overlap : float in [0, 1]
        Overlap atteso iniziale con l'archetipo (0 = rumore puro, 1 = identico).
        Implementazione tramite flip indipendenti con P(flip) = (1 - start_overlap)/2.
    rng : np.random.Generator opzionale

    Returns
    -------
    σ0 : (K * reps_per_archetype, N) in {±1}
        Stati iniziali per la dinamica di Hopfield.
    targets : (K * reps_per_archetype,)
        Indici degli archetipi corrispondenti a ciascuno stato iniziale.
    """
    rng = np.random.default_rng() if rng is None else rng
    K, N = xi_true.shape
    p_flip = (1.0 - float(start_overlap)) * 0.5
    total = K * int(reps_per_archetype)
    σ0 = np.empty((total, N), dtype=int)
    targets = np.empty((total,), dtype=int)

    t = 0
    for μ in range(K):
        flips = rng.random(size=(reps_per_archetype, N)) < p_flip
        # flip bit ⇒ moltiplicare per -1
        σ0[t:t + reps_per_archetype] = np.where(flips, -xi_true[μ], xi_true[μ]).astype(int)
        targets[t:t + reps_per_archetype] = μ
        t += reps_per_archetype
    return σ0, targets


@dataclass
class HopfieldEvalResult:
    """Risultati aggregati della valutazione Hopfield."""
    magnetization_by_mu: Dict[int, np.ndarray]  # μ -> array (reps,)
    mean_by_mu: Dict[int, float]
    std_by_mu: Dict[int, float]
    overall_mean: float
    overall_std: float


def run_hopfield_test(
    J_server: np.ndarray,
    xi_true: np.ndarray,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> HopfieldEvalResult:
    """
    Esegue la dinamica Hopfield su J_server con iniziali corrotti degli archetipi.

    Parametri
    ---------
    J_server : (N, N)
        Matrice sinaptica da valutare (quella finale del server).
    xi_true : (K, N) in {±1}
        Archetipi di riferimento per il calcolo delle magnetizzazioni.
    beta : float
        Inverse temperature per la dinamica (controlla la spinta del segnale).
    updates : int
        Numero di aggiornamenti paralleli.
    reps_per_archetype : int
        Repliche per ogni archetipo.
    start_overlap : float
        Overlap iniziale desiderato con l'archetipo (0..1).
    rng : np.random.Generator opzionale

    Returns
    -------
    HopfieldEvalResult
        Magnetizzazioni per archetipo e statistiche aggregate.
    """
    rng = np.random.default_rng() if rng is None else rng
    K, N = xi_true.shape
    # Stati iniziali e mapping verso il "bersaglio" μ
    σ0, targets = corrupt_like_archetype(xi_true, reps_per_archetype, start_overlap, rng=rng)

    # Prepara rete di Hopfield e inietta direttamente J_server
    net = Hopfield_Network()
    net.N = int(J_server.shape[0])
    net.J = np.asarray(J_server, dtype=np.float32)

    # Dinamica parallela
    net.dynamics(σ0.astype(np.float32), β=beta, updates=updates, mode="parallel")
    σf = np.asarray(net.σ, dtype=int)
    if σf is None:
        raise RuntimeError("Hopfield_Network.dynamics non ha prodotto stati finali.")

    # Magnetizzazione finale verso il rispettivo target
    # m = |<σf, ξ_target>|/N
    mag_by_mu: Dict[int, list] = {μ: [] for μ in range(K)}
    for i in range(σf.shape[0]):
        μ = int(targets[i])
        m = float(np.abs(np.dot(σf[i], xi_true[μ])) / N)
        mag_by_mu[μ].append(m)

    # Aggrega
    mag_by_mu_np: Dict[int, np.ndarray] = {μ: np.asarray(mag_by_mu[μ], dtype=float) for μ in range(K)}
    mean_by_mu = {μ: float(v.mean()) if v.size else 0.0 for μ, v in mag_by_mu_np.items()}
    std_by_mu = {μ: float(v.std(ddof=1)) if v.size > 1 else 0.0 for μ, v in mag_by_mu_np.items()}
    all_vals = np.concatenate([v for v in mag_by_mu_np.values() if v.size], axis=0) if any(
        v.size for v in mag_by_mu_np.values()
    ) else np.array([0.0])

    return HopfieldEvalResult(
        magnetization_by_mu=mag_by_mu_np,
        mean_by_mu=mean_by_mu,
        std_by_mu=std_by_mu,
        overall_mean=float(all_vals.mean()),
        overall_std=float(all_vals.std(ddof=1)) if all_vals.size > 1 else 0.0,
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size == 0:
        return float("nan")
    xc = x - x.mean(); yc = y - y.mean()
    num = float(np.dot(xc, yc))
    den = float(np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
    return num / den


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # rank correlation simple implementation
    def _ranks(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1, dtype=float)
        return ranks
    xr = _ranks(np.asarray(x, dtype=float))
    yr = _ranks(np.asarray(y, dtype=float))
    return _pearson(xr, yr)


def eval_retrieval_vs_exposure(
    J_server: np.ndarray,
    xi_true: np.ndarray,
    exposure_counts: np.ndarray,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    Esegue `run_hopfield_test` e correla la magnetizzazione media per archetipo
    con il numero di esposizioni (quante volte l'archetipo è apparso nei round).

    Returns
    -------
    dict con chiavi:
      - 'mean_by_mu'       : dict μ -> float
      - 'std_by_mu'        : dict μ -> float
      - 'pearson'          : float
      - 'spearman'         : float
      - 'overall_mean/std' : float, float
      - 'magnetization_by_mu' : μ -> np.ndarray (tutte le repliche)
    """
    res = run_hopfield_test(
        J_server=J_server,
        xi_true=xi_true,
        beta=beta,
        updates=updates,
        reps_per_archetype=reps_per_archetype,
        start_overlap=start_overlap,
        rng=rng,
    )
    K = xi_true.shape[0]
    expo = np.asarray(exposure_counts, dtype=float).reshape(K)
    means = np.array([res.mean_by_mu.get(μ, 0.0) for μ in range(K)], dtype=float)

    return {
        "mean_by_mu": res.mean_by_mu,
        "std_by_mu": res.std_by_mu,
        "overall_mean": res.overall_mean,
        "overall_std": res.overall_std,
        "pearson": _pearson(expo, means),
        "spearman": _spearman(expo, means),
        "magnetization_by_mu": res.magnetization_by_mu,
    }
