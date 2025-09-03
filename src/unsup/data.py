# src/unsup/data.py
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Sequence, Dict

import numpy as np

# Reuse generator for true patterns from the existing functions module
# (avoid duplications and keep one single source of truth)
from .functions import gen_patterns as _gen_patterns  # noqa: F401


__all__ = [
    "make_client_subsets",
    "gen_dataset_partial_archetypes",
    "new_round_single",
    "compute_round_coverage",
    "count_exposures",
]


def make_client_subsets(
    K: int,
    L: int,
    K_per_client: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Crea i sottoinsiemi di archetipi per ciascun client garantendo (quando fattibile)
    che l'unione copra tutti gli archetipi {0,...,K-1}.

    Vincolo di fattibilità: L * K_per_client >= K (altrimenti non è possibile coprire tutti).

    Strategia:
      1) Round-robin su una permutazione casuale per garantire copertura.
      2) Riempimento fino a K_per_client per ciascun client con scelte senza rimpiazzo.

    Returns
    -------
    subsets : list of list[int]
        Lista di lunghezza L; ogni entry è l'insieme (ordinato) di archetipi assegnati.
    """
    if K <= 0 or L <= 0:
        raise ValueError("K e L devono essere positivi.")
    if K_per_client <= 0:
        raise ValueError("K_per_client deve essere > 0.")
    if K_per_client > K:
        raise ValueError("K_per_client non può superare K.")
    if L * K_per_client < K:
        raise ValueError(
            f"Impossibile coprire {K} archetipi con L={L} e K_per_client={K_per_client} "
            f"(L*K_per_client={L*K_per_client} < K)."
        )

    subsets: List[List[int]] = [[] for _ in range(L)]
    perm = rng.permutation(K)

    # Passo 1: round-robin per la copertura
    for i, mu in enumerate(perm):
        subsets[i % L].append(int(mu))

    # Passo 2: riempimento fino a K_per_client con scelte senza rimpiazzo locali
    for l in range(L):
        need = K_per_client - len(subsets[l])
        if need <= 0:
            continue
        pool = [mu for mu in range(K) if mu not in subsets[l]]
        if need > len(pool):
            # Se qui capitasse, significa che K_per_client è molto grande e gli altri client
            # hanno già “assorbito” quasi tutto: in tal caso consenti duplicati controllati.
            extra = rng.choice(range(K), size=need, replace=False)
        else:
            extra = rng.choice(pool, size=need, replace=False)
        subsets[l].extend(int(x) for x in extra)

    # Normalizza (ordina) e deduplica
    return [sorted(set(s)) for s in subsets]


def gen_dataset_partial_archetypes(
    xi_true: np.ndarray,
    M_total: int,
    r_ex: float,
    n_batch: int,
    L: int,
    client_subsets: Sequence[Sequence[int]],
    rng: np.random.Generator,
    use_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera dataset UNSUPERVISED in SINGLE-mode con copertura parziale per client.

    Ogni client l ha un sottoinsieme consentito di archetipi (client_subsets[l]).
    Per ogni round t, si generano M_c esempi campionando archetipi solo dal sottoinsieme:
        eta[l, t, m] = chi * xi_true[mu], chi ∈ {±1}^N, con P(chi_i = +1) = (1 + r_ex)/2.

    Parametri
    ---------
    xi_true : (K, N)
        Archetipi veri (±1).
    M_total : int
        Numero desiderato di esempi totali (su TUTTI i client e TUTTI i round).
        Ogni client e round riceve M_c = ceil(M_total / (L * n_batch)) esempi.
    r_ex : float in [0, 1]
        Livello medio di qualità/correlazione degli esempi.
    n_batch : int
        Numero di round.
    L : int
        Numero di client.
    client_subsets : sequence of sequences
        Liste degli archetipi consentiti per client (indici in [0, K-1]).
    rng : np.random.Generator
        Generatore di numeri casuali.
    use_tqdm : bool
        Se True, mostra barre di avanzamento (se disponibile).

    Returns
    -------
    ETA : (L, n_batch, M_c, N) float32
        Esempi generati per client/round.
    labels : (L, n_batch, M_c) int32
        Indice di archetipo usato per ciascun esempio (per comodità/metriche).
    """
    K, N = xi_true.shape
    if len(client_subsets) != L:
        raise ValueError("client_subsets deve avere lunghezza L.")

    M_c = int(math.ceil(M_total / float(L * n_batch)))
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)

    p_keep = 0.5 * (1.0 + float(r_ex))
    itL = range(L)

    # TQDM opzionale, senza dipendenza hard
    if use_tqdm:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            itL = _tqdm(itL, desc="dataset: clients", leave=False)
        except Exception:
            pass

    for l in itL:
        allowed = list(client_subsets[l])
        if not allowed:
            raise ValueError(f"Client {l} ha subset vuoto.")
        itT = range(n_batch)
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                itT = _tqdm(itT, desc=f"client {l} rounds", leave=False)
            except Exception:
                pass

        for t in itT:
            mus = rng.choice(allowed, size=M_c, replace=True).astype(int)  # (M_c,)
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0).astype(np.float32)  # (M_c, N)
            xi_sel = xi_true[mus].astype(np.float32)                         # (M_c, N)
            ETA[l, t] = (chi * xi_sel).astype(np.float32)
            labels[l, t] = mus.astype(np.int32)

    return ETA, labels


def new_round_single(ETA: np.ndarray, t: int) -> np.ndarray:
    """
    Estrae il tensore del SOLO round t: (L, M_c, N) da un ETA (L, n_batch, M_c, N).
    """
    if ETA.ndim != 4:
        raise ValueError("ETA deve avere shape (L, n_batch, M_c, N).")
    return np.asarray(ETA[:, t, :, :], dtype=np.float32)


def compute_round_coverage(labels_t: np.ndarray, K: int) -> float:
    """
    Coverage round-wise: frazione di archetipi distinti osservati al round t.

    Parametri
    ---------
    labels_t : (L, M_c) oppure (M_c,)
        Etichette (indici archetipi) del solo round t (eventualmente già aggregate sui client).
    K : int
        Numero totale di archetipi.

    Returns
    -------
    cov : float in [0,1]
    """
    if labels_t.ndim == 1:
        seen = set(int(mu) for mu in labels_t)
    elif labels_t.ndim == 2:
        L, M_c = labels_t.shape
        seen = set(int(mu) for l in range(L) for mu in labels_t[l])
    else:
        raise ValueError("labels_t deve avere ndim 1 o 2.")
    return len(seen) / float(K)


def count_exposures(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Conta quante volte ciascun archetipo è stato “visto” in tutto il dataset.
    Utile per correlare “esposizione” e magnetizzazione nel test Hopfield post-hoc.

    Parametri
    ---------
    labels : (L, n_batch, M_c)
    K : int

    Returns
    -------
    exposure : (K,) np.ndarray di int
    """
    if labels.ndim != 3:
        raise ValueError("labels deve avere shape (L, n_batch, M_c).")
    expo = np.zeros((K,), dtype=np.int64)
    flat = labels.reshape(-1)
    vals, cnts = np.unique(flat, return_counts=True)
    expo[vals.astype(int)] = cnts.astype(int)
    return expo
