# -*- coding: utf-8 -*-
"""
Adapter per dataset reali (MNIST) in modalità HFL (single-mode).

Funzioni principali:
- load_mnist(...)                  : caricamento sicuro (facoltativo, via torchvision se disponibile)
- binarize_images(...)             : binarizzazione in {±1} con threshold
- class_prototypes_sign_mean(...)  : archetipi da media di classe (sign(mean))
- build_class_mapping(...)         : mappa class_label -> archetype index (0..K-1)
- make_mnist_hfl_subsets(...)      : sottoinsiemi di classi per client (come nel tuo esempio)
- gen_dataset_from_mnist_single(...) : genera ETA/labels (L, T, M_c, N) pescando immagini reali

Nota: questo modulo NON usa la generazione sintetica da ξ_true; qui i campioni
ETA sono immagini MNIST binarizzate. Le labels sono gli indici di archetipo
secondo la mappatura class->arch costruita per l'esperimento.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np


__all__ = [
    "load_mnist",
    "binarize_images",
    "class_prototypes_sign_mean",
    "build_class_mapping",
    "make_mnist_hfl_subsets",
    "gen_dataset_from_mnist_single",
]


def load_mnist(root: str, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica MNIST tramite torchvision se disponibile. Restituisce (X, y):
      X: (num, 28, 28) uint8
      y: (num,) int64
    """
    try:
        from torchvision.datasets import MNIST
        import torchvision.transforms as T
    except Exception as e:
        raise ImportError(
            "torchvision non disponibile. Installa torchvision oppure "
            "fornisci direttamente (X, y) alle altre funzioni."
        ) from e

    ds = MNIST(root=root, train=train, download=True, transform=None)
    # ds.data: (num, 28, 28) uint8; ds.targets: (num,) int64
    X = ds.data.numpy()
    y = ds.targets.numpy()
    return X, y


def binarize_images(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarizza immagini grayscale in {±1}.

    Accetta:
      - uint8 0..255 (scala automaticamente a 0..1)
      - float 0..1
      - shape (num, H, W) o (num, H, W, 1)

    Restituisce:
      - (num, N) int ∈ {±1}, flatten row-wise.
    """
    X = np.asarray(X)
    if X.ndim not in (3, 4):
        raise ValueError("X deve avere shape (num,H,W) o (num,H,W,1).")
    if X.ndim == 4:
        X = X[..., 0]
    X = X.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    H, W = X.shape[1], X.shape[2]
    Xf = X.reshape(X.shape[0], H * W)
    thr = float(threshold)
    binX = np.where(Xf >= thr, 1, -1).astype(int)
    return binX


def class_prototypes_sign_mean(
    X_bin: np.ndarray,  # (num, N) in {±1}
    y: np.ndarray,      # (num,)
    classes: Sequence[int],
) -> np.ndarray:
    """
    Costruisce prototipi di classe come sign(mean) (pseudo-archetipi).
    Ritorna ξ_prot con righe ordinate secondo 'classes' (len K, N).
    """
    X_bin = np.asarray(X_bin, dtype=int)
    y = np.asarray(y, dtype=int)
    N = X_bin.shape[1]
    protos = []
    for c in classes:
        idx = np.where(y == int(c))[0]
        if idx.size == 0:
            raise ValueError(f"Nessuna immagine per classe {c}.")
        m = X_bin[idx].mean(axis=0)
        xi = np.where(m >= 0.0, 1, -1).astype(int)
        protos.append(xi)
    return np.vstack(protos).reshape(len(classes), N)


def build_class_mapping(classes: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Mappa bidirezionale tra class label MNIST e archetype index (0..K-1).
    Restituisce: (class_to_arch, arch_to_class)
    """
    unique_sorted = [int(c) for c in sorted(set(classes))]
    class_to_arch = {c: i for i, c in enumerate(unique_sorted)}
    arch_to_class = {i: c for c, i in class_to_arch.items()}
    return class_to_arch, arch_to_class


def make_mnist_hfl_subsets(
    L: int,
    client_classes: Sequence[Sequence[int]],
) -> List[List[int]]:
    """
    Sottoinsiemi di CLASSI per ciascun client, già specificati (es. [[1,2,3],[4,5,6],[7,8,9]]).
    Ritorna una lista di liste (lunghezza L), con classi ordinate e univoche per client.

    Nota: QUI si parla di classi originali MNIST, non di indici archetipo.
    """
    if len(client_classes) != L:
        raise ValueError("client_classes deve avere lunghezza L.")
    subsets: List[List[int]] = []
    for l in range(L):
        uniq_sorted = sorted(set(int(c) for c in client_classes[l]))
        if len(uniq_sorted) == 0:
            raise ValueError(f"Client {l} senza classi assegnate.")
        subsets.append(uniq_sorted)
    return subsets


def _sample_indices_by_class(
    y: np.ndarray,
    allowed_classes: Sequence[int],
    num: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Estrae 'num' indici di immagini dalle classi consentite (equiprobabile sulle classi).
    """
    allowed = list(allowed_classes)
    if len(allowed) == 0:
        raise ValueError("allowed_classes vuoto.")
    # seleziona la classe uniformemente, poi un indice casuale da quella classe
    per_class_idx = {c: np.where(y == c)[0] for c in allowed}
    for c, idx in per_class_idx.items():
        if idx.size == 0:
            raise ValueError(f"Nessuna immagine per classe {c}.")
    cls_choices = rng.integers(0, len(allowed), size=num)
    out = np.empty((num,), dtype=int)
    for i, r in enumerate(cls_choices):
        c = allowed[int(r)]
        pool = per_class_idx[c]
        j = int(rng.integers(0, pool.size))
        out[i] = pool[j]
    return out


def gen_dataset_from_mnist_single(
    X: np.ndarray,                  # immagini raw (num, H, W) o (num,H,W,1)
    y: np.ndarray,                  # labels raw (num,)
    client_classes: Sequence[Sequence[int]],  # classi per client (MNIST labels)
    n_batch: int,
    L: int,
    M_total: int,
    class_to_arch: Dict[int, int],  # mappa: classe MNIST -> indice archetipo [0..K-1]
    rng: Optional[np.random.Generator] = None,
    binarize_threshold: float = 0.5,
    use_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera dataset federato SINGLE usando immagini reali MNIST.

    Restituisce:
      ETA   : (L, n_batch, M_c, N)  campioni binarizzati in {±1}
      labels: (L, n_batch, M_c)     indici archetipo (secondo class_to_arch)
    """
    rng = np.random.default_rng() if rng is None else rng
    X_bin = binarize_images(X, threshold=binarize_threshold)  # (num, N)
    num, N = X_bin.shape

    if len(client_classes) != L:
        raise ValueError("client_classes deve avere lunghezza L.")
    if n_batch <= 0:
        raise ValueError("n_batch deve essere > 0.")
    M_c = int(np.ceil(M_total / float(L * n_batch)))

    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.int32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)

    itL = range(L)
    if use_tqdm:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            itL = _tqdm(itL, desc="MNIST clients", leave=False)
        except Exception:
            pass

    for l in itL:
        allowed = list(client_classes[l])
        if len(allowed) == 0:
            raise ValueError(f"Client {l} senza classi consentite.")
        itT = range(n_batch)
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                itT = _tqdm(itT, desc=f"client {l} rounds", leave=False)
            except Exception:
                pass
        for t in itT:
            idxs = _sample_indices_by_class(y, allowed_classes=allowed, num=M_c, rng=rng)
            ETA[l, t] = X_bin[idxs]
            # mappa le classi MNIST verso indici archetipo
            labs = [class_to_arch[int(y[i])] for i in idxs]
            labels[l, t] = np.asarray(labs, dtype=np.int32)

    return ETA.astype(np.float32), labels.astype(np.int32)
