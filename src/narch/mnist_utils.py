
# -*- coding: utf-8 -*-
"""
mnist_utils.py — Utilities to use MNIST (structured data) in Exp-07 single-mode.

Design goals:
  • Reuse existing dataset/binarisation/prototype builders from your codebase when available.
  • Provide light, dependency-free fallbacks operating on numpy arrays (X,y) already loaded.
  • Build ±1 binarised images and class prototypes ξ_true = sign(mean image per class).
  • Offer a helper to assemble per-round, per-client batches in SINGLE mode given π_t.

This module deliberately avoids downloading datasets. If you need to fetch MNIST,
use your project-specific loaders and pass (X,y) to the fallbacks here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Prefer reusing your project loaders / transformers, if present.
# These imports are *optional* and only used if available.
try:
    from .data import binarize_pm1 as _cb_binarize_pm1  # type: ignore
except Exception:
    _cb_binarize_pm1 = None

try:
    from .data import build_class_prototypes as _cb_build_class_prototypes  # type: ignore
except Exception:
    _cb_build_class_prototypes = None

# -----------------------------------------------------------------------------
# Basic transforms
# -----------------------------------------------------------------------------
def binarize_pm1(X: np.ndarray, *, threshold: Optional[float] = None) -> np.ndarray:
    """
    Binarise grayscale images to ±1. If threshold is None, use the per-pixel median.
    X: array of shape (M, H, W) or (M, N).
    Returns: Xb in shape (M, N) with values in {-1, +1}.
    """
    Xf = X.astype(np.float32)
    if Xf.ndim == 3:  # (M,H,W) → (M,N)
        M, H, W = Xf.shape
        Xf = Xf.reshape(M, H * W)
    if threshold is None:
        thr = np.median(Xf, axis=0, keepdims=True)
    else:
        thr = float(threshold)
    Xb = np.where(Xf >= thr, 1.0, -1.0).astype(np.float32)
    return Xb

def build_class_prototypes(X_pm1: np.ndarray, y: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """
    Build ξ_true as sign(mean image) per class in 'classes', from binarised data X_pm1 ∈ {±1}.
    Returns an array of shape (K, N) in {±1}, ordered as 'classes'.
    """
    if _cb_build_class_prototypes is not None:
        return _cb_build_class_prototypes(X_pm1, y, classes)  # type: ignore

    cls = list(classes)
    N = X_pm1.shape[1]
    K = len(cls)
    xi = np.zeros((K, N), dtype=np.float32)
    for i, c in enumerate(cls):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            raise ValueError(f"No samples for class {c}")
        mu = X_pm1[idx].mean(axis=0, keepdims=True)
        xi[i] = np.where(mu >= 0.0, 1.0, -1.0)
    return xi

def select_classes(X: np.ndarray, y: np.ndarray, classes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    cls = np.array(classes, dtype=int).ravel().tolist()
    mask = np.isin(y, cls)
    return X[mask], y[mask]

# -----------------------------------------------------------------------------
# Round-wise SINGLE batching
# -----------------------------------------------------------------------------
@dataclass
class SingleBatchSpec:
    L: int                  # number of clients
    M_c: int                # examples per client per round
    classes: Sequence[int]  # ordered list of K classes, e.g., (0,1,2)
    seed: int = 0

def sample_round_single(
    X_pm1: np.ndarray,
    y: np.ndarray,
    pi_t: np.ndarray,
    spec: SingleBatchSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a SINGLE-mode round:
      • X_pm1: (M, N) binarised images in {±1}
      • y    : (M,) integer labels
      • pi_t : (K,) mixing vector for the *global* round composition
      • spec : SingleBatchSpec(L, M_c, classes)

    Returns:
      E_t: (L, M_c, N) examples (clients first, then local batch)
      y_t: (L, M_c) class indices in {0, ..., K-1} matching 'classes' order
    """
    rng = np.random.default_rng(spec.seed)
    classes = list(spec.classes)
    K = len(classes)
    if pi_t.shape[0] != K:
        raise ValueError(f"pi_t has shape {pi_t.shape}, expected length {K}")

    # Pre-split indices per chosen class
    idx_per = [np.where(y == c)[0] for c in classes]
    for arr in idx_per:
        if arr.size == 0:
            raise ValueError("One of the selected classes has no samples.")

    # Global pool per class → draw totals according to pi_t then split to clients evenly
    total = spec.L * spec.M_c
    counts = np.random.multinomial(total, (pi_t / (pi_t.sum() + 1e-9)).astype(float))

    # Prepare per-client allocation (round-robin across classes)
    y_out = np.empty((spec.L, spec.M_c), dtype=int)
    E_out = np.empty((spec.L, spec.M_c, X_pm1.shape[1]), dtype=np.float32)

    # For each class, draw 'counts[k]' unique indices (with replacement=False if enough, else True)
    drawn = []
    for k, arr in enumerate(idx_per):
        if counts[k] <= arr.size:
            choose = rng.choice(arr, size=counts[k], replace=False)
        else:
            choose = rng.choice(arr, size=counts[k], replace=True)
        drawn.append(choose)

    # Interleave samples into clients batches
    ptr = [0] * K
    for i in range(spec.L):
        for j in range(spec.M_c):
            # choose class by proportional allocation left
            remaining = np.array([counts[k] - ptr[k] for k in range(K)], dtype=int)
            if (remaining <= 0).all():
                # fallback: uniform
                k = int(rng.integers(0, K))
            else:
                probs = remaining / (remaining.sum() + 1e-9)
                k = int(rng.choice(np.arange(K), p=probs))
            idx = drawn[k][ptr[k]]
            ptr[k] += 1
            E_out[i, j] = X_pm1[idx]
            y_out[i, j] = k

    return E_out, y_out

# -----------------------------------------------------------------------------
# High-level: build ξ_true and per-round batches from raw arrays
# -----------------------------------------------------------------------------
def prepare_mnist_triplet_single(
    X: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    pis_over_time: np.ndarray,
    L: int,
    M_c: int,
    seed: int = 0,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Convenience wrapper:
      • Select three classes (K=3 recommended for Δ₂ figures)
      • Binarise to ±1
      • Build ξ_true via sign(mean)
      • For each t, draw SINGLE-mode batch (E_t, y_t) according to π_t

    Returns:
      xi_true: (K, N)
      rounds : list of length T with elements (E_t, y_t)
    """
    Xc, yc = select_classes(X, y, classes)
    X_pm1 = binarize_pm1(Xc, threshold=threshold)
    xi_true = build_class_prototypes(X_pm1, yc, classes)

    T = pis_over_time.shape[0]
    spec = SingleBatchSpec(L=L, M_c=M_c, classes=classes, seed=seed)
    rounds: List[Tuple[np.ndarray, np.ndarray]] = []
    for t in range(T):
        E_t, y_t = sample_round_single(X_pm1, yc, pis_over_time[t], spec)
        rounds.append((E_t, y_t))
        spec.seed += 1  # different sampling per round
    return xi_true, rounds
