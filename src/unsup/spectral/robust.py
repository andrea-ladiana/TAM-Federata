"""
Robust scatter estimators and normalizations used across spectral panels.

This module currently exposes a trace-normalized Tyler shape estimator together
with a few helper utilities to keep the implementation numerically stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "TylerShapeResult",
    "tyler_shape_matrix",
    "normalize_trace",
    "normalize_diagonal",
]


@dataclass(frozen=True)
class TylerShapeResult:
    """Container returning both the matrix and diagnostics."""

    scatter: np.ndarray
    iters: int
    converged: bool
    rel_change: float


def normalize_trace(M: np.ndarray, target: float | None = None) -> np.ndarray:
    """
    Rescale a symmetric matrix so that its trace matches `target` (default: dim).
    """
    M = 0.5 * (np.asarray(M, dtype=np.float64) + np.asarray(M, dtype=np.float64).T)
    tr = np.trace(M)
    if tr <= 0:
        raise ValueError("Matrix trace must be positive to normalize.")
    scale = (float(target) if target is not None else M.shape[0]) / tr
    return M * scale


def normalize_diagonal(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Whiten a covariance-like matrix so that its diagonal entries become 1.
    """
    M = 0.5 * (np.asarray(M, dtype=np.float64) + np.asarray(M, dtype=np.float64).T)
    diag = np.clip(np.diag(M), eps, None)
    inv_sqrt = 1.0 / np.sqrt(diag)
    return (inv_sqrt[:, None] * M) * inv_sqrt[None, :]


def _stable_inverse(M: np.ndarray, jitter: float) -> Tuple[np.ndarray, float]:
    """
    Compute the inverse of `M`, retrying with an additive jitter if needed.
    """
    eps = float(jitter)
    eye = np.eye(M.shape[0], dtype=np.float64)
    attempt = 0
    while True:
        try:
            return np.linalg.inv(M + eps * eye), eps
        except np.linalg.LinAlgError:
            eps *= 10.0
            attempt += 1
            if attempt > 6:
                raise


def tyler_shape_matrix(
    X: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    jitter: float = 1e-6,
) -> TylerShapeResult:
    """
    Trace-normalized Tyler shape estimator.

    Parameters
    ----------
    X : (N, n)
        Data matrix with samples stored column-wise.
    tol : float
        Relative Frobenius tolerance for convergence.
    max_iter : int
        Soft limit on iterations (loop exits early if converged).
    jitter : float
        Initial ridge added to the iterate when inversion becomes unstable.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D with shape (N, n).")
    N, n = X.shape
    if n <= N:
        # Tyler still works but convergence slows down; warn via docstring.
        pass

    S = normalize_trace(np.eye(N, dtype=np.float64), target=N)
    rel_change = np.inf
    converged = False

    for it in range(1, max_iter + 1):
        S_inv, used_jitter = _stable_inverse(S, jitter)
        quad = np.sum(X * (S_inv @ X), axis=0)  # diag of X^T S^{-1} X
        quad = np.clip(quad, 1e-12, None)
        weights = 1.0 / quad
        weighted = X * weights
        S_new = (N / float(n)) * (weighted @ X.T)
        S_new = normalize_trace(S_new, target=N)
        rel_change = np.linalg.norm(S_new - S, ord="fro") / np.linalg.norm(S, ord="fro")
        S = S_new
        if rel_change < tol:
            converged = True
            break

    return TylerShapeResult(scatter=normalize_trace(S, target=N), iters=it, converged=converged, rel_change=rel_change)

