"""
Practical helpers to estimate deformed Marchenko-Pastur edges.

The functions here keep the implementation lightweight (NumPy-only) while still
providing the knobs required by the panel scripts:

- `marchenko_pastur_edge`: closed-form classical MP right edge.
- `approximate_deformed_edge`: Monte-Carlo estimator of the right edge for a
   separable covariance model defined by a noise scatter `Sigma`.

Even though the “true” deformed edge can be obtained by solving Silverstein’s
fixed-point equations, in practice a short Monte-Carlo sweep on the known
noise covariance is enough for the stress tests we run in the repo.  The helper
below keeps this behaviour configurable so it can be swapped with a full QuEST
solver in the future without touching the panel code.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "marchenko_pastur_edge",
    "approximate_deformed_edge",
]


def marchenko_pastur_edge(q: float, variance: float = 1.0) -> float:
    """
    Classical MP right edge for variance `variance` and aspect ratio q = N/n.
    """
    if q <= 0:
        raise ValueError("Aspect ratio q must be positive.")
    if variance <= 0:
        raise ValueError("Variance must be positive.")
    return float(variance) * (1.0 + np.sqrt(float(q))) ** 2


def approximate_deformed_edge(
    sigma: np.ndarray,
    q: float,
    *,
    n_probe: int | None = None,
    n_mc: int = 16,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte-Carlo approximation of the deformed MP right edge for covariance `sigma`.

    Parameters
    ----------
    sigma : (N, N)
        Positive-definite noise covariance (without spikes).
    q : float
        Aspect ratio N / n used in the experiment.
    n_probe : int or None
        Number of samples used to probe the edge.  Defaults to ceil(N / q).
    n_mc : int
        Number of independent Monte-Carlo replicas to stabilize the estimate.
    rng : np.random.Generator or None
        RNG for reproducibility.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be square.")
    N = sigma.shape[0]
    if q <= 0:
        raise ValueError("q must be positive.")
    if n_probe is None:
        n_probe = int(np.ceil(N / float(q)))
    if n_probe <= 0:
        n_probe = N
    rng = rng or np.random.default_rng()

    evals, vecs = np.linalg.eigh(sigma)
    evals = np.clip(evals, 1e-9, None)
    sigma_sqrt = (vecs * np.sqrt(evals)) @ vecs.T

    max_eigs: list[float] = []
    for _ in range(int(n_mc)):
        Z = rng.standard_normal((N, n_probe))
        X = sigma_sqrt @ Z
        S = (X @ X.T) / float(n_probe)
        vals = np.linalg.eigvalsh(S)
        max_eigs.append(float(vals[-1]))
    return float(np.median(max_eigs))
