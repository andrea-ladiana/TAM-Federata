"""
Spectral utilities exposed for panels and experiments.
"""

from .robust import (
    TylerShapeResult,
    tyler_shape_matrix,
    normalize_trace,
    normalize_diagonal,
)
from .deformed_mp import (
    marchenko_pastur_edge,
    approximate_deformed_edge,
)

__all__ = [
    "TylerShapeResult",
    "tyler_shape_matrix",
    "normalize_trace",
    "normalize_diagonal",
    "marchenko_pastur_edge",
    "approximate_deformed_edge",
]
