"""Dense-grid piecewise-affine homeomorphism layers."""

from .monotone import DenseMonotoneGridLayer, MultiscaleMonotoneGridLayer
from .patches import LocalPatchComposition, LocalPatchMonotoneLayer
from .edge_woodbury import SparseEdgeWoodburyTutteLayer
from .exact_block_schur import ExactBlockSchurTutteLayer
from .alternating import ExactAlternatingMonotoneComposition

__all__ = [
    "DenseMonotoneGridLayer",
    "MultiscaleMonotoneGridLayer",
    "LocalPatchMonotoneLayer",
    "LocalPatchComposition",
    "SparseEdgeWoodburyTutteLayer",
    "ExactBlockSchurTutteLayer",
    "ExactAlternatingMonotoneComposition",
]
