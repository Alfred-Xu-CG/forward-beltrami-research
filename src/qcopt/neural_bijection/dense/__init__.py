"""Dense-grid piecewise-affine homeomorphism layers."""

from .monotone import DenseMonotoneGridLayer, MultiscaleMonotoneGridLayer
from .patches import LocalPatchComposition, LocalPatchMonotoneLayer
from .edge_woodbury import SparseEdgeWoodburyTutteLayer
from .exact_block_schur import ExactBlockSchurTutteLayer
from .reusable_interface_schur import ReusableInterfaceSchurTutteLayer
from .alternating import ExactAlternatingMonotoneComposition, evaluate_structured_p1_with_jacobian
from .convex_quad import HierarchicalConvexQuadLayer, HierarchicalConvexQuadFreeCenterLayer, certify_convex_quad_output, split_convex_quad_grid
from .convex_quad_composition import CoarseFineConvexQuadComposition, CoarseFineConvexQuadResult

__all__ = [
    "DenseMonotoneGridLayer",
    "MultiscaleMonotoneGridLayer",
    "LocalPatchMonotoneLayer",
    "LocalPatchComposition",
    "SparseEdgeWoodburyTutteLayer",
    "ExactBlockSchurTutteLayer",
    "ReusableInterfaceSchurTutteLayer",
    "ExactAlternatingMonotoneComposition",
    "evaluate_structured_p1_with_jacobian",
    "HierarchicalConvexQuadLayer",
    "HierarchicalConvexQuadFreeCenterLayer",
    "certify_convex_quad_output",
    "split_convex_quad_grid",
    "CoarseFineConvexQuadComposition",
    "CoarseFineConvexQuadResult",
]
