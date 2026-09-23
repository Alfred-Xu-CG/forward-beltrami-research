"""Dense-grid piecewise-affine homeomorphism layers."""

from .monotone import DenseMonotoneGridLayer, MultiscaleMonotoneGridLayer
from .patches import LocalPatchComposition, LocalPatchMonotoneLayer
from .edge_woodbury import SparseEdgeWoodburyTutteLayer
from .exact_block_schur import ExactBlockSchurTutteLayer
from .pardiso_all_edge import PardisoAllEdgeTutteLayer
from .reusable_interface_schur import ReusableInterfaceSchurTutteLayer
from .alternating import ExactAlternatingMonotoneComposition, evaluate_structured_p1_with_jacobian
from .convex_quad import HierarchicalConvexQuadLayer, HierarchicalConvexQuadFreeCenterLayer, certify_convex_quad_output, split_convex_quad_grid
from .convex_quad_composition import CoarseFineConvexQuadComposition, CoarseFineConvexQuadResult
from .colored_vertex_relaxation import SafeColoredVertexRelaxation, HierarchicalConvexQuadLocalLayer
from .forward_p1_pyramid import ForwardP1Pyramid, exact_dyadic_p1_refine
from .multilevel_forward_p1 import CoarsePatchFineVertexP1Layer, CertifiedForwardP1Pyramid, certify_p1_or_identity
from .forward_p1_encoder import ForwardP1ImageEncoder, PatchPyramidImageEncoder
from .patch_field import SafePatchFieldPass, StaggeredPatchP1Layer, TiedStaggeredPatchP1Layer, ResidualStaggeredPatchP1Layer, ForwardPatchP1Pyramid, ResidualPatchP1Pyramid
from .spectral_p1_refinement import SineModeP1Refiner
from .monotone_fiber_p1 import MonotoneFiberP1Layer, MultiscaleMonotoneFiberP1Layer, SoftplusPotentialFiberP1Layer, SoftplusPotentialFiberRefiner, HybridMonotoneFiberP1Layer
from .photometric_hint import local_photometric_logits, physical_image_gradient
from .sine_spectral import dst1, dst2, idst2, keep_vector_sine_modes, spectralize_bounded_logits
from .face_qc import structured_p1_face_beltrami_modulus, structured_p1_qc_tail_penalty
from .sine_pcg_tutte import SinePreconditionedTutteLayer
from .conductance_synthesis import synthesize_bounded_conductances
from .spectral_feedback import SpectralSafeFeedbackLayer
from .qc_radial_relaxation import SafeColoredQCRadialRelaxation
from .photometric_conductance import PhotometricSpectralTutteLayer
from .nested_p1_feedback import NestedP1PhotometricFeedbackLayer

__all__ = [
    "MonotoneFiberP1Layer",
    "MultiscaleMonotoneFiberP1Layer",
    "SoftplusPotentialFiberP1Layer",
    "SoftplusPotentialFiberRefiner",
    "HybridMonotoneFiberP1Layer",
    "SineModeP1Refiner",
    "DenseMonotoneGridLayer",
    "MultiscaleMonotoneGridLayer",
    "LocalPatchMonotoneLayer",
    "LocalPatchComposition",
    "SparseEdgeWoodburyTutteLayer",
    "ExactBlockSchurTutteLayer",
    "PardisoAllEdgeTutteLayer",
    "ReusableInterfaceSchurTutteLayer",
    "ExactAlternatingMonotoneComposition",
    "evaluate_structured_p1_with_jacobian",
    "HierarchicalConvexQuadLayer",
    "HierarchicalConvexQuadFreeCenterLayer",
    "certify_convex_quad_output",
    "split_convex_quad_grid",
    "CoarseFineConvexQuadComposition",
    "CoarseFineConvexQuadResult",
    "SafeColoredVertexRelaxation",
    "HierarchicalConvexQuadLocalLayer",
    "CoarsePatchFineVertexP1Layer",
    "CertifiedForwardP1Pyramid",
    "ResidualPatchP1Pyramid",
    "certify_p1_or_identity",
    "local_photometric_logits",
    "physical_image_gradient",
    "dst1",
    "dst2",
    "idst2",
    "keep_vector_sine_modes",
    "spectralize_bounded_logits",
    "structured_p1_face_beltrami_modulus",
    "structured_p1_qc_tail_penalty",
    "SinePreconditionedTutteLayer",
    "synthesize_bounded_conductances",
    "SpectralSafeFeedbackLayer",
    "SafeColoredQCRadialRelaxation",
    "PhotometricSpectralTutteLayer",
    "NestedP1PhotometricFeedbackLayer",
]
