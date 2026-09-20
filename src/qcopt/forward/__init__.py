"""Shared primitives for forward Beltrami route experiments."""

from .benchmarks import affine_map, manufactured_mu, smooth_twist_map
from .beurling import (
    PeriodicFrequencies,
    PeriodicFixedPointResult,
    periodic_beltrami_fixed_point,
    periodic_beurling_apply,
    periodic_cauchy_inverse,
    periodic_dbar_apply,
    periodic_dz_apply,
    periodic_frequency_grid,
    periodic_lift_derivatives,
    periodic_beltrami_map_from_h,
    torus_affine_derivatives,
    torus_affine_periods,
    torus_affine_grid,
    torus_periods_to_affine_coefficients,
    zero_padded_beurling_apply,
    zero_padded_cauchy_inverse,
    zero_padded_dbar_apply,
    zero_padded_dz_apply,
)
from .operators import (
    ForwardMetrics,
    beltrami_residual,
    evaluate_forward_metrics,
    face_jacobian_determinants,
    periodic_grid_edges,
)
from .discrete_symbols import central_difference_beltrami_symbol
from .p1_symbols import p1_beurling_block_symbol, p1_block_symbol_grid, p1_face_derivative_symbols
from .p1_symbols_torch import p1_block_symbol_grid_torch
from .torus_tutte import (
    periodic_positive_graph_embedding,
    periodic_torus_face_determinants,
    periodic_tutte_embedding,
)
from .tutte import tutte_embedding, tutte_embedding_weighted, tutte_weights
from .mmatrix import (
    DirectionalConductances,
    DictionaryConductances,
    SpectralConductances,
    beltrami_conductivity,
    batch_positive_directional_conductances,
    directional_conductances,
    dictionary_conductances,
    conditioning_safe_dictionary_conductances,
    integer_wide_stencil_conductances,
    integer_wide_stencil_directions,
    edge_direction_conductances,
    spectral_conductances,
)
from .anisotropic_hodge import DiamondStencilResult, diamond_stencil_decomposition
from .safe_step import (
    certified_residual_step,
    certified_step_toward_target,
    maximum_safe_step,
    maximum_safe_step_directional_derivative,
)
from .compatibility import compatibility_matrix, compatibility_residual
from .compatibility_projection import BeltramiProjectionResult, project_facewise_mu, projection_mu_jacobian
from .flow import SafePLFlowResult, safe_pl_flow
from .tutte_torch import tutte_embedding_torch
from .prolongation import prolongate_positive_increments, prolongate_regular_grid
from .tutte_implicit import TutteImplicitSystem, tutte_embedding_torch_implicit
from .tutte_directed_implicit import (
    DirectedTutteSystem,
    directed_tutte_embedding_torch_implicit,
    rectangle_boundary_from_logits,
)
from .tutte_weighted_implicit import weighted_tutte_vjp
from .sphere import mobius_scale_sphere, mobius_scale_velocity_sphere, sphere_face_orientation, uv_sphere_mesh
from .orbifold import polar_disk_mesh, radial_cone_inverse, radial_cone_map
from .sphere_charts import (
    stereographic_forward,
    stereographic_inverse,
    stereographic_transition,
    stereographic_transition_velocity,
)
from .beurling_gmres import (
    PeriodicAndersonResult,
    PeriodicGMRESResult,
    periodic_beltrami_anderson,
    periodic_beltrami_gmres,
    periodic_neumann_initial_guess,
    periodic_polynomial_initial_guess,
)
from .beurling_zero_padded import (
    ZeroPaddedFixedPointResult,
    zero_padded_beltrami_fixed_point,
    zero_padded_lift_derivatives,
)
from .beurling_direct import direct_beurling_apply
from .beurling_torch import direct_beurling_apply_torch
from .beurling_implicit import periodic_beltrami_implicit_vjp, periodic_beltrami_solve_with_vjp, periodic_beltrami_map_h_vjp, torus_affine_map_vjp, torus_affine_period_vjp
from .shear import affine_shear_log_abs_det, affine_shear_matrix, apply_affine_shear, invert_affine_shear
from .coupled_decoder import (
    coupled_monotone_shear_inverse,
    coupled_monotone_shear_map,
    triangular_spatial_monotone_inverse,
    triangular_spatial_monotone_map,
)
from .monotone import monotone_axis_inverse, monotone_axis_map, separable_monotone_map
from .monotone_torch import positive_increment_knots
from .rectangle_fd import RectangleFDFactorization, RectangleFDSolveResult, rectangle_beltrami_fd
from .rectangle_fd_iterative import RectangleFDIterativeResult, solve_rectangle_fd_gmres
from .rectangle_conductivity import RectangleConductivityResult, rectangle_beltrami_conductivity
from .rectangle_fd_implicit import rectangle_fd_boundary_vjp, rectangle_fd_coefficient_vjp
from .bhf_variation import (
    arbitrary_base_bhf_variation,
    bhf_kernel_R,
    duffy_triangle_kernel_integral,
    vertex_near_far_bhf_variation,
    interior_point_near_far_bhf_variation,
    edge_point_near_far_bhf_variation,
    all_vertex_near_far_bhf_variation,
    normalized_bhf_variation,
    triangle_bhf_variation,
)
from .bhf_torch import (
    bhf_near_far_apply_torch,
    bhf_near_far_apply_torch_differentiable_real_pair,
)
from .bhf_flow_recompute import recompute_bhf_flow_real_pair
from .beurling_particle_mesh import particle_mesh_beurling_apply
from .beurling_treecode import treecode_beurling_apply, treecode_beurling_values_vjp
from .beurling_reflection import reflection_extend_mu

__all__ = [
    "affine_map",
    "manufactured_mu",
    "smooth_twist_map",
    "PeriodicFrequencies",
    "PeriodicFixedPointResult",
    "periodic_beltrami_fixed_point",
    "periodic_beurling_apply",
    "periodic_cauchy_inverse",
    "periodic_dbar_apply",
    "periodic_dz_apply",
    "periodic_frequency_grid",
    "periodic_lift_derivatives",
    "periodic_beltrami_map_from_h",
    "torus_affine_derivatives",
    "torus_affine_periods",
    "torus_affine_grid",
    "torus_periods_to_affine_coefficients",
    "zero_padded_beurling_apply",
    "zero_padded_cauchy_inverse",
    "zero_padded_dbar_apply",
    "zero_padded_dz_apply",
    "tutte_embedding",
    "tutte_embedding_weighted",
    "tutte_weights",
    "DirectionalConductances",
    "DictionaryConductances",
    "SpectralConductances",
    "beltrami_conductivity",
    "batch_positive_directional_conductances",
    "directional_conductances",
    "dictionary_conductances",
    "conditioning_safe_dictionary_conductances",
    "integer_wide_stencil_conductances",
    "integer_wide_stencil_directions",
    "edge_direction_conductances",
    "spectral_conductances",
    "DiamondStencilResult",
    "diamond_stencil_decomposition",
    "certified_residual_step",
    "maximum_safe_step",
    "certified_step_toward_target",
    "maximum_safe_step_directional_derivative",
    "compatibility_matrix",
    "compatibility_residual",
    "BeltramiProjectionResult",
    "project_facewise_mu",
    "projection_mu_jacobian",
    "SafePLFlowResult",
    "safe_pl_flow",
    "tutte_embedding_torch",
    "prolongate_regular_grid",
    "prolongate_positive_increments",
    "TutteImplicitSystem",
    "tutte_embedding_torch_implicit",
    "DirectedTutteSystem",
    "directed_tutte_embedding_torch_implicit",
    "rectangle_boundary_from_logits",
    "weighted_tutte_vjp",
    "mobius_scale_sphere",
    "mobius_scale_velocity_sphere",
    "sphere_face_orientation",
    "uv_sphere_mesh",
    "polar_disk_mesh",
    "radial_cone_map",
    "radial_cone_inverse",
    "stereographic_forward",
    "stereographic_inverse",
    "stereographic_transition",
    "stereographic_transition_velocity",
    "PeriodicGMRESResult",
    "PeriodicAndersonResult",
    "periodic_beltrami_anderson",
    "periodic_beltrami_gmres",
    "periodic_neumann_initial_guess",
    "periodic_polynomial_initial_guess",
    "ZeroPaddedFixedPointResult",
    "zero_padded_beltrami_fixed_point",
    "zero_padded_lift_derivatives",
    "direct_beurling_apply",
    "direct_beurling_apply_torch",
    "periodic_beltrami_implicit_vjp",
    "periodic_beltrami_solve_with_vjp",
    "periodic_beltrami_map_h_vjp",
    "torus_affine_period_vjp",
    "torus_affine_map_vjp",
    "affine_shear_matrix",
    "apply_affine_shear",
    "invert_affine_shear",
    "affine_shear_log_abs_det",
    "monotone_axis_map",
    "monotone_axis_inverse",
    "separable_monotone_map",
    "positive_increment_knots",
    "coupled_monotone_shear_map",
    "coupled_monotone_shear_inverse",
    "triangular_spatial_monotone_map",
    "triangular_spatial_monotone_inverse",
    "RectangleFDSolveResult",
    "RectangleFDFactorization",
    "rectangle_beltrami_fd",
    "RectangleFDIterativeResult",
    "solve_rectangle_fd_gmres",
    "RectangleConductivityResult",
    "rectangle_beltrami_conductivity",
    "rectangle_fd_boundary_vjp",
    "rectangle_fd_coefficient_vjp",
    "normalized_bhf_variation",
    "arbitrary_base_bhf_variation",
    "bhf_kernel_R",
    "duffy_triangle_kernel_integral",
    "vertex_near_far_bhf_variation",
    "interior_point_near_far_bhf_variation",
    "edge_point_near_far_bhf_variation",
    "all_vertex_near_far_bhf_variation",
    "triangle_bhf_variation",
    "bhf_near_far_apply_torch",
    "bhf_near_far_apply_torch_differentiable_real_pair",
    "recompute_bhf_flow_real_pair",
    "particle_mesh_beurling_apply",
    "treecode_beurling_apply",
    "treecode_beurling_values_vjp",
    "reflection_extend_mu",
    "beltrami_residual",
    "ForwardMetrics",
    "evaluate_forward_metrics",
    "face_jacobian_determinants",
    "periodic_grid_edges",
    "central_difference_beltrami_symbol",
    "p1_face_derivative_symbols",
    "p1_block_symbol_grid_torch",
    "p1_beurling_block_symbol",
    "p1_block_symbol_grid",
    "periodic_tutte_embedding",
    "periodic_positive_graph_embedding",
    "periodic_torus_face_determinants",
]
