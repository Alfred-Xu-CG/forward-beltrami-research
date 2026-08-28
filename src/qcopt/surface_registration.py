"""Landmark and curvature refinement by native multi-chart surface flows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_atlas import SurfacePoint
from .surface_flow import (
    FlowHistory,
    FlowStep,
    IntrinsicFaceField,
    advect_points,
    required_substeps_for_cfl,
)
from .surface_geometry import (
    ScreenedFieldSmoother,
    curvature_descriptors,
    interpolate_vertex_field,
    scatter_point_forces,
    surface_scalar_gradient,
)
from .surface_mesh import SurfaceMesh


@dataclass(frozen=True)
class RegistrationConfig:
    iterations: int = 30
    landmark_weight: float = 4.0
    curvature_weight: float = 1.0
    smoothing: float = 1.0
    descriptor_smoothing: float = 0.25
    max_update_fraction: float = 0.1
    backtracking_steps: int = 8
    minimum_relative_decrease: float = 1e-10
    integration_substeps: int = 8
    cfl_limit: float = 0.15

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.backtracking_steps < 1:
            raise ValueError("iteration counts must be nonnegative")
        if self.landmark_weight < 0.0 or self.curvature_weight < 0.0:
            raise ValueError("objective weights must be nonnegative")
        if self.smoothing < 0.0 or self.descriptor_smoothing < 0.0:
            raise ValueError("smoothing weights must be nonnegative")
        if self.max_update_fraction <= 0.0:
            raise ValueError("max_update_fraction must be positive")
        if self.integration_substeps < 1:
            raise ValueError("integration_substeps must be positive")
        if self.cfl_limit <= 0.0 or not np.isfinite(self.cfl_limit):
            raise ValueError("cfl_limit must be positive and finite")


@dataclass(frozen=True)
class RegistrationIteration:
    iteration: int
    objective: float
    landmark_rmse: float
    curvature_rmse: float
    accepted_scale: float
    chart_transitions: int


@dataclass(frozen=True)
class RegistrationResult:
    points: tuple[SurfacePoint, ...]
    flow_history: FlowHistory
    iterations: tuple[RegistrationIteration, ...]
    initial_landmark_rmse: float
    final_landmark_rmse: float
    initial_curvature_rmse: float
    final_curvature_rmse: float
    total_chart_transitions: int
    convergence_reason: str


@dataclass(frozen=True)
class _ObjectiveState:
    total: float
    landmark_rmse: float
    curvature_rmse: float
    forces: np.ndarray | None


def register_surface_map(
    source_mesh: SurfaceMesh,
    target_mesh: SurfaceMesh,
    source_samples: tuple[SurfacePoint, ...] | list[SurfacePoint],
    initial_target_points: tuple[SurfacePoint, ...] | list[SurfacePoint],
    *,
    landmark_indices: np.ndarray | list[int] | tuple[int, ...] = (),
    landmark_targets: tuple[SurfacePoint, ...] | list[SurfacePoint] = (),
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """Refine a topology-valid map without a global flattening.

    ``source_samples`` and ``initial_target_points`` are paired evaluations of
    the initial homeomorphism.  Chart ids of target points are updated only by
    face walking; they are not optimization variables.
    """

    config = config or RegistrationConfig()
    if not source_mesh.topology.closed or not target_mesh.topology.closed:
        raise ValueError("registration requires closed surfaces")
    if source_mesh.topology.genus is None or source_mesh.topology.genus != target_mesh.topology.genus:
        raise ValueError("source and target must have the same orientable genus")
    source_samples = tuple(source_samples)
    current = tuple(initial_target_points)
    if len(source_samples) == 0 or len(source_samples) != len(current):
        raise ValueError("source and target sample counts must agree and be nonzero")
    landmark_indices = np.asarray(landmark_indices, dtype=np.int64)
    landmark_targets = tuple(landmark_targets)
    if len(landmark_indices) != len(landmark_targets):
        raise ValueError("landmark index and target counts must agree")
    if len(landmark_indices) and (
        landmark_indices.min() < 0 or landmark_indices.max() >= len(source_samples)
    ):
        raise ValueError("landmark sample index out of range")

    source_descriptor = curvature_descriptors(source_mesh)
    target_descriptor = curvature_descriptors(target_mesh)
    source_smoother = ScreenedFieldSmoother(source_mesh, config.descriptor_smoothing)
    target_descriptor_smoother = ScreenedFieldSmoother(
        target_mesh, config.descriptor_smoothing
    )
    source_fields = np.column_stack(
        (
            source_smoother.smooth(source_descriptor.gaussian),
            source_smoother.smooth(source_descriptor.mean),
        )
    )
    target_fields = np.column_stack(
        (
            target_descriptor_smoother.smooth(target_descriptor.gaussian),
            target_descriptor_smoother.smooth(target_descriptor.mean),
        )
    )
    source_values = interpolate_vertex_field(
        source_mesh, source_fields, list(source_samples)
    )
    target_gradients = np.stack(
        (
            surface_scalar_gradient(target_mesh, target_fields[:, 0]),
            surface_scalar_gradient(target_mesh, target_fields[:, 1]),
        ),
        axis=1,
    )
    length_scale = max(float(np.sqrt(target_mesh.face_areas.sum())), 1e-15)
    velocity_smoother = ScreenedFieldSmoother(target_mesh, config.smoothing)

    unnormalized_initial = _objective_and_forces(
        target_mesh,
        current,
        source_values,
        target_fields,
        target_gradients,
        landmark_indices,
        landmark_targets,
        config,
        length_scale,
        landmark_normalization=1.0,
        curvature_normalization=1.0,
        need_forces=False,
    )
    landmark_normalization = max(
        unnormalized_initial.landmark_rmse**2,
        (1e-6 * target_mesh.median_edge_length) ** 2,
    )
    curvature_normalization = max(
        unnormalized_initial.curvature_rmse**2,
        1e-12,
    )
    initial_state = _objective_and_forces(
        target_mesh,
        current,
        source_values,
        target_fields,
        target_gradients,
        landmark_indices,
        landmark_targets,
        config,
        length_scale,
        landmark_normalization=landmark_normalization,
        curvature_normalization=curvature_normalization,
        need_forces=True,
    )
    records: list[RegistrationIteration] = [
        RegistrationIteration(
            0,
            initial_state.total,
            initial_state.landmark_rmse,
            initial_state.curvature_rmse,
            0.0,
            0,
        )
    ]
    steps: list[FlowStep] = []
    total_transitions = 0
    convergence_reason = "iteration budget reached"
    state = initial_state

    for iteration in range(1, config.iterations + 1):
        assert state.forces is not None
        scattered, weights = scatter_point_forces(target_mesh, list(current), state.forces)
        supported = weights > 1e-15
        scattered[supported] /= weights[supported, None]
        # Keep an ambient-continuous generator.  The face-chart tracer projects
        # it to the active tangent plane at evaluation time; projecting again
        # at irregular overlay vertices would create artificial discontinuities.
        ambient_field = velocity_smoother.smooth(scattered)
        field = IntrinsicFaceField.from_vertex_field(target_mesh, ambient_field)
        maximum = field.maximum_norm
        if maximum <= 1e-14 * length_scale:
            convergence_reason = "vanishing registration force"
            break
        trust_radius = config.max_update_fraction * target_mesh.median_edge_length
        # The assembled force has arbitrary scale because landmark and dense
        # samples scatter with different support.  Use it as a direction and
        # let the trust radius plus backtracking determine the step magnitude.
        field = field.scaled(trust_radius / maximum)

        accepted = False
        for backtrack in range(config.backtracking_steps):
            scale = 0.5**backtrack
            trial_field = field.scaled(scale)
            safe_substeps = max(
                config.integration_substeps,
                required_substeps_for_cfl(
                    target_mesh,
                    trial_field,
                    duration=1.0,
                    cfl_limit=config.cfl_limit,
                ),
            )
            trial = advect_points(
                target_mesh,
                current,
                trial_field,
                fixed_substeps=safe_substeps,
            )
            trial_state = _objective_and_forces(
                target_mesh,
                trial.points,
                source_values,
                target_fields,
                target_gradients,
                landmark_indices,
                landmark_targets,
                config,
                length_scale,
                landmark_normalization=landmark_normalization,
                curvature_normalization=curvature_normalization,
                need_forces=True,
            )
            required = config.minimum_relative_decrease * max(abs(state.total), 1.0)
            if trial_state.total <= state.total - required:
                current = trial.points
                state = trial_state
                total_transitions += trial.transition_count
                steps.append(FlowStep(trial_field, 1.0, trial.substeps))
                records.append(
                    RegistrationIteration(
                        iteration,
                        state.total,
                        state.landmark_rmse,
                        state.curvature_rmse,
                        scale,
                        trial.transition_count,
                    )
                )
                accepted = True
                break
        if not accepted:
            convergence_reason = "backtracking found no decreasing flow"
            break

    return RegistrationResult(
        current,
        FlowHistory(tuple(steps)),
        tuple(records),
        initial_state.landmark_rmse,
        state.landmark_rmse,
        initial_state.curvature_rmse,
        state.curvature_rmse,
        total_transitions,
        convergence_reason,
    )


def _objective_and_forces(
    target_mesh: SurfaceMesh,
    points: tuple[SurfacePoint, ...],
    source_values: np.ndarray,
    target_fields: np.ndarray,
    target_gradients: np.ndarray,
    landmark_indices: np.ndarray,
    landmark_targets: tuple[SurfacePoint, ...],
    config: RegistrationConfig,
    length_scale: float,
    *,
    landmark_normalization: float,
    curvature_normalization: float,
    need_forces: bool,
) -> _ObjectiveState:
    sampled_target = interpolate_vertex_field(target_mesh, target_fields, list(points))
    residual = sampled_target - source_values
    curvature_mse = float(np.mean(residual * residual))
    forces = np.zeros((len(points), 3), dtype=np.float64) if need_forces else None
    if need_forces and config.curvature_weight > 0.0:
        for sample, point in enumerate(points):
            forces[sample] -= config.curvature_weight / curvature_normalization * (
                residual[sample, 0] * target_gradients[point.face, 0]
                + residual[sample, 1] * target_gradients[point.face, 1]
            )

    if len(landmark_indices):
        current_positions = np.vstack([points[index].position(target_mesh) for index in landmark_indices])
        target_positions = np.vstack([point.position(target_mesh) for point in landmark_targets])
        landmark_delta = target_positions - current_positions
        landmark_mse = float(np.mean(np.sum(landmark_delta * landmark_delta, axis=1)))
        if need_forces and config.landmark_weight > 0.0:
            for index, delta in zip(landmark_indices, landmark_delta):
                forces[index] += config.landmark_weight * delta / landmark_normalization
    else:
        landmark_mse = 0.0

    total = (
        config.curvature_weight * curvature_mse / curvature_normalization
        + config.landmark_weight * landmark_mse / landmark_normalization
    )
    return _ObjectiveState(
        total,
        float(np.sqrt(landmark_mse)),
        float(np.sqrt(curvature_mse)),
        forces,
    )
