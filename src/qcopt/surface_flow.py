"""Small reversible flows of global points through a surface's local charts."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from .surface_atlas import SurfacePoint, trace_tangent_step, transport_across_edge
from .surface_geometry import interpolate_vertex_field
from .surface_mesh import SurfaceMesh


@dataclass(frozen=True)
class AdvectionResult:
    points: tuple[SurfacePoint, ...]
    transition_count: int
    substeps: int
    maximum_step_length: float


@dataclass(frozen=True)
class IntrinsicFaceField:
    """Edge-compatible quadratic tangent field in the native face atlas.

    Vertex values vanish at PL cone singularities.  Every shared edge owns one
    midpoint value; its representation in the adjacent face is obtained by the
    exact unfolding transition.  A face bubble restores interior freedom while
    vanishing on all chart boundaries.
    """

    edge_values: np.ndarray
    bubble_values: np.ndarray

    def __post_init__(self) -> None:
        edge_values = np.asarray(self.edge_values, dtype=np.float64)
        bubble_values = np.asarray(self.bubble_values, dtype=np.float64)
        if (
            edge_values.ndim != 3
            or edge_values.shape[1:] != (3, 3)
            or bubble_values.shape != (edge_values.shape[0], 3)
            or not np.all(np.isfinite(edge_values))
            or not np.all(np.isfinite(bubble_values))
        ):
            raise ValueError("invalid intrinsic face-field arrays")
        edge_values = np.array(edge_values, copy=True, order="C")
        bubble_values = np.array(bubble_values, copy=True, order="C")
        edge_values.setflags(write=False)
        bubble_values.setflags(write=False)
        object.__setattr__(self, "edge_values", edge_values)
        object.__setattr__(self, "bubble_values", bubble_values)

    @classmethod
    def from_vertex_field(
        cls, mesh: SurfaceMesh, vertex_field: np.ndarray
    ) -> "IntrinsicFaceField":
        vertex_field = np.asarray(vertex_field, dtype=np.float64)
        if vertex_field.shape != (mesh.n_vertices, 3):
            raise ValueError("vertex_field must have shape (mesh.n_vertices, 3)")
        edge_values = np.zeros((mesh.n_faces, 3, 3), dtype=np.float64)
        assigned = np.zeros((mesh.n_faces, 3), dtype=bool)
        for face in range(mesh.n_faces):
            for local_edge in range(3):
                if assigned[face, local_edge]:
                    continue
                endpoints = mesh.faces[
                    face, [(local_edge + 1) % 3, (local_edge + 2) % 3]
                ]
                ambient = vertex_field[endpoints].mean(axis=0)
                source_value = ambient - float(
                    ambient @ mesh.face_normals[face]
                ) * mesh.face_normals[face]
                edge_values[face, local_edge] = source_value
                assigned[face, local_edge] = True
                neighbor = int(mesh.face_neighbors[face, local_edge])
                if neighbor >= 0:
                    neighbor_local = int(
                        np.flatnonzero(mesh.face_neighbors[neighbor] == face)[0]
                    )
                    _, target_value = transport_across_edge(
                        mesh, face, local_edge, source_value
                    )
                    edge_values[neighbor, neighbor_local] = target_value
                    assigned[neighbor, neighbor_local] = True

        bubble_values = np.empty((mesh.n_faces, 3), dtype=np.float64)
        for face in range(mesh.n_faces):
            ambient = vertex_field[mesh.faces[face]].mean(axis=0)
            ambient -= float(ambient @ mesh.face_normals[face]) * mesh.face_normals[face]
            edge_at_centroid = (4.0 / 9.0) * edge_values[face].sum(axis=0)
            bubble_values[face] = ambient - edge_at_centroid
        return cls(edge_values, bubble_values)

    @property
    def maximum_norm(self) -> float:
        return max(
            float(np.max(np.linalg.norm(self.edge_values, axis=2), initial=0.0)),
            float(np.max(np.linalg.norm(self.bubble_values, axis=1), initial=0.0)),
        )

    def scaled(self, factor: float) -> "IntrinsicFaceField":
        return IntrinsicFaceField(
            float(factor) * self.edge_values,
            float(factor) * self.bubble_values,
        )


@dataclass(frozen=True)
class FlowStep:
    vertex_field: np.ndarray | IntrinsicFaceField
    duration: float
    substeps: int

    def __post_init__(self) -> None:
        if isinstance(self.vertex_field, IntrinsicFaceField):
            field: np.ndarray | IntrinsicFaceField = self.vertex_field
        else:
            array = np.asarray(self.vertex_field, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
                raise ValueError("vertex_field must be a finite (n, 3) array")
            array = np.array(array, copy=True, order="C")
            array.setflags(write=False)
            field = array
        if not np.isfinite(self.duration):
            raise ValueError("duration must be finite")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        object.__setattr__(self, "vertex_field", field)
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "substeps", int(self.substeps))


@dataclass(frozen=True)
class FlowHistory:
    steps: tuple[FlowStep, ...] = ()

    def apply(
        self, mesh: SurfaceMesh, points: tuple[SurfacePoint, ...] | list[SurfacePoint]
    ) -> AdvectionResult:
        current = tuple(points)
        transitions = 0
        substeps = 0
        maximum = 0.0
        for step in self.steps:
            result = advect_points(
                mesh,
                current,
                step.vertex_field,
                duration=step.duration,
                fixed_substeps=step.substeps,
            )
            current = result.points
            transitions += result.transition_count
            substeps += result.substeps
            maximum = max(maximum, result.maximum_step_length)
        return AdvectionResult(current, transitions, substeps, maximum)

    def inverse(
        self, mesh: SurfaceMesh, points: tuple[SurfacePoint, ...] | list[SurfacePoint]
    ) -> AdvectionResult:
        current = tuple(points)
        transitions = 0
        substeps = 0
        maximum = 0.0
        for step in reversed(self.steps):
            result = advect_points(
                mesh,
                current,
                step.vertex_field,
                duration=-step.duration,
                fixed_substeps=step.substeps,
            )
            current = result.points
            transitions += result.transition_count
            substeps += result.substeps
            maximum = max(maximum, result.maximum_step_length)
        return AdvectionResult(current, transitions, substeps, maximum)


def points_at_vertices(
    mesh: SurfaceMesh, *, interior_epsilon: float = 1e-8
) -> tuple[SurfacePoint, ...]:
    """Give every used vertex a deterministic, infinitesimally interior chart.

    A PL vertex has no unique tangent plane.  The tiny one-ring offset chooses a
    reproducible branch for numerical flow integration while converging to the
    exact vertex as ``interior_epsilon`` tends to zero.
    """

    if not 0.0 < interior_epsilon < 1.0 / 3.0:
        raise ValueError("interior_epsilon must lie in (0, 1/3)")
    result: list[SurfacePoint] = []
    for vertex, incident in enumerate(mesh.vertex_faces):
        if len(incident) == 0:
            raise ValueError("surface contains an isolated vertex")
        face = int(incident[0])
        local = int(np.flatnonzero(mesh.faces[face] == vertex)[0])
        barycentric = np.full(3, 0.5 * interior_epsilon)
        barycentric[local] = 1.0 - interior_epsilon
        result.append(SurfacePoint(face, barycentric))
    return tuple(result)


def points_at_face_centroids(mesh: SurfaceMesh) -> tuple[SurfacePoint, ...]:
    """Return one nonsingular audit sample in every face chart."""

    return tuple(
        SurfacePoint(face, np.full(3, 1.0 / 3.0))
        for face in range(mesh.n_faces)
    )


def advect_points(
    mesh: SurfaceMesh,
    points: tuple[SurfacePoint, ...] | list[SurfacePoint],
    vertex_field: np.ndarray | IntrinsicFaceField,
    *,
    duration: float = 1.0,
    max_step_fraction: float = 0.2,
    fixed_substeps: int | None = None,
    max_crossings_per_substep: int = 64,
) -> AdvectionResult:
    """Integrate a PL tangent field with explicit midpoint face-chart steps."""

    if isinstance(vertex_field, IntrinsicFaceField):
        field: np.ndarray | IntrinsicFaceField = vertex_field
        if field.edge_values.shape[0] != mesh.n_faces:
            raise ValueError("intrinsic field face count does not match mesh")
        max_speed = field.maximum_norm
    else:
        array = np.asarray(vertex_field, dtype=np.float64)
        if array.shape != (mesh.n_vertices, 3) or not np.all(np.isfinite(array)):
            raise ValueError("vertex_field must have shape (mesh.n_vertices, 3)")
        field = array
        max_speed = float(np.max(np.linalg.norm(field, axis=1)))
    if max_step_fraction <= 0.0:
        raise ValueError("max_step_fraction must be positive")
    current = tuple(points)
    if not current:
        return AdvectionResult((), 0, 1, 0.0)
    if fixed_substeps is None:
        permitted = max_step_fraction * mesh.minimum_edge_length
        substeps = max(1, int(ceil(abs(duration) * max_speed / permitted)))
    else:
        substeps = int(fixed_substeps)
        if substeps < 1:
            raise ValueError("fixed_substeps must be positive")
    dt = float(duration) / substeps
    transitions = 0
    maximum_step = 0.0

    for _ in range(substeps):
        next_points: list[SurfacePoint] = []
        # Midpoint sampling reduces the forward/backward asymmetry of Euler
        # integration while still allowing face ids to change dynamically.
        initial_velocity = evaluate_flow_field(mesh, field, current)
        midpoint_points: list[SurfacePoint] = []
        for point, velocity in zip(current, initial_velocity):
            half = trace_tangent_step(
                mesh,
                point,
                0.5 * dt * velocity,
                max_crossings=max_crossings_per_substep,
            )
            midpoint_points.append(half.point)
        midpoint_velocity = evaluate_flow_field(mesh, field, tuple(midpoint_points))
        for point, velocity in zip(current, midpoint_velocity):
            displacement = dt * velocity
            maximum_step = max(maximum_step, float(np.linalg.norm(displacement)))
            traced = trace_tangent_step(
                mesh,
                point,
                displacement,
                max_crossings=max_crossings_per_substep,
            )
            transitions += len(traced.transitions)
            next_points.append(traced.point)
        current = tuple(next_points)
    return AdvectionResult(current, transitions, substeps, maximum_step)


def evaluate_flow_field(
    mesh: SurfaceMesh,
    field: np.ndarray | IntrinsicFaceField,
    points: tuple[SurfacePoint, ...] | list[SurfacePoint],
) -> np.ndarray:
    if not isinstance(field, IntrinsicFaceField):
        return interpolate_vertex_field(mesh, np.asarray(field), list(points))
    result = np.empty((len(points), 3), dtype=np.float64)
    for index, point in enumerate(points):
        barycentric = point.barycentric
        value = np.zeros(3)
        for local_edge in range(3):
            first, second = (local_edge + 1) % 3, (local_edge + 2) % 3
            value += (
                4.0
                * barycentric[first]
                * barycentric[second]
                * field.edge_values[point.face, local_edge]
            )
        value += (
            27.0
            * float(np.prod(barycentric))
            * field.bubble_values[point.face]
        )
        result[index] = value
    return result


def intrinsic_field_lipschitz_bound(
    mesh: SurfaceMesh, field: IntrinsicFaceField
) -> float:
    """Conservative representative bound for a quadratic atlas field.

    Exported real meshes often contain a handful of sub-quantization edges.
    The first-percentile minimum face edge is used consistently by both the
    integrator policy and independent audit; the exact minimum remains visible
    in mesh statistics but would let one isolated edge dictate every time step.
    """

    if field.edge_values.shape[0] != mesh.n_faces:
        raise ValueError("intrinsic field face count does not match mesh")
    triangles = mesh.vertices[mesh.faces]
    minimum_face_edges = np.minimum.reduce(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        )
    )
    representative_scale = float(np.quantile(minimum_face_edges, 0.01))
    if representative_scale <= 0.0:
        raise ValueError("mesh has no positive representative face scale")
    return 12.0 * field.maximum_norm / representative_scale


def required_substeps_for_cfl(
    mesh: SurfaceMesh,
    field: IntrinsicFaceField,
    *,
    duration: float,
    cfl_limit: float,
) -> int:
    """Return the smallest integer subdivision satisfying the field CFL."""

    if cfl_limit <= 0.0 or not np.isfinite(cfl_limit):
        raise ValueError("cfl_limit must be positive and finite")
    if not np.isfinite(duration):
        raise ValueError("duration must be finite")
    bound = intrinsic_field_lipschitz_bound(mesh, field)
    return max(1, int(ceil(bound * abs(float(duration)) / cfl_limit)))
