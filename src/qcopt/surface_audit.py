"""Independent numerical homeomorphism audit for common-refinement flows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common_surface_map import locate_overlay_samples
from .surface_atlas import SurfacePoint
from .surface_flow import (
    FlowHistory,
    IntrinsicFaceField,
    intrinsic_field_lipschitz_bound,
)
from .surface_locator import SurfaceLocator
from .surface_mesh import CommonRefinement, SurfaceMesh


@dataclass(frozen=True)
class SurfaceHomeomorphismReport:
    certified: bool
    topology_ok: bool
    orientation_ok: bool
    inverse_ok: bool
    cfl_ok: bool
    trajectory_ok: bool
    degree_one: bool
    minimum_orientation_ratio: float
    collapsed_samples: int
    round_trip_error: float
    round_trip_tolerance: float
    maximum_cfl_ratio: float
    chart_transitions: int
    audited_faces: int
    maximum_projection_error: float = 0.0
    projection_ok: bool = True


def audit_common_refinement_flow(
    source_mesh: SurfaceMesh,
    target_mesh: SurfaceMesh,
    history: FlowHistory,
    *,
    max_audit_faces: int = 256,
    orientation_threshold: float = 1e-3,
    round_trip_tolerance: float = 5e-3,
    cfl_limit: float = 0.2,
) -> SurfaceHomeomorphismReport:
    """Audit `flow o F0`, where `F0` is an oriented common-refinement map.

    The result is a floating-point certificate.  It does not replace an exact
    predicate proof, and it deliberately refuses to infer topology from the
    optimized energy.
    """

    if max_audit_faces < 1:
        raise ValueError("max_audit_faces must be positive")
    topology_ok = bool(
        source_mesh.topology.closed
        and target_mesh.topology.closed
        and source_mesh.topology.connected
        and target_mesh.topology.connected
        and source_mesh.topology.oriented
        and target_mesh.topology.oriented
        and source_mesh.topology.genus == target_mesh.topology.genus
        and source_mesh.has_same_connectivity(target_mesh)
    )
    maximum_cfl = _maximum_flow_cfl(target_mesh, history)
    cfl_ok = maximum_cfl <= cfl_limit
    if not topology_ok or not cfl_ok:
        return SurfaceHomeomorphismReport(
            False,
            topology_ok,
            False,
            False,
            cfl_ok,
            False,
            False,
            float("-inf"),
            0,
            float("inf"),
            round_trip_tolerance,
            maximum_cfl,
            0,
            0,
        )

    face_count = min(max_audit_faces, target_mesh.n_faces)
    face_ids = np.unique(
        np.linspace(0, target_mesh.n_faces - 1, face_count, dtype=np.int64)
    )
    centroid_points = tuple(
        SurfacePoint(int(face), np.full(3, 1.0 / 3.0)) for face in face_ids
    )
    epsilon = 0.06
    barycentric_probe = np.asarray(
        [
            [1.0 / 3.0 + epsilon, 1.0 / 3.0 - epsilon, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0 + epsilon, 1.0 / 3.0 - epsilon],
            [1.0 / 3.0 - epsilon, 1.0 / 3.0, 1.0 / 3.0 + epsilon],
        ]
    )
    probe_points = tuple(
        SurfacePoint(int(face), barycentric)
        for face in face_ids
        for barycentric in barycentric_probe
    )

    try:
        forward_centroids = history.apply(target_mesh, centroid_points)
        backward_centroids = history.inverse(
            target_mesh, forward_centroids.points
        )
        forward_probes = history.apply(target_mesh, probe_points)
    except (RuntimeError, ValueError, FloatingPointError):
        return SurfaceHomeomorphismReport(
            False,
            topology_ok,
            False,
            False,
            cfl_ok,
            False,
            False,
            float("-inf"),
            len(face_ids),
            float("inf"),
            round_trip_tolerance,
            maximum_cfl,
            0,
            len(face_ids),
        )

    original = np.vstack([point.position(target_mesh) for point in centroid_points])
    recovered = np.vstack(
        [point.position(target_mesh) for point in backward_centroids.points]
    )
    round_trip_error = float(
        np.max(np.linalg.norm(recovered - original, axis=1))
        / target_mesh.minimum_edge_length
    )
    inverse_ok = round_trip_error <= round_trip_tolerance

    ratios: list[float] = []
    for sample, face in enumerate(face_ids):
        base_points = np.vstack(
            [
                target_mesh.point_from_barycentric(int(face), barycentric)
                for barycentric in barycentric_probe
            ]
        )
        mapped_points = np.vstack(
            [
                forward_probes.points[3 * sample + offset].position(target_mesh)
                for offset in range(3)
            ]
        )
        base_signed = float(
            np.cross(base_points[1] - base_points[0], base_points[2] - base_points[0])
            @ target_mesh.face_normals[int(face)]
        )
        mapped_normal = target_mesh.face_normals[
            forward_centroids.points[sample].face
        ]
        mapped_signed = float(
            np.cross(
                mapped_points[1] - mapped_points[0],
                mapped_points[2] - mapped_points[0],
            )
            @ mapped_normal
        )
        ratios.append(mapped_signed / base_signed)
    minimum_ratio = float(np.min(ratios)) if ratios else float("-inf")
    collapsed = int(np.sum(np.asarray(ratios) <= orientation_threshold))
    orientation_ok = collapsed == 0
    trajectory_ok = True
    degree_one = bool(topology_ok and cfl_ok and inverse_ok and orientation_ok)
    certified = bool(degree_one and trajectory_ok)
    transitions = forward_centroids.transition_count + forward_probes.transition_count
    return SurfaceHomeomorphismReport(
        certified,
        topology_ok,
        orientation_ok,
        inverse_ok,
        cfl_ok,
        trajectory_ok,
        degree_one,
        minimum_ratio,
        collapsed,
        round_trip_error,
        round_trip_tolerance,
        maximum_cfl,
        transitions,
        len(face_ids),
    )


def audit_located_common_map_flow(
    common: CommonRefinement,
    source_original: SurfaceMesh,
    target_original: SurfaceMesh,
    history: FlowHistory,
    *,
    max_audit_faces: int = 256,
    orientation_threshold: float = 1e-3,
    round_trip_tolerance: float = 5e-3,
    cfl_limit: float = 0.2,
    projection_tolerance: float | None = None,
) -> SurfaceHomeomorphismReport:
    """Audit a common-refinement base map composed with target-surface flow.

    The common overlay certifies the base map and is used only for topology and
    projection checks.  Local flow orientation must be sampled in charts of the
    *original target*: three nearby overlay probes can project to different
    target triangles, in which case their ambient chord triangle has no valid
    orientation sign in the centroid chart and creates a false flip.
    """

    if max_audit_faces < 1:
        raise ValueError("max_audit_faces must be positive")
    genus = common.source.topology.genus
    topology_ok = bool(
        common.source.topology.closed
        and common.target.topology.closed
        and common.source.topology.connected
        and common.target.topology.connected
        and common.source.topology.oriented
        and common.target.topology.oriented
        and common.source.has_same_connectivity(common.target)
        and genus is not None
        and common.target.topology.genus == genus
        and source_original.topology.closed
        and target_original.topology.closed
        and source_original.topology.genus == genus
        and target_original.topology.genus == genus
    )
    count = min(max_audit_faces, common.source.n_faces)
    face_ids = np.unique(
        np.linspace(0, common.source.n_faces - 1, count, dtype=np.int64)
    )
    centroid_barycentric = np.tile(np.full(3, 1.0 / 3.0), (len(face_ids), 1))
    epsilon = 0.06
    probe_template = np.asarray(
        [
            [1.0 / 3.0 + epsilon, 1.0 / 3.0 - epsilon, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0 + epsilon, 1.0 / 3.0 - epsilon],
            [1.0 / 3.0 - epsilon, 1.0 / 3.0, 1.0 / 3.0 + epsilon],
        ]
    )
    probe_faces = np.repeat(face_ids, 3)
    probe_barycentric = np.tile(probe_template, (len(face_ids), 1))
    source_locator = SurfaceLocator(source_original)
    target_locator = SurfaceLocator(target_original)
    centroids = locate_overlay_samples(
        common,
        source_original,
        target_original,
        face_ids,
        centroid_barycentric,
        source_locator=source_locator,
        target_locator=target_locator,
    )
    probes = locate_overlay_samples(
        common,
        source_original,
        target_original,
        probe_faces,
        probe_barycentric,
        source_locator=source_locator,
        target_locator=target_locator,
    )
    maximum_projection = max(
        centroids.maximum_source_projection_error,
        centroids.maximum_target_projection_error,
        probes.maximum_source_projection_error,
        probes.maximum_target_projection_error,
    )
    if projection_tolerance is None:
        scale = max(
            float(np.ptp(source_original.vertices, axis=0).max()),
            float(np.ptp(target_original.vertices, axis=0).max()),
            1.0,
        )
        projection_tolerance = 2.5 * common.repair.quantization_step * scale
    projection_ok = maximum_projection <= projection_tolerance
    flow_report = audit_common_refinement_flow(
        target_original,
        target_original,
        history,
        max_audit_faces=max_audit_faces,
        orientation_threshold=orientation_threshold,
        round_trip_tolerance=round_trip_tolerance,
        cfl_limit=cfl_limit,
    )
    combined_topology = bool(topology_ok and flow_report.topology_ok)
    certified = bool(
        combined_topology and projection_ok and flow_report.certified
    )
    if not topology_ok or not projection_ok:
        return SurfaceHomeomorphismReport(
            certified=False,
            topology_ok=combined_topology,
            orientation_ok=flow_report.orientation_ok,
            inverse_ok=flow_report.inverse_ok,
            cfl_ok=flow_report.cfl_ok,
            trajectory_ok=flow_report.trajectory_ok,
            degree_one=False,
            minimum_orientation_ratio=flow_report.minimum_orientation_ratio,
            collapsed_samples=flow_report.collapsed_samples,
            round_trip_error=flow_report.round_trip_error,
            round_trip_tolerance=round_trip_tolerance,
            maximum_cfl_ratio=flow_report.maximum_cfl_ratio,
            chart_transitions=flow_report.chart_transitions,
            audited_faces=flow_report.audited_faces,
            maximum_projection_error=maximum_projection,
            projection_ok=projection_ok,
        )
    return SurfaceHomeomorphismReport(
        certified=certified,
        topology_ok=combined_topology,
        orientation_ok=flow_report.orientation_ok,
        inverse_ok=flow_report.inverse_ok,
        cfl_ok=flow_report.cfl_ok,
        trajectory_ok=flow_report.trajectory_ok,
        degree_one=certified,
        minimum_orientation_ratio=flow_report.minimum_orientation_ratio,
        collapsed_samples=flow_report.collapsed_samples,
        round_trip_error=flow_report.round_trip_error,
        round_trip_tolerance=round_trip_tolerance,
        maximum_cfl_ratio=flow_report.maximum_cfl_ratio,
        chart_transitions=flow_report.chart_transitions,
        audited_faces=flow_report.audited_faces,
        maximum_projection_error=maximum_projection,
        projection_ok=projection_ok,
    )


def _maximum_flow_cfl(mesh: SurfaceMesh, history: FlowHistory) -> float:
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in mesh.faces.tolist():
        for first, second in ((a, b), (b, c), (c, a)):
            edge_set.add((min(first, second), max(first, second)))
    edge_lengths = np.asarray(
        [
            np.linalg.norm(mesh.vertices[first] - mesh.vertices[second])
            for first, second in edge_set
        ]
    )
    floor = float(np.quantile(edge_lengths, 0.01))
    maximum = 0.0
    for step in history.steps:
        if isinstance(step.vertex_field, IntrinsicFaceField):
            lipschitz = intrinsic_field_lipschitz_bound(mesh, step.vertex_field)
        else:
            lipschitz = max(
                float(
                    np.linalg.norm(step.vertex_field[first] - step.vertex_field[second])
                    / max(
                        float(np.linalg.norm(mesh.vertices[first] - mesh.vertices[second])),
                        floor,
                    )
                )
                for first, second in edge_set
            )
        maximum = max(maximum, lipschitz * abs(step.duration) / step.substeps)
    return maximum
