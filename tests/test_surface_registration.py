import numpy as np

from qcopt.surface_atlas import SurfacePoint
from qcopt.surface_flow import (
    FlowHistory,
    FlowStep,
    IntrinsicFaceField,
    intrinsic_field_lipschitz_bound,
    points_at_face_centroids,
)
from qcopt.surface_mesh import SurfaceMesh
from qcopt.surface_registration import RegistrationConfig, register_surface_map


def _asymmetric_torus(nu: int = 12, nv: int = 8) -> SurfaceMesh:
    vertices = []
    major, minor = 2.0, 0.6
    for i in range(nu):
        u = 2.0 * np.pi * i / nu
        bump = 1.0 + 0.14 * np.cos(u) + 0.06 * np.sin(2.0 * u)
        for j in range(nv):
            v = 2.0 * np.pi * j / nv
            radius = major + bump * minor * np.cos(v)
            vertices.append(
                [
                    radius * np.cos(u),
                    radius * np.sin(u),
                    bump * minor * np.sin(v),
                ]
            )
    index = lambda i, j: (i % nu) * nv + (j % nv)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i + 1, j + 1), index(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
    return SurfaceMesh(np.asarray(vertices), np.asarray(faces))


def _perturbed_points(mesh: SurfaceMesh):
    ground_truth = points_at_face_centroids(mesh)
    swirl = np.column_stack(
        (-mesh.vertices[:, 1], mesh.vertices[:, 0], np.zeros(mesh.n_vertices))
    )
    normal = np.sum(swirl * mesh.vertex_normals, axis=1)
    swirl -= normal[:, None] * mesh.vertex_normals
    history = FlowHistory((FlowStep(0.32 * swirl, duration=0.55, substeps=28),))
    return ground_truth, history.apply(mesh, ground_truth).points


def _dense_position_error(mesh, points, ground_truth):
    current = np.vstack([point.position(mesh) for point in points])
    expected = np.vstack([point.position(mesh) for point in ground_truth])
    return float(np.sqrt(np.mean(np.sum((current - expected) ** 2, axis=1))))


def test_landmark_registration_reduces_landmark_and_dense_error():
    mesh = _asymmetric_torus()
    ground_truth, initial = _perturbed_points(mesh)
    landmarks = np.arange(0, mesh.n_faces, 12, dtype=np.int64)
    config = RegistrationConfig(
        iterations=18,
        landmark_weight=8.0,
        curvature_weight=0.0,
        smoothing=0.6,
        max_update_fraction=0.12,
    )
    result = register_surface_map(
        mesh,
        mesh,
        ground_truth,
        initial,
        landmark_indices=landmarks,
        landmark_targets=tuple(ground_truth[index] for index in landmarks),
        config=config,
    )

    assert result.iterations
    assert result.iterations[-1].objective < result.iterations[0].objective
    assert result.final_landmark_rmse < 0.55 * result.initial_landmark_rmse
    assert _dense_position_error(mesh, result.points, ground_truth) < _dense_position_error(
        mesh, initial, ground_truth
    )
    assert all(
        later.objective <= earlier.objective + 1e-12
        for earlier, later in zip(result.iterations, result.iterations[1:])
    )


def test_curvature_and_landmarks_improve_both_residuals():
    mesh = _asymmetric_torus()
    ground_truth, initial = _perturbed_points(mesh)
    landmarks = np.arange(0, mesh.n_faces, 24, dtype=np.int64)
    config = RegistrationConfig(
        iterations=24,
        landmark_weight=4.0,
        curvature_weight=1.0,
        smoothing=0.8,
        max_update_fraction=0.1,
    )
    result = register_surface_map(
        mesh,
        mesh,
        ground_truth,
        initial,
        landmark_indices=landmarks,
        landmark_targets=tuple(ground_truth[index] for index in landmarks),
        config=config,
    )

    assert result.final_landmark_rmse < result.initial_landmark_rmse
    assert result.final_curvature_rmse < result.initial_curvature_rmse
    assert result.total_chart_transitions > 0
    assert len(result.flow_history.steps) > 0
    assert all(
        isinstance(step.vertex_field, IntrinsicFaceField)
        and intrinsic_field_lipschitz_bound(mesh, step.vertex_field)
        * abs(step.duration)
        / step.substeps
        <= config.cfl_limit * (1.0 + 1e-12)
        for step in result.flow_history.steps
    )


def test_mismatched_topology_or_common_sample_count_is_rejected():
    mesh = _asymmetric_torus()
    points = points_at_face_centroids(mesh)
    open_mesh = SurfaceMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]]),
    )
    try:
        register_surface_map(open_mesh, mesh, (), (), config=RegistrationConfig())
    except ValueError as error:
        assert "closed" in str(error) or "genus" in str(error)
    else:
        raise AssertionError("topology mismatch must be rejected")

    try:
        register_surface_map(mesh, mesh, points[:-1], points, config=RegistrationConfig())
    except ValueError as error:
        assert "sample" in str(error)
    else:
        raise AssertionError("sample count mismatch must be rejected")
