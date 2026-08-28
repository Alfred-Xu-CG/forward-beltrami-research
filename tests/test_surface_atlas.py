import numpy as np
import pytest

from qcopt.surface_atlas import (
    SurfacePoint,
    trace_tangent_step,
    transport_across_edge,
)
from qcopt.surface_mesh import SurfaceMesh


def _tetrahedron() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
    )
    return SurfaceMesh(vertices, faces)


def test_surface_point_barycentric_cartesian_round_trip():
    mesh = _tetrahedron()
    point = SurfacePoint(2, np.array([0.2, 0.3, 0.5]))
    cartesian = point.position(mesh)
    recovered = mesh.barycentric_from_point(2, cartesian)
    assert np.allclose(recovered, point.barycentric, atol=1e-14)


def test_surface_point_owns_an_immutable_barycentric_state():
    barycentric = np.array([0.2, 0.3, 0.5])
    point = SurfacePoint(0, barycentric)
    barycentric[:] = [1.0, 0.0, 0.0]

    assert np.allclose(point.barycentric, [0.2, 0.3, 0.5])
    assert not point.barycentric.flags.writeable


def test_edge_transition_preserves_length_and_is_reversible():
    mesh = _tetrahedron()
    face = 0
    local_edge = 0
    vector = np.array([0.2, 0.25, 0.0])

    neighbor, transported = transport_across_edge(mesh, face, local_edge, vector)
    neighbor_local_edge = int(np.flatnonzero(mesh.face_neighbors[neighbor] == face)[0])
    returned_face, returned = transport_across_edge(
        mesh, neighbor, neighbor_local_edge, transported
    )

    assert neighbor == mesh.face_neighbors[face, local_edge]
    assert returned_face == face
    assert np.isclose(np.linalg.norm(transported), np.linalg.norm(vector), atol=1e-14)
    assert np.allclose(returned, vector, atol=1e-14)
    assert abs(transported @ mesh.face_normals[neighbor]) < 1e-14


def test_face_walking_crosses_edge_and_returns_valid_surface_point():
    mesh = _tetrahedron()
    start = SurfacePoint(0, np.array([1.0 / 3.0] * 3))
    triangle = mesh.vertices[mesh.faces[start.face]]
    edge_midpoint = 0.5 * (triangle[1] + triangle[2])
    displacement = 1.35 * (edge_midpoint - start.position(mesh))

    result = trace_tangent_step(mesh, start, displacement)

    assert result.point.face == mesh.face_neighbors[start.face, 0]
    assert len(result.transitions) == 1
    assert np.all(result.point.barycentric >= -1e-12)
    assert np.isclose(result.point.barycentric.sum(), 1.0)
    assert result.consumed_fraction == pytest.approx(1.0)


def test_large_step_can_cross_multiple_charts_without_chart_assignment():
    mesh = _tetrahedron()
    start = SurfacePoint(0, np.array([0.8, 0.1, 0.1]))
    direction = mesh.face_frames[0, 0] + 0.31 * mesh.face_frames[0, 1]
    result = trace_tangent_step(mesh, start, 2.0 * direction, max_crossings=20)

    assert len(result.transitions) >= 2
    assert 0 <= result.point.face < mesh.n_faces
    assert np.all(result.point.barycentric >= -1e-10)


def test_open_surface_step_that_leaves_boundary_is_rejected():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = SurfaceMesh(vertices, np.array([[0, 1, 2]]))
    start = SurfacePoint(0, np.array([0.1, 0.45, 0.45]))
    displacement = np.array([0.5, 0.5, 0.0])

    with pytest.raises(RuntimeError, match="boundary"):
        trace_tangent_step(mesh, start, displacement)


def test_crossing_budget_failure_is_explicit():
    mesh = _tetrahedron()
    start = SurfacePoint(0, np.array([0.8, 0.1, 0.1]))
    direction = mesh.face_frames[0, 0] + 0.31 * mesh.face_frames[0, 1]
    with pytest.raises(RuntimeError, match="crossing budget"):
        trace_tangent_step(mesh, start, 2.0 * direction, max_crossings=1)
