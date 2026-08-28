import numpy as np
import pytest

from qcopt.surface_locator import SurfaceLocator
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
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return SurfaceMesh(vertices, faces)


def test_locator_recovers_exact_face_and_barycentric_coordinates():
    mesh = _tetrahedron()
    locator = SurfaceLocator(mesh)
    face = 2
    barycentric = np.array([0.2, 0.3, 0.5])
    position = mesh.point_from_barycentric(face, barycentric)
    result = locator.locate(position)

    assert result.distance < 1e-14
    assert np.allclose(result.point.position(mesh), position, atol=1e-14)
    assert np.all(result.point.barycentric >= 0.0)


def test_locator_projects_off_surface_query_to_closest_triangle():
    mesh = _tetrahedron()
    locator = SurfaceLocator(mesh)
    result = locator.locate(np.array([0.25, 0.25, -0.2]))
    assert result.point.face == 0
    assert np.allclose(result.point.position(mesh), [0.25, 0.25, 0.0], atol=1e-14)
    assert np.isclose(result.distance, 0.2)


def test_batch_location_reports_maximum_projection_error():
    mesh = _tetrahedron()
    locator = SurfaceLocator(mesh)
    queries = np.array([[0.2, 0.2, 0.0], [0.2, 0.0, 0.4], [0.0, 0.3, 0.2]])
    batch = locator.locate_many(queries)
    recovered = np.vstack([point.position(mesh) for point in batch.points])
    assert np.allclose(recovered, queries, atol=1e-14)
    assert batch.maximum_distance < 1e-14


def test_locator_does_not_miss_a_long_skinny_closest_triangle():
    # With candidate_count=1, both the nearest centroid and nearest vertex
    # belong to the compact decoy triangle.  The long triangle nevertheless
    # passes directly underneath the query and is the true closest face.
    mesh = SurfaceMesh(
        np.asarray(
            [
                [-100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 1.0, 0.0],
                [4.0, 4.0, 0.0],
                [5.0, 4.0, 0.0],
                [4.0, 5.0, 0.0],
            ]
        ),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )
    result = SurfaceLocator(mesh, candidate_count=1).locate(
        np.asarray([0.0, 0.1, 0.2])
    )

    assert result.point.face == 0
    assert result.distance == pytest.approx(0.2)
