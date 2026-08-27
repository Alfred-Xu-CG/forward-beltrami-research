import numpy as np
import pytest

from qcopt.mesh import TriMesh, structured_rectangle


def test_unit_triangle_gradients_reproduce_affine_coordinates():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = TriMesh(vertices, np.array([[0, 1, 2]]))

    assert np.allclose(mesh.areas, [0.5])
    assert np.allclose(mesh.gradients[0].T @ vertices[:, 0], [1.0, 0.0])
    assert np.allclose(mesh.gradients[0].T @ vertices[:, 1], [0.0, 1.0])
    assert np.allclose(mesh.gradients[0].sum(axis=0), [0.0, 0.0])


def test_structured_rectangle_is_positive_and_has_one_ordered_boundary():
    mesh = structured_rectangle(3, 2)

    assert mesh.n_vertices == 12
    assert mesh.n_faces == 12
    assert np.all(mesh.areas > 0.0)
    assert len(mesh.boundary_loops) == 1
    loop = mesh.boundary_loops[0]
    assert len(loop) == 10
    polygon = mesh.vertices[loop]
    twice_area = np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    assert twice_area > 0.0


def test_rejects_degenerate_or_clockwise_faces():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive orientation"):
        TriMesh(vertices, np.array([[0, 2, 1]]))
    with pytest.raises(ValueError, match="distinct"):
        TriMesh(vertices, np.array([[0, 1, 1]]))
