import numpy as np

from qcopt.forward.discrete_conjugacy import (
    assemble_facewise_conductivity,
    p1_conjugacy_residual,
)
from qcopt.mesh import structured_rectangle


def _layered_manufactured_map():
    mesh = structured_rectangle(4, 2)
    y = mesh.vertices[:, 1]
    g = np.where(y <= 0.5, 0.4 * y, 0.2 + 1.6 * (y - 0.5))
    u = mesh.vertices[:, 0]
    v = g
    face_a = []
    for face in mesh.faces:
        y_face = float(np.mean(mesh.vertices[face, 1]))
        slope = 0.4 if y_face < 0.5 else 1.6
        face_a.append(np.diag((slope, 1.0 / slope)))
    return mesh, u, v, np.asarray(face_a)


def test_layered_piecewise_affine_map_is_exactly_conjugate_and_mixed_harmonic():
    mesh, u, v, face_a = _layered_manufactured_map()
    residual = p1_conjugacy_residual(mesh, u, v, face_a)
    assert residual < 1e-12
    stiffness = assemble_facewise_conductivity(mesh, face_a)
    left_right = np.isclose(mesh.vertices[:, 0], 0.0) | np.isclose(mesh.vertices[:, 0], 1.0)
    assert np.max(np.abs((stiffness @ u)[~left_right])) < 1e-12
    assert np.max(np.abs((stiffness @ v)[~np.isclose(mesh.vertices[:, 1], 0.0) & ~np.isclose(mesh.vertices[:, 1], 1.0)])) < 1e-12


def test_layered_map_is_a_homeomorphism_with_positive_face_determinants():
    mesh, u, v, face_a = _layered_manufactured_map()
    determinants = []
    for face in mesh.faces:
        points = np.column_stack((u[face], v[face]))
        determinants.append(np.linalg.det(np.column_stack((points[1] - points[0], points[2] - points[0]))))
    assert min(determinants) > 0.0


def test_conjugacy_rejects_negative_definite_and_non_unit_determinant_tensors():
    mesh, u, v, _ = _layered_manufactured_map()
    negative = np.repeat((-np.eye(2))[None, :, :], mesh.n_faces, axis=0)
    with np.testing.assert_raises(ValueError):
        p1_conjugacy_residual(mesh, u, v, negative)
    non_unit = np.repeat(np.diag((2.0, 1.0))[None, :, :], mesh.n_faces, axis=0)
    with np.testing.assert_raises(ValueError):
        p1_conjugacy_residual(mesh, u, v, non_unit)
