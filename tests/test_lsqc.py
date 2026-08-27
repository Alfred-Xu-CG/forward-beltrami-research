import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.constraints import two_pin_constraints
from qcopt.lsqc import assemble_lsqc_operator, solve_lsqc
from qcopt.mesh import structured_rectangle


def _opposite_pins(mesh, target):
    return two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        target[[0, mesh.n_vertices - 1]],
    )


def test_unweighted_lsqc_operator_annihilates_compatible_affine_map():
    mesh = structured_rectangle(2, 2)
    matrix = np.array([[1.7, 0.2], [-0.3, 0.8]])
    target = mesh.vertices @ matrix.T + np.array([0.4, -0.2])
    mu = face_beltrami(mesh, target)
    vector = target.T.reshape(-1)
    residual = assemble_lsqc_operator(mesh, mu, weighted=False) @ vector

    assert np.linalg.norm(residual, ord=np.inf) < 1e-12


def test_augmented_lsqc_recovers_identity_and_affine_maps():
    mesh = structured_rectangle(4, 4)
    identity = solve_lsqc(
        mesh,
        np.zeros(mesh.n_faces, dtype=np.complex128),
        _opposite_pins(mesh, mesh.vertices),
    )
    assert np.max(np.abs(identity.uv - mesh.vertices)) < 1e-10
    assert identity.primal_residual < 1e-10
    assert identity.constraint_residual < 1e-12

    matrix = np.array([[1.3, -0.25], [0.15, 0.7]])
    target = mesh.vertices @ matrix.T + np.array([-0.2, 0.5])
    mu = face_beltrami(mesh, target)
    affine = solve_lsqc(mesh, mu, _opposite_pins(mesh, target))
    assert np.max(np.abs(affine.uv - target)) < 1e-9


def test_weighted_operator_uses_one_minus_abs_mu_squared_not_fourth_power():
    mesh = structured_rectangle(1, 1)
    mu = np.array([0.3 + 0.2j, -0.5 + 0.1j])
    unweighted = assemble_lsqc_operator(mesh, mu, weighted=False).toarray()
    weighted = assemble_lsqc_operator(mesh, mu, weighted=True).toarray()

    expected = 1.0 / np.sqrt(1.0 - np.abs(mu) ** 2)
    assert np.allclose(weighted[0:2], expected[0] * unweighted[0:2])
    assert np.allclose(weighted[2:4], expected[1] * unweighted[2:4])


def test_lsqc_state_is_augmented_not_a_normal_equation():
    mesh = structured_rectangle(2, 2)
    result = solve_lsqc(
        mesh,
        np.zeros(mesh.n_faces, dtype=np.complex128),
        _opposite_pins(mesh, mesh.vertices),
    )
    n_residuals = 2 * mesh.n_faces
    n_coordinates = 2 * mesh.n_vertices
    n_constraints = 4
    assert result.state.system.shape == (
        n_residuals + n_coordinates + n_constraints,
        n_residuals + n_coordinates + n_constraints,
    )
    assert result.state.coordinate_slice == slice(n_residuals, n_residuals + n_coordinates)
