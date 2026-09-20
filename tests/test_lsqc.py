import numpy as np
import pytest
from scipy import sparse

from qcopt.beltrami import face_beltrami
from qcopt.constraints import (
    LinearConstraints,
    fixed_vertex_constraints,
    rectangle_sliding_constraints,
    two_pin_constraints,
)
from qcopt.lsqc import (
    assemble_lsqc_operator,
    assemble_weighted_lsqc_hessian,
    solve_lsqc,
    solve_lsqc_fast,
)
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


def test_direct_weighted_hessian_matches_reference_normal_equation():
    rng = np.random.default_rng(101)
    mesh = structured_rectangle(3, 2)
    mu = 0.18 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))

    operator = assemble_lsqc_operator(mesh, mu, weighted=True)
    reference = (operator.T @ operator).tocsr()
    direct = assemble_weighted_lsqc_hessian(mesh, mu)

    assert direct.shape == (2 * mesh.n_vertices, 2 * mesh.n_vertices)
    assert np.max(np.abs((direct - direct.T).data), initial=0.0) < 1e-13
    assert np.max(np.abs((direct - reference).data), initial=0.0) < 2e-12


@pytest.mark.parametrize("constraint_kind", ["two_pin", "fixed_boundary", "sliding"])
def test_fast_weighted_lsqc_matches_augmented_reference(constraint_kind):
    rng = np.random.default_rng(202)
    mesh = structured_rectangle(5, 4)
    mu = 0.08 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))

    if constraint_kind == "two_pin":
        constraints = _opposite_pins(mesh, mesh.vertices)
    elif constraint_kind == "fixed_boundary":
        boundary = mesh.boundary_loops[0]
        constraints = fixed_vertex_constraints(
            mesh.n_vertices, boundary, mesh.vertices[boundary]
        )
    else:
        constraints = rectangle_sliding_constraints(mesh)

    reference = solve_lsqc(mesh, mu, constraints, weighted=True)
    fast = solve_lsqc_fast(mesh, mu, constraints)

    assert np.max(np.abs(fast.uv - reference.uv)) < 2e-9
    assert fast.primal_residual < 1e-10
    assert fast.constraint_residual < 1e-12
    assert fast.state.system.shape == (
        2 * mesh.n_vertices - constraints.C.shape[0],
        2 * mesh.n_vertices - constraints.C.shape[0],
    )


def test_fast_lsqc_rejects_nonselector_constraints():
    mesh = structured_rectangle(2, 2)
    matrix = sparse.csr_matrix(
        ([1.0, -1.0], ([0, 0], [0, 1])), shape=(1, 2 * mesh.n_vertices)
    )
    constraints = LinearConstraints(matrix, np.array([0.0]))

    with pytest.raises(ValueError, match="coordinate-selector"):
        solve_lsqc_fast(mesh, np.zeros(mesh.n_faces, dtype=np.complex128), constraints)


def test_area_coupling_is_assembled_from_oriented_boundary_edges():
    from qcopt.lsqc import assemble_lsqc_area_coupling

    mesh = structured_rectangle(5, 4)
    operator = assemble_lsqc_operator(
        mesh, np.zeros(mesh.n_faces, dtype=np.complex128), weighted=True
    )
    normal = (operator.T @ operator).tocsr()
    coupling = assemble_lsqc_area_coupling(mesh)
    n = mesh.n_vertices

    assert np.max(np.abs((coupling - normal[:n, n:]).data), initial=0.0) < 1e-13
    assert coupling.nnz == 2 * sum(len(loop) for loop in mesh.boundary_loops)


def test_fast_lsqc_uses_fill_reducing_symmetric_ordering(monkeypatch):
    import qcopt.lsqc as lsqc_module

    mesh = structured_rectangle(3, 3)
    constraints = _opposite_pins(mesh, mesh.vertices)
    calls = []
    original_splu = lsqc_module.splinalg.splu

    def recording_splu(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_splu(*args, **kwargs)

    monkeypatch.setattr(lsqc_module.splinalg, "splu", recording_splu)
    result = solve_lsqc_fast(
        mesh, np.zeros(mesh.n_faces, dtype=np.complex128), constraints
    )

    assert calls == [
        {
            "permc_spec": "MMD_AT_PLUS_A",
            "options": {"SymmetricMode": True},
        }
    ]
    assert not hasattr(result.state, "hessian")
