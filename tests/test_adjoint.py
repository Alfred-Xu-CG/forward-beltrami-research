import numpy as np
import pytest

from qcopt.adjoint import lbs_mu_vjp, lsqc_fast_mu_vjp, lsqc_mu_vjp
from qcopt.constraints import (
    fixed_vertex_constraints,
    rectangle_sliding_constraints,
    two_pin_constraints,
)
from qcopt.lbs import solve_lbs
from qcopt.lsqc import solve_lsqc, solve_lsqc_fast
from qcopt.mesh import TriMesh, structured_rectangle


def _directional_error(analytic, evaluate, mu, direction, epsilon=1e-6):
    plus = evaluate(mu + epsilon * (direction[:, 0] + 1j * direction[:, 1]))
    minus = evaluate(mu - epsilon * (direction[:, 0] + 1j * direction[:, 1]))
    finite_difference = (plus - minus) / (2.0 * epsilon)
    prediction = float(np.sum(analytic * direction))
    return abs(prediction - finite_difference) / max(
        1.0, abs(prediction), abs(finite_difference)
    )


def test_lbs_adjoint_returns_every_facewise_mu_gradient():
    rng = np.random.default_rng(7)
    mesh = structured_rectangle(3, 3)
    boundary = mesh.boundary_loops[0]
    constraints = fixed_vertex_constraints(
        mesh.n_vertices, boundary, mesh.vertices[boundary]
    )
    mu = 0.08 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))
    result = solve_lbs(mesh, mu, constraints)
    adjoint = lbs_mu_vjp(mesh, mu, result, uv_bar)

    assert adjoint.gradient.shape == (mesh.n_faces, 2)
    assert adjoint.residual < 1e-10
    for _ in range(4):
        direction = rng.normal(size=(mesh.n_faces, 2))
        error = _directional_error(
            adjoint.gradient,
            lambda value: float(np.sum(solve_lbs(mesh, value, constraints).uv * uv_bar)),
            mu,
            direction,
        )
        assert error < 2e-6


@pytest.mark.parametrize("weighted", [False, True])
def test_lsqc_adjoint_includes_operator_and_weight_derivatives(weighted):
    rng = np.random.default_rng(19 if weighted else 11)
    mesh = structured_rectangle(2, 2)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    mu = 0.12 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))
    result = solve_lsqc(mesh, mu, constraints, weighted=weighted)
    adjoint = lsqc_mu_vjp(mesh, mu, result, uv_bar, weighted=weighted)

    assert adjoint.gradient.shape == (mesh.n_faces, 2)
    assert adjoint.residual < 1e-10
    for _ in range(4):
        direction = rng.normal(size=(mesh.n_faces, 2))
        error = _directional_error(
            adjoint.gradient,
            lambda value: float(
                np.sum(solve_lsqc(mesh, value, constraints, weighted=weighted).uv * uv_bar)
            ),
            mu,
            direction,
        )
        assert error < 3e-6


def test_fast_lsqc_adjoint_matches_augmented_and_finite_differences(monkeypatch):
    rng = np.random.default_rng(109)
    mesh = structured_rectangle(3, 3)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    mu = 0.10 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))

    import qcopt.lsqc as lsqc_module

    factorization_calls = 0
    original_splu = lsqc_module.splinalg.splu

    def counting_splu(*args, **kwargs):
        nonlocal factorization_calls
        factorization_calls += 1
        return original_splu(*args, **kwargs)

    monkeypatch.setattr(lsqc_module.splinalg, "splu", counting_splu)
    fast_result = solve_lsqc_fast(mesh, mu, constraints)
    fast_adjoint = lsqc_fast_mu_vjp(mesh, mu, fast_result, uv_bar)
    assert factorization_calls == 1

    reference_result = solve_lsqc(mesh, mu, constraints, weighted=True)
    reference_adjoint = lsqc_mu_vjp(
        mesh, mu, reference_result, uv_bar, weighted=True
    )
    assert fast_adjoint.residual < 1e-10
    assert np.max(np.abs(fast_adjoint.gradient - reference_adjoint.gradient)) < 3e-9

    for _ in range(4):
        direction = rng.normal(size=(mesh.n_faces, 2))
        error = _directional_error(
            fast_adjoint.gradient,
            lambda value: float(
                np.sum(solve_lsqc_fast(mesh, value, constraints).uv * uv_bar)
            ),
            mu,
            direction,
        )
        assert error < 3e-6


def test_fast_lsqc_adjoint_uses_specialized_weighted_residual_contraction(
    monkeypatch,
):
    """The LSQC path should avoid the more general LBS tensor contraction."""

    rng = np.random.default_rng(139)
    mesh = structured_rectangle(3, 3)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    mu = 0.12 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    result = solve_lsqc_fast(mesh, mu, constraints)

    import qcopt.adjoint as adjoint_module

    def forbidden_general_contraction(*args, **kwargs):
        raise AssertionError("general tensor contraction was used")

    monkeypatch.setattr(
        adjoint_module, "_tensor_mu_vjp", forbidden_general_contraction
    )
    adjoint = lsqc_fast_mu_vjp(
        mesh, mu, result, rng.normal(size=(mesh.n_vertices, 2))
    )

    assert adjoint.gradient.shape == (mesh.n_faces, 2)
    assert adjoint.residual < 1e-10


@pytest.mark.parametrize("weighted", [False, True])
def test_augmented_lsqc_adjoint_uses_batched_derivative_actions(monkeypatch, weighted):
    """The reference VJP must not fall back to the per-face Python block builder."""

    rng = np.random.default_rng(149 if weighted else 151)
    mesh = structured_rectangle(3, 2)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    mu = 0.08 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    result = solve_lsqc(mesh, mu, constraints, weighted=weighted)
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))

    import qcopt.adjoint as adjoint_module

    def forbidden_face_loop(*args, **kwargs):
        raise AssertionError("per-face LSQC derivative construction was used")

    monkeypatch.setattr(
        adjoint_module, "_lsqc_local_block_and_derivatives", forbidden_face_loop
    )
    adjoint = lsqc_mu_vjp(mesh, mu, result, uv_bar, weighted=weighted)

    assert adjoint.gradient.shape == (mesh.n_faces, 2)
    assert adjoint.residual < 1e-10


def test_fast_lsqc_matches_reference_on_irregular_high_distortion_mesh():
    """Exercise the reduced solve and VJP away from a uniform low-mu grid."""

    rng = np.random.default_rng(173)
    base = structured_rectangle(7, 5)
    vertices = base.vertices.copy()
    boundary = np.unique(np.concatenate(base.boundary_loops))
    interior_mask = np.ones(base.n_vertices, dtype=bool)
    interior_mask[boundary] = False
    vertices[interior_mask] += rng.uniform(
        -0.12, 0.12, size=(np.count_nonzero(interior_mask), 2)
    ) * np.array([1.0 / 7.0, 1.0 / 5.0])
    mesh = TriMesh(vertices, base.faces)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    mu = 0.9 * np.exp(2j * np.pi * rng.random(mesh.n_faces))
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))

    reference_result = solve_lsqc(mesh, mu, constraints, weighted=True)
    fast_result = solve_lsqc_fast(mesh, mu, constraints)
    reference_adjoint = lsqc_mu_vjp(
        mesh, mu, reference_result, uv_bar, weighted=True
    )
    fast_adjoint = lsqc_fast_mu_vjp(mesh, mu, fast_result, uv_bar)

    assert np.max(np.abs(fast_result.uv - reference_result.uv)) < 1e-9
    assert np.max(
        np.abs(fast_adjoint.gradient - reference_adjoint.gradient)
    ) < 1e-8
    assert fast_result.primal_residual < 1e-10
    assert fast_adjoint.residual < 1e-10


@pytest.mark.parametrize("constraint_kind", ["fixed_boundary", "sliding"])
def test_fast_lsqc_adjoint_handles_fixed_and_sliding_boundaries(constraint_kind):
    rng = np.random.default_rng(181 if constraint_kind == "sliding" else 179)
    mesh = structured_rectangle(4, 3)
    if constraint_kind == "fixed_boundary":
        boundary = mesh.boundary_loops[0]
        constraints = fixed_vertex_constraints(
            mesh.n_vertices, boundary, mesh.vertices[boundary]
        )
    else:
        constraints = rectangle_sliding_constraints(mesh)
    mu = 0.16 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))

    reference_result = solve_lsqc(mesh, mu, constraints, weighted=True)
    fast_result = solve_lsqc_fast(mesh, mu, constraints)
    reference_adjoint = lsqc_mu_vjp(
        mesh, mu, reference_result, uv_bar, weighted=True
    )
    fast_adjoint = lsqc_fast_mu_vjp(mesh, mu, fast_result, uv_bar)

    assert np.max(
        np.abs(fast_adjoint.gradient - reference_adjoint.gradient)
    ) < 3e-9
    direction = rng.normal(size=(mesh.n_faces, 2))
    error = _directional_error(
        fast_adjoint.gradient,
        lambda value: float(
            np.sum(solve_lsqc_fast(mesh, value, constraints).uv * uv_bar)
        ),
        mu,
        direction,
    )
    assert error < 3e-6
