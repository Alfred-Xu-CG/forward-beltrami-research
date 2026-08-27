import numpy as np
import pytest

from qcopt.adjoint import lbs_mu_vjp, lsqc_mu_vjp
from qcopt.constraints import fixed_vertex_constraints, two_pin_constraints
from qcopt.lbs import solve_lbs
from qcopt.lsqc import solve_lsqc
from qcopt.mesh import structured_rectangle


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
