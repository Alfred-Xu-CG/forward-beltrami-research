import numpy as np

from qcopt.forward.rectangle_fd import rectangle_beltrami_fd
from qcopt.forward.rectangle_fd_implicit import (
    rectangle_fd_boundary_vjp,
    rectangle_fd_coefficient_vjp,
)


def test_rectangle_boundary_vjp_matches_finite_difference():
    n = 12
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mu = np.full((n, n), 0.18 + 0.07j, dtype=np.complex128)
    boundary = xx + 1j * yy
    rng = np.random.default_rng(31)
    cotangent = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    gradient, residual = rectangle_fd_boundary_vjp(mu, boundary, cotangent)
    assert residual < 1e-9
    direction = np.zeros_like(boundary)
    direction[0, :] = rng.normal(size=n) + 1j * rng.normal(size=n)
    direction[-1, :] = rng.normal(size=n) + 1j * rng.normal(size=n)
    direction[:, 0] = rng.normal(size=n) + 1j * rng.normal(size=n)
    direction[:, -1] = rng.normal(size=n) + 1j * rng.normal(size=n)
    eps = 1e-6
    plus = rectangle_beltrami_fd(mu, boundary + eps * direction).map
    minus = rectangle_beltrami_fd(mu, boundary - eps * direction).map
    finite = float(np.real(np.vdot(cotangent, plus - minus)) / (2.0 * eps))
    predicted = float(np.real(np.vdot(gradient, direction)))
    assert abs(finite - predicted) < 2e-6


def test_rectangle_coefficient_vjp_matches_finite_difference():
    n = 10
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mu = np.full((n, n), 0.12 + 0.05j, dtype=np.complex128)
    boundary = xx + 1j * yy
    rng = np.random.default_rng(32)
    cotangent = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    gradient, residual = rectangle_fd_coefficient_vjp(mu, boundary, cotangent)
    assert residual < 1e-9
    direction = np.zeros_like(mu)
    direction[1:-1, 1:-1] = 0.01 * (
        rng.normal(size=(n - 2, n - 2)) + 1j * rng.normal(size=(n - 2, n - 2))
    )
    eps = 1e-5
    plus = rectangle_beltrami_fd(mu + eps * direction, boundary).map
    minus = rectangle_beltrami_fd(mu - eps * direction, boundary).map
    finite = float(np.real(np.vdot(cotangent, plus - minus)) / (2.0 * eps))
    predicted = float(np.real(np.vdot(gradient, direction)))
    assert abs(finite - predicted) < 2e-6


def test_rectangle_ilu_gmres_and_transpose_control_match_direct():
    from qcopt.forward.rectangle_fd_iterative import solve_rectangle_fd_gmres

    n = 24
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    coefficient = np.full((n, n), 0.14 + 0.06j, dtype=np.complex128)
    boundary = xx + 1j * yy + (0.14 + 0.06j) * (xx - 1j * yy)
    cotangent = np.zeros_like(boundary)
    cotangent[1:-1, 1:-1] = 0.3 + 0.2j
    result = solve_rectangle_fd_gmres(coefficient, boundary, transpose_rhs=cotangent)
    direct = rectangle_beltrami_fd(coefficient, boundary, method="direct")
    assert result.info == 0
    assert result.transpose_info == 0
    assert result.equation_residual < 1e-7
    assert result.transpose_residual < 1e-7
    assert np.linalg.norm(result.map - direct.map) / np.linalg.norm(direct.map) < 1e-7
