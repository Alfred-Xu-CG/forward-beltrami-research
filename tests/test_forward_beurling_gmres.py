import numpy as np

from qcopt.forward.beurling_gmres import (
    periodic_beltrami_anderson,
    periodic_beltrami_gmres,
    periodic_neumann_initial_guess,
    periodic_polynomial_initial_guess,
)
from qcopt.forward.beurling import periodic_lift_derivatives


def test_periodic_gmres_solves_the_mean_zero_fixed_point_equation():
    nx, ny = 32, 24
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.35 * np.exp(2j * np.pi * (2 * xx - yy))

    result = periodic_beltrami_gmres(mu, rtol=1e-10, maxiter=100)
    fz, fbar = periodic_lift_derivatives(result.h)

    assert result.converged
    assert result.operator_residual < 1e-9
    assert np.max(np.abs(fbar - mu * fz)) < 1e-8


def test_periodic_gmres_constant_mode_still_needs_an_affine_lift():
    mu = np.full((24, 32), 0.35 + 0.1j, dtype=np.complex128)

    result = periodic_beltrami_gmres(mu, rtol=1e-10, maxiter=100)
    fz, fbar = periodic_lift_derivatives(result.h)

    assert result.converged
    assert result.operator_residual < 1e-10
    assert np.max(np.abs(fbar - mu * fz)) > 0.3


def test_periodic_gmres_warm_start_changes_iterations_not_the_certified_root():
    nx, ny = 40, 32
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.25 * np.exp(2j * np.pi * (xx + 2.0 * yy))
    warm = mu.copy()
    cold = periodic_beltrami_gmres(mu, rtol=1e-10, maxiter=100)
    seeded = periodic_beltrami_gmres(mu, rtol=1e-10, maxiter=100, initial_guess=warm)
    assert cold.converged and seeded.converged
    assert cold.operator_residual < 1e-9 and seeded.operator_residual < 1e-9
    assert np.max(np.abs(cold.h - seeded.h)) < 1e-8


def test_truncated_neumann_initializer_is_a_certified_warm_start():
    nx, ny = 64, 48
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.4 * np.exp(2j * np.pi * (2.0 * xx - yy + 0.3 * np.sin(2.0 * np.pi * yy)))
    cold = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=100)
    guess = periodic_neumann_initial_guess(mu, order=2)
    seeded = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=100, initial_guess=guess)
    assert cold.converged and seeded.converged
    assert seeded.operator_residual < 1e-8
    assert np.max(np.abs(cold.h - seeded.h)) < 1e-7
    assert np.all(np.isfinite(guess))


def test_polynomial_initializer_is_finite_and_keeps_the_certified_root():
    nx, ny = 32, 24
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.3 * np.exp(2j * np.pi * (2.0 * xx - yy))
    guess = periodic_polynomial_initial_guess(mu, (0.95 + 0.01j, 1.05 - 0.02j))
    result = periodic_beltrami_gmres(mu, initial_guess=guess, rtol=1e-9)
    cold = periodic_beltrami_gmres(mu, rtol=1e-9)
    assert result.converged and result.operator_residual < 1e-8
    np.testing.assert_allclose(result.h, cold.h, atol=1e-7, rtol=1e-7)


def test_truncated_neumann_preconditioner_preserves_root_and_residual():
    nx, ny = 64, 48
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.42 * np.exp(2j * np.pi * (2.0 * xx - yy + 0.2 * np.sin(2.0 * np.pi * yy)))
    cold = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=100)
    preconditioned = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=100, preconditioner_order=2)
    assert cold.converged and preconditioned.converged
    assert preconditioned.operator_residual < 1e-8
    np.testing.assert_allclose(preconditioned.h, cold.h, atol=1e-7, rtol=1e-7)


def test_anderson_deq_reaches_the_certified_beurling_root():
    nx, ny = 48, 40
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.28 * np.exp(2j * np.pi * (xx + 2.0 * yy))
    anderson = periodic_beltrami_anderson(mu, depth=5, rtol=1e-9, maxiter=100)
    gmres = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=100)
    assert anderson.converged
    assert anderson.fixed_point_residual < 1e-8
    np.testing.assert_allclose(anderson.h, gmres.h, atol=2e-7, rtol=2e-7)
