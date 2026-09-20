import numpy as np

from qcopt.forward.beurling import (
    periodic_beltrami_fixed_point,
    periodic_lift_derivatives,
    periodic_beurling_apply,
    periodic_cauchy_inverse,
    periodic_frequency_grid,
    torus_affine_periods,
    torus_affine_derivatives,
    torus_affine_grid,
    torus_periods_to_affine_coefficients,
    zero_padded_beurling_apply,
)


def test_periodic_beurling_multiplier_is_exact_on_a_fourier_mode():
    nx, ny = 32, 24
    kx_mode, ky_mode = 3, -2
    frequencies = periodic_frequency_grid(nx, ny)
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mode = np.exp(2j * np.pi * (kx_mode * xx + ky_mode * yy))

    transformed = periodic_beurling_apply(mode)
    expected = (kx_mode - 1j * ky_mode) / (kx_mode + 1j * ky_mode)

    assert frequencies.kx.shape == (ny, nx)
    assert np.allclose(transformed, expected * mode, atol=1e-12, rtol=1e-12)


def test_periodic_beurling_transform_annihilates_the_zero_mode():
    transformed = periodic_beurling_apply(np.ones((16, 12), dtype=np.complex128))

    assert np.max(np.abs(transformed)) < 1e-14


def test_periodic_cauchy_inverse_has_no_zero_mode_by_definition():
    inverse = periodic_cauchy_inverse(np.ones((16, 12), dtype=np.complex128))

    assert np.max(np.abs(inverse)) < 1e-14


def test_zero_padding_changes_the_boundary_response_of_a_compact_source():
    field = np.zeros((32, 32), dtype=np.complex128)
    field[0, 0] = 1.0

    periodic = periodic_beurling_apply(field)
    free_space_approx = zero_padded_beurling_apply(field, padding_factor=4)

    assert np.max(np.abs(periodic - free_space_approx)) > 1e-4


def test_periodic_fixed_point_converges_for_a_mean_zero_coefficient():
    nx, ny = 32, 24
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.2 * np.exp(2j * np.pi * (2 * xx - yy))

    result = periodic_beltrami_fixed_point(mu, max_iter=200, tolerance=1e-11)

    assert result.converged
    assert result.iterations < 200
    fz, fbar = periodic_lift_derivatives(result.h)
    assert np.max(np.abs(fbar - mu * fz)) < 1e-9


def test_constant_coefficient_exposes_the_missing_periodic_affine_mode():
    mu = np.full((24, 32), 0.2 + 0.1j, dtype=np.complex128)

    result = periodic_beltrami_fixed_point(mu, max_iter=20, tolerance=1e-12)
    fz, fbar = periodic_lift_derivatives(result.h)

    assert result.converged
    assert np.max(np.abs(fbar)) < 1e-14
    assert np.max(np.abs(fbar - mu * fz)) > 0.2


def test_torus_affine_lift_restores_constant_coefficient_with_changed_periods():
    coefficient = 0.2 + 0.1j

    fz, fbar = torus_affine_derivatives(coefficient)
    period_x, period_y = torus_affine_periods(coefficient)

    assert fz == 1.0
    assert fbar == coefficient
    assert period_x == 1.0 + coefficient
    assert period_y == 1j * (1.0 - coefficient)


def test_torus_affine_grid_has_periodic_connectivity_and_positive_lift_jacobian():
    mu = 0.35 + 0.12j
    vertices, faces = torus_affine_grid(mu, 24, 20)
    assert vertices.shape == (24 * 20, 2)
    assert faces.shape == (2 * 24 * 20, 3)
    assert np.all((faces >= 0) & (faces < len(vertices)))
    assert 1.0 - abs(mu) ** 2 > 0.0


def test_torus_periods_parameterize_the_target_affine_modulus():
    a, b, mu = torus_periods_to_affine_coefficients(1.4 + 0.2j, -0.1 + 0.8j)
    assert np.allclose(a + b, 1.4 + 0.2j)
    assert np.allclose(1j * (a - b), -0.1 + 0.8j)
    assert np.allclose(mu, b / a)
