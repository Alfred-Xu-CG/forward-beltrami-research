import numpy as np

from qcopt.forward.rectangle_conductivity import rectangle_beltrami_conductivity


def test_conductivity_recovery_matches_affine_boundary_data() -> None:
    n = 20
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    coefficient = 0.18 + 0.11j
    truth = xx + 1j * yy + coefficient * (xx - 1j * yy)
    mu = np.full((n, n), coefficient, dtype=np.complex128)
    result = rectangle_beltrami_conductivity(mu, truth)
    assert result.converged
    assert np.max(np.abs(result.map - truth)) < 2e-10
    assert result.boundary_imag_mismatch < 1e-12


def test_conductivity_recovery_has_no_central_difference_checkerboard() -> None:
    n = 24
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    amplitude = 0.08
    truth = xx + amplitude * np.sin(2.0 * np.pi * yy) + 1j * yy
    beta = amplitude * np.pi * np.cos(2.0 * np.pi * yy)
    mu = 1j * beta / (1.0 - 1j * beta)
    result = rectangle_beltrami_conductivity(mu, truth)
    assert result.converged
    assert result.min_triangle_determinant > 0.1
    assert np.sqrt(np.mean(np.abs(result.map - truth) ** 2)) < 3e-3
    assert result.equation_residual < 5e-2


def test_conductivity_ignores_supplied_interior_values() -> None:
    n = 20
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    coefficient = 0.18 + 0.11j
    truth = xx + 1j * yy + coefficient * (xx - 1j * yy)
    mu = np.full((n, n), coefficient, dtype=np.complex128)

    boundary_only = truth.copy()
    interior = np.ones((n, n), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    boundary_only[interior] = 37.0 - 19.0j

    result = rectangle_beltrami_conductivity(mu, boundary_only)
    assert result.converged
    assert np.max(np.abs(result.map - truth)) < 2e-10
