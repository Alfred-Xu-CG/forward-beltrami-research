import numpy as np

from qcopt.forward.beurling_zero_padded import (
    zero_padded_beltrami_fixed_point,
    zero_padded_lift_derivatives,
)


def test_zero_padded_fixed_point_reconstructs_a_compact_interior_coefficient():
    nx, ny = 48, 40
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.2 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2.0 * 0.08**2))

    result = zero_padded_beltrami_fixed_point(mu, padding_factor=4, max_iter=100)
    fz, fbar = zero_padded_lift_derivatives(result.h, padding_factor=4)

    assert result.converged
    assert result.fixed_point_residual < 1e-9
    # A periodic padded box cannot invert the nonzero Fourier mode of h.  The
    # raw residual is exactly the omitted padded-box mean; after accounting for
    # that explicit normalization defect, the local equation is recovered.
    padded_mean = np.mean(result.h) / (4.0**2)
    raw_residual = np.max(np.abs(fbar - mu * fz))
    assert abs(raw_residual - abs(padded_mean)) < 1e-10
    assert np.max(np.abs(fbar + padded_mean - mu * fz)) < 1e-8


def test_zero_padded_solver_reports_nonperiodic_boundary_sensitivity():
    nx, ny = 48, 40
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.2 * np.exp(-((xx - 0.04) ** 2 + (yy - 0.5) ** 2) / (2.0 * 0.08**2))

    result = zero_padded_beltrami_fixed_point(mu, padding_factor=2, max_iter=100)
    fz, fbar = zero_padded_lift_derivatives(result.h, padding_factor=2)

    assert result.converged
    assert np.max(np.abs(fbar - mu * fz)) > 1e-5
