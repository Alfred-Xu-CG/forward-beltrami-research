import numpy as np

from qcopt.forward.discrete_symbols import central_difference_beltrami_symbol


def test_central_difference_symbol_matches_fourier_stencil():
    nx, ny = 40, 32
    mu = 0.2 + 0.1j
    kx, ky = 5, -3
    symbol = central_difference_beltrami_symbol(mu, kx, ky, nx, ny)
    dx, dy = 1.0 / nx, 1.0 / ny
    sx = np.sin(2.0 * np.pi * kx / nx) / dx
    sy = np.sin(2.0 * np.pi * ky / ny) / dy
    expected = 0.5j * sx - 0.5 * sy - mu * (0.5j * sx + 0.5 * sy)
    np.testing.assert_allclose(symbol, expected)

