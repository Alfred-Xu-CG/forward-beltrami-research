"""Discretization-matched Fourier symbols for central-difference dbar operators."""

from __future__ import annotations

import numpy as np


def central_difference_beltrami_symbol(
    mu: complex, kx: int, ky: int, nx: int, ny: int
) -> complex:
    """Return the periodic central-difference symbol of ``dbar-mu*dz``.

    This is not the continuous Beurling multiplier. With grid spacings
    ``dx=1/nx`` and ``dy=1/ny``, the derivative symbols are
    ``i sin(2*pi*kx/nx)/dx`` and ``i sin(2*pi*ky/ny)/dy``. The distinction is
    important when a Fourier preconditioner is used for a finite-difference or
    P1 discretization.
    """

    mu = complex(mu)
    if not np.isfinite(mu.real) or not np.isfinite(mu.imag) or abs(mu) >= 1.0:
        raise ValueError("mu must be finite and lie inside the unit disk")
    if nx < 3 or ny < 3:
        raise ValueError("grid dimensions must be at least three")
    sx = np.sin(2.0 * np.pi * kx / nx) / (1.0 / nx)
    sy = np.sin(2.0 * np.pi * ky / ny) / (1.0 / ny)
    dbar = 0.5j * sx - 0.5 * sy
    dz = 0.5j * sx + 0.5 * sy
    return complex(dbar - mu * dz)

