"""Whole-plane truncation reference solver on a bounded sampling window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beurling import (
    periodic_frequency_grid,
    zero_padded_beurling_apply,
    zero_padded_cauchy_inverse,
    zero_padded_dbar_apply,
    zero_padded_dz_apply,
)


@dataclass(frozen=True)
class ZeroPaddedFixedPointResult:
    h: np.ndarray
    iterations: int
    fixed_point_residual: float
    converged: bool


def zero_padded_beltrami_fixed_point(
    mu: np.ndarray,
    *,
    padding_factor: int = 2,
    max_iter: int = 200,
    tolerance: float = 1e-10,
) -> ZeroPaddedFixedPointResult:
    """Solve the compact-support whole-plane truncation by Neumann iteration."""

    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 2:
        raise ValueError("mu must be a two-dimensional grid")
    if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must be finite and lie inside the unit disk")
    if max_iter < 1 or tolerance <= 0.0:
        raise ValueError("max_iter and tolerance must be positive")
    h = np.zeros_like(coefficients)
    residual = float("inf")
    for iteration in range(1, max_iter + 1):
        updated = coefficients * (
            1.0 + zero_padded_beurling_apply(h, padding_factor=padding_factor)
        )
        residual = float(np.max(np.abs(updated - h)))
        h = updated
        if residual <= tolerance:
            return ZeroPaddedFixedPointResult(h, iteration, residual, True)
    return ZeroPaddedFixedPointResult(h, max_iter, residual, False)


def zero_padded_lift_derivatives(
    h: np.ndarray, *, padding_factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return derivatives of ``f=z+C(h)`` on the bounded output window."""

    # Keep the Cauchy inverse and both derivatives on the *same* padded box.
    # Applying the public crop-after-each-operation helpers successively would
    # re-pad an already cropped field and introduce an artificial boundary
    # error unrelated to the whole-plane truncation.
    values = np.asarray(h, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("h must be a two-dimensional array")
    if not isinstance(padding_factor, (int, np.integer)) or padding_factor < 2:
        raise ValueError("padding_factor must be an integer at least two")
    ny, nx = values.shape
    padded_shape = (int(padding_factor) * ny, int(padding_factor) * nx)
    offset_y = (padded_shape[0] - ny) // 2
    offset_x = (padded_shape[1] - nx) // 2
    padded = np.zeros(padded_shape, dtype=np.complex128)
    padded[offset_y : offset_y + ny, offset_x : offset_x + nx] = values
    frequencies = periodic_frequency_grid(padded_shape[1], padded_shape[0])
    denominator = frequencies.kx + 1j * frequencies.ky
    cauchy = frequencies.cauchy_inverse
    correction = np.fft.ifft2(np.fft.fft2(padded) * cauchy)
    dz = 0.5j * (frequencies.kx - 1j * frequencies.ky)
    dbar = 0.5j * (frequencies.kx + 1j * frequencies.ky)
    fz_full = 1.0 + np.fft.ifft2(np.fft.fft2(correction) * dz)
    fbar_full = np.fft.ifft2(np.fft.fft2(correction) * dbar)
    del denominator
    fz = fz_full[offset_y : offset_y + ny, offset_x : offset_x + nx]
    fbar = fbar_full[offset_y : offset_y + ny, offset_x : offset_x + nx]
    return fz, fbar
