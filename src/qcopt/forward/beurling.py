"""Periodic Fourier-symbol operators for the discrete Beurling route."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PeriodicFrequencies:
    """Fourier frequencies and zero-mode-safe multipliers on a flat torus."""

    kx: ComplexArray
    ky: ComplexArray
    beurling: ComplexArray
    cauchy_inverse: ComplexArray


@dataclass(frozen=True)
class PeriodicFixedPointResult:
    """Result of the periodic Neumann iteration for ``h = mu(1+B h)``."""

    h: ComplexArray
    iterations: int
    residual: float
    converged: bool


def periodic_frequency_grid(nx: int, ny: int) -> PeriodicFrequencies:
    """Return angular frequencies and symbols for an ``nx`` by ``ny`` torus."""

    if not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
        raise TypeError("nx and ny must be integers")
    if nx < 2 or ny < 2:
        raise ValueError("periodic grid dimensions must both be at least two")
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")
    denominator = kx + 1j * ky
    nonzero = np.abs(denominator) > 0.0
    beurling = np.zeros_like(denominator, dtype=np.complex128)
    cauchy_inverse = np.zeros_like(denominator, dtype=np.complex128)
    beurling[nonzero] = (kx[nonzero] - 1j * ky[nonzero]) / denominator[nonzero]
    cauchy_inverse[nonzero] = 2.0 / (1j * denominator[nonzero])
    return PeriodicFrequencies(
        np.ascontiguousarray(kx),
        np.ascontiguousarray(ky),
        np.ascontiguousarray(beurling),
        np.ascontiguousarray(cauchy_inverse),
    )


def periodic_beurling_apply(field: ComplexArray) -> ComplexArray:
    """Apply the torus Beurling multiplier with the zero Fourier mode set to 0."""

    values = _validate_field(field)
    frequencies = periodic_frequency_grid(values.shape[1], values.shape[0])
    transformed = np.fft.ifft2(np.fft.fft2(values) * frequencies.beurling)
    return np.ascontiguousarray(transformed)


def periodic_cauchy_inverse(field: ComplexArray) -> ComplexArray:
    """Invert ``d_bar`` on the zero-mean torus subspace."""

    values = _validate_field(field)
    frequencies = periodic_frequency_grid(values.shape[1], values.shape[0])
    transformed = np.fft.ifft2(np.fft.fft2(values) * frequencies.cauchy_inverse)
    return np.ascontiguousarray(transformed)


def periodic_dbar_apply(field: ComplexArray) -> ComplexArray:
    """Apply the spectral ``d_bar`` derivative on the zero-mean torus space."""

    values = _validate_field(field)
    frequencies = periodic_frequency_grid(values.shape[1], values.shape[0])
    symbol = 0.5j * (frequencies.kx + 1j * frequencies.ky)
    return np.ascontiguousarray(np.fft.ifft2(np.fft.fft2(values) * symbol))


def periodic_dz_apply(field: ComplexArray) -> ComplexArray:
    """Apply the spectral ``d_z`` derivative on the zero-mean torus space."""

    values = _validate_field(field)
    frequencies = periodic_frequency_grid(values.shape[1], values.shape[0])
    symbol = 0.5j * (frequencies.kx - 1j * frequencies.ky)
    return np.ascontiguousarray(np.fft.ifft2(np.fft.fft2(values) * symbol))


def periodic_beltrami_fixed_point(
    mu: ComplexArray, *, max_iter: int = 100, tolerance: float = 1e-10
) -> PeriodicFixedPointResult:
    """Solve ``h = mu * (1 + B h)`` by periodic Neumann iteration."""

    coefficients = _validate_field(mu)
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    if max_iter < 1 or tolerance <= 0.0:
        raise ValueError("max_iter and tolerance must be positive")
    h = np.zeros_like(coefficients)
    residual = float("inf")
    for iteration in range(1, max_iter + 1):
        updated = coefficients * (1.0 + periodic_beurling_apply(h))
        residual = float(np.max(np.abs(updated - h)))
        h = updated
        if residual <= tolerance:
            return PeriodicFixedPointResult(h, iteration, residual, True)
    return PeriodicFixedPointResult(h, max_iter, residual, False)


def periodic_lift_derivatives(h: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    """Return ``(f_z, f_bar)`` for ``f(z)=z+C(h)`` with zero periodic mode."""

    correction = periodic_cauchy_inverse(h)
    return 1.0 + periodic_dz_apply(correction), periodic_dbar_apply(correction)


def periodic_beltrami_map_from_h(h: ComplexArray) -> tuple[ComplexArray, complex, complex]:
    """Construct the quasi-periodic lift while retaining ``mean(h)``.

    Writing ``b=mean(h)`` and ``u=C(h-b)`` gives
    ``f(z)=z+b*conj(z)+u(z)``, so ``f_bar=h`` and the two periods are
    ``1+b`` and ``i(1-b)``.  This is the map-level zero-mode-safe lift.
    """

    values = _validate_field(h)
    ny, nx = values.shape
    mean_mode = complex(np.mean(values))
    correction = periodic_cauchy_inverse(values - mean_mode)
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    z = xx + 1j * yy
    mapped = z + mean_mode * np.conjugate(z) + correction
    return np.ascontiguousarray(mapped), 1.0 + mean_mode, 1j * (1.0 - mean_mode)


def torus_affine_derivatives(mu: complex) -> tuple[complex, complex]:
    """Return derivatives of the affine lift ``f(z)=z+mu*conj(z)``."""

    coefficient = complex(mu)
    if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
        raise ValueError("mu must be finite")
    if abs(coefficient) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    return 1.0 + 0.0j, coefficient


def torus_affine_periods(mu: complex) -> tuple[complex, complex]:
    """Return the two quasi-periods of the constant-coefficient affine lift."""

    _, coefficient = torus_affine_derivatives(mu)
    return 1.0 + coefficient, 1j * (1.0 - coefficient)


def torus_periods_to_affine_coefficients(
    period_x: complex, period_y: complex
) -> tuple[complex, complex, complex]:
    """Recover ``(a,b,mu=b/a)`` from two complex affine periods."""

    px, py = complex(period_x), complex(period_y)
    if not np.isfinite(px.real + px.imag + py.real + py.imag):
        raise ValueError("periods must be finite")
    a = 0.5 * (px - 1j * py)
    b = 0.5 * (px + 1j * py)
    if abs(a) <= 1e-14:
        raise ValueError("periods do not define a nondegenerate affine lift")
    mu = b / a
    if abs(mu) >= 1.0:
        raise ValueError("periods define a non-quasiconformal affine lift")
    return a, b, mu


def torus_affine_grid(
    mu: complex, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a periodic affine lift and its parallelogram triangulation."""

    coefficient = complex(mu)
    if abs(coefficient) >= 1.0 or not np.isfinite(coefficient.real + coefficient.imag):
        raise ValueError("mu must be finite and lie strictly inside the unit disk")
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least two")
    # Use a half-open parameter grid; the final row/column are identified by
    # the two transformed periods rather than duplicated in the vertex array.
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    z = xx + 1j * yy
    mapped = z + coefficient * np.conjugate(z)
    vertices = np.stack((mapped.real.ravel(), mapped.imag.ravel()), axis=1)
    faces: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            a = j * nx + i
            b = j * nx + ((i + 1) % nx)
            c = ((j + 1) % ny) * nx + i
            d = ((j + 1) % ny) * nx + ((i + 1) % nx)
            faces.extend(((a, b, d), (a, d, c)))
    return np.ascontiguousarray(vertices), np.asarray(faces, dtype=np.int64)


def zero_padded_beurling_apply(
    field: ComplexArray, *, padding_factor: int = 2
) -> ComplexArray:
    """Approximate the whole-plane operator by centered zero-padding.

    The input is interpreted as a compactly supported function on a bounded
    window. Padding suppresses the periodic image interaction but does not turn
    the result into an exact bounded-domain boundary-value solver.
    """

    return _zero_padded_multiplier(field, padding_factor, "beurling")


def zero_padded_cauchy_inverse(
    field: ComplexArray, *, padding_factor: int = 2
) -> ComplexArray:
    """Apply a centered zero-padded Cauchy inverse approximation."""

    return _zero_padded_multiplier(field, padding_factor, "cauchy_inverse")


def zero_padded_dbar_apply(
    field: ComplexArray, *, padding_factor: int = 2
) -> ComplexArray:
    """Apply a centered zero-padded ``d_bar`` spectral derivative."""

    return _zero_padded_multiplier(field, padding_factor, "dbar")


def zero_padded_dz_apply(
    field: ComplexArray, *, padding_factor: int = 2
) -> ComplexArray:
    """Apply a centered zero-padded ``d_z`` spectral derivative."""

    return _zero_padded_multiplier(field, padding_factor, "dz")


def _zero_padded_multiplier(
    field: ComplexArray, padding_factor: int, kind: str
) -> ComplexArray:
    values = _validate_field(field)
    if not isinstance(padding_factor, (int, np.integer)) or padding_factor < 2:
        raise ValueError("padding_factor must be an integer at least two")
    ny, nx = values.shape
    padded_shape = (int(padding_factor) * ny, int(padding_factor) * nx)
    padded = np.zeros(padded_shape, dtype=np.complex128)
    offset_y = (padded_shape[0] - ny) // 2
    offset_x = (padded_shape[1] - nx) // 2
    padded[offset_y : offset_y + ny, offset_x : offset_x + nx] = values
    frequencies = periodic_frequency_grid(padded_shape[1], padded_shape[0])
    if kind == "beurling":
        multiplier = frequencies.beurling
    elif kind == "cauchy_inverse":
        # ``periodic_frequency_grid`` receives the original sample spacing
        # (``d=1/n``), so the padded array already has the correct physical
        # frequencies on a period ``padding_factor`` window.  No extra scale
        # factor belongs here.
        multiplier = frequencies.cauchy_inverse
    elif kind == "dbar":
        multiplier = 0.5j * (frequencies.kx + 1j * frequencies.ky)
    elif kind == "dz":
        multiplier = 0.5j * (frequencies.kx - 1j * frequencies.ky)
    else:
        raise ValueError(f"unknown zero-padded multiplier: {kind}")
    transformed = np.fft.ifft2(np.fft.fft2(padded) * multiplier)
    return np.ascontiguousarray(
        transformed[offset_y : offset_y + ny, offset_x : offset_x + nx]
    )


def _validate_field(field: ComplexArray) -> ComplexArray:
    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("field must be a two-dimensional array with both axes >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must be finite")
    return values
