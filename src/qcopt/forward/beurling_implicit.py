"""Implicit VJP for the periodic linear Beltrami fixed-point equation."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .beurling import periodic_beurling_apply, periodic_frequency_grid
from .beurling_gmres import periodic_beltrami_gmres


def torus_affine_period_vjp(
    grad_period_x: complex, grad_period_y: complex
) -> complex:
    """Return the real-inner-product VJP of affine periods wrt constant ``mu``.

    For ``p=1+mu`` and ``q=i(1-mu)``, a real loss with complex covectors
    ``g_p,g_q`` has ``dL = Re(conj(g_p)dp + conj(g_q)dq)`` and therefore
    ``dL = Re(conj(g_p+i*g_q)dmu)``. This makes the zero-mode/period channel
    explicit instead of silently dropping it in the periodic inverse.
    """

    gx = complex(grad_period_x)
    gy = complex(grad_period_y)
    if not np.isfinite(gx.real + gx.imag + gy.real + gy.imag):
        raise ValueError("period gradients must be finite")
    return gx + 1j * gy


def torus_affine_map_vjp(
    points: np.ndarray,
    grad_map: np.ndarray,
    *,
    grad_period_x: complex = 0.0j,
    grad_period_y: complex = 0.0j,
) -> complex:
    """VJP of the constant-``mu`` affine torus lift including zero-mode periods.

    For ``f(z)=z+mu*conj(z)``, a complex map cotangent ``g`` contributes
    ``Re(conj(g) conj(z) dmu)`` and therefore the real-inner-product VJP is
    ``sum(g*z)``.  The affine period channel contributes the explicit
    ``torus_affine_period_vjp`` term.  This is the complete constant/affine
    zero-mode coupling; it does not pretend to differentiate a spatially
    varying nonlinear torus solve.
    """

    z = np.asarray(points, dtype=np.complex128)
    g = np.asarray(grad_map, dtype=np.complex128)
    if z.shape != g.shape or z.ndim != 1 or not np.all(np.isfinite(z)) or not np.all(np.isfinite(g)):
        raise ValueError("points and grad_map must be matching finite 1D arrays")
    return complex(np.sum(g * z) + torus_affine_period_vjp(grad_period_x, grad_period_y))


def periodic_beltrami_implicit_vjp(
    mu: np.ndarray,
    h: np.ndarray,
    grad_h: np.ndarray,
    *,
    rtol: float = 1e-10,
    maxiter: int = 200,
) -> tuple[np.ndarray, float, int]:
    """Return the real-inner-product VJP ``dL/dmu`` without unrolling.

    The forward equation is ``(I-M_mu B)h=mu``.  If ``q=1+B h`` and the
    adjoint solve is ``(I-M_mu B)^H lambda=grad_h``, then
    ``dL/dmu=lambda*conj(q)`` for the real Euclidean inner product.
    """

    coefficients = np.asarray(mu, dtype=np.complex128)
    solution = np.asarray(h, dtype=np.complex128)
    cotangent = np.asarray(grad_h, dtype=np.complex128)
    if coefficients.ndim != 2 or solution.shape != coefficients.shape or cotangent.shape != coefficients.shape:
        raise ValueError("mu, h, and grad_h must have matching 2D shapes")
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    shape = coefficients.shape
    size = coefficients.size
    frequencies = periodic_frequency_grid(shape[1], shape[0])
    beurling_adjoint_symbol = np.conjugate(frequencies.beurling)

    def apply_adjoint(vector: np.ndarray) -> np.ndarray:
        values = vector.reshape(shape)
        multiplied = np.conjugate(coefficients) * values
        transformed = np.fft.ifft2(np.fft.fft2(multiplied) * beurling_adjoint_symbol)
        return (values - transformed).ravel()

    operator = LinearOperator((size, size), matvec=apply_adjoint, dtype=np.complex128)
    lambda_flat, info = gmres(
        operator,
        cotangent.ravel(),
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )
    lam = lambda_flat.reshape(shape)
    adjoint_residual = float(np.max(np.abs(apply_adjoint(lambda_flat) - cotangent.ravel())))
    q = 1.0 + periodic_beurling_apply(solution)
    gradient = lam * np.conjugate(q)
    return np.ascontiguousarray(gradient), adjoint_residual, int(info)


def periodic_beltrami_solve_with_vjp(
    mu: np.ndarray, *, rtol: float = 1e-10, maxiter: int = 200
) -> tuple[np.ndarray, callable]:
    """Solve forward and return a closure implementing the implicit VJP."""

    result = periodic_beltrami_gmres(mu, rtol=rtol, maxiter=maxiter)

    def vjp(grad_h: np.ndarray) -> np.ndarray:
        gradient, residual, info = periodic_beltrami_implicit_vjp(
            mu, result.h, grad_h, rtol=rtol, maxiter=maxiter
        )
        if info != 0:
            raise RuntimeError(f"adjoint GMRES failed with info={info}")
        if residual > max(10.0 * rtol, 1e-8):
            raise RuntimeError(f"adjoint residual too large: {residual}")
        return gradient

    return result.h, vjp


def periodic_beltrami_map_h_vjp(h: np.ndarray, grad_map: np.ndarray) -> np.ndarray:
    """VJP of the zero-mode-safe quasi-periodic map with respect to ``h``."""

    values = np.asarray(h, dtype=np.complex128)
    cotangent = np.asarray(grad_map, dtype=np.complex128)
    if values.ndim != 2 or cotangent.shape != values.shape or not np.all(np.isfinite(values)) or not np.all(np.isfinite(cotangent)):
        raise ValueError("h and grad_map must be matching finite 2D arrays")
    ny, nx = values.shape
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    z = xx + 1j * yy
    frequencies = periodic_frequency_grid(nx, ny)
    cauchy_adjoint = np.conjugate(frequencies.cauchy_inverse)
    periodic_part = np.fft.ifft2(np.fft.fft2(cotangent) * cauchy_adjoint)
    # d mean(h) enters as conj(z) d mean(h); under Re(conj(g) delta f)
    # the scalar coefficient is sum(g*z), distributed uniformly over h.
    affine_coefficient = np.sum(cotangent * z)
    return np.ascontiguousarray(periodic_part + affine_coefficient / values.size)
