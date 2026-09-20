"""Periodic P1 triangular-mesh block symbols for d_z and d_bar."""

from __future__ import annotations

import numpy as np


def p1_face_derivative_symbols(
    kx: int, ky: int, nx: int, ny: int, *, hx: float | None = None, hy: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A,B`` (two-face symbols) for ``d_z,d_bar`` on a regular P1 mesh.

    Faces are ordered as ``(v00,v10,v11)`` and ``(v00,v11,v01)``.  A vertex
    Fourier mode has phase ``X=exp(2*pi*i*kx/nx)`` and ``Y=exp(2*pi*i*ky/ny)``.
    The returned vectors satisfy ``d_z u=A*u`` and ``d_bar u=B*u`` on the two
    face orientations.  The constant mode has both symbols exactly zero.
    """

    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least two")
    hx = 1.0 / nx if hx is None else float(hx)
    hy = 1.0 / ny if hy is None else float(hy)
    if hx <= 0.0 or hy <= 0.0 or not np.isfinite(hx + hy):
        raise ValueError("mesh spacings must be positive and finite")
    X = np.exp(2j * np.pi * kx / nx)
    Y = np.exp(2j * np.pi * ky / ny)
    fx_plus = (X - 1.0) / hx
    fy_plus = X * (Y - 1.0) / hy
    fx_minus = Y * (X - 1.0) / hx
    fy_minus = (Y - 1.0) / hy
    fx = np.asarray([fx_plus, fx_minus], dtype=np.complex128)
    fy = np.asarray([fy_plus, fy_minus], dtype=np.complex128)
    return 0.5 * (fx - 1j * fy), 0.5 * (fx + 1j * fy)


def p1_beurling_block_symbol(
    kx: int,
    ky: int,
    nx: int,
    ny: int,
    *,
    face_weights: np.ndarray | None = None,
    tolerance: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(A,B,P,S)``-style data as ``(A,B,P,S)`` for one mode.

    ``P`` is the weighted Moore--Penrose inverse of the 2-by-1 ``B`` symbol,
    and ``S=A@P`` is the rank-one P1 block Beurling symbol.  For a derivative
    null mode ``P`` and ``S`` are zero rather than producing a division by a
    tiny number.
    """

    A, B = p1_face_derivative_symbols(kx, ky, nx, ny)
    weights = np.ones(2, dtype=np.float64) if face_weights is None else np.asarray(face_weights, dtype=np.float64)
    if weights.shape != (2,) or np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("face_weights must be two positive finite values")
    denominator = np.vdot(B * weights, B)
    if abs(denominator) <= tolerance:
        P = np.zeros((1, 2), dtype=np.complex128)
    else:
        P = (np.conjugate(B) * weights / denominator)[None, :]
    S = A[:, None] @ P
    return A, B, np.ascontiguousarray(P), np.ascontiguousarray(S)


def p1_block_symbol_grid(
    nx: int, ny: int, *, face_weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Batch ``S`` and the projector ``B@P`` over a periodic Fourier grid."""

    symbols = np.empty((ny, nx, 2, 2), dtype=np.complex128)
    projectors = np.empty_like(symbols)
    for ky in range(ny):
        for kx in range(nx):
            A, B, P, S = p1_beurling_block_symbol(kx, ky, nx, ny, face_weights=face_weights)
            symbols[ky, kx] = S
            projectors[ky, kx] = B[:, None] @ P
    return np.ascontiguousarray(symbols), np.ascontiguousarray(projectors)
