"""Particle-mesh gridding baseline for Beurling transforms on scattered points."""

from __future__ import annotations

import numpy as np

from .beurling import periodic_beurling_apply


def particle_mesh_beurling_apply(
    points: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    grid_shape: tuple[int, int] = (128, 128),
    deconvolve: bool = False,
    kernel: str = "cic",
    target_points: np.ndarray | None = None,
) -> np.ndarray:
    """Approximate a periodic Beurling integral using compact-kernel gridding.

    The method is a practical NUFFT-like control rather than an exact
    unstructured discretization: particles are deposited with CIC or cubic
    B-spline weights, a uniform FFT multiplier is applied, and the result is
    gathered with the same stencil. It requires points in the unit torus and
    positive quadrature weights.
    """

    z = np.asarray(points, dtype=np.complex128)
    targets = z if target_points is None else np.asarray(target_points, dtype=np.complex128)
    h = np.asarray(values, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    if z.ndim != 1 or targets.ndim != 1 or h.shape != z.shape or area.shape != z.shape:
        raise ValueError("points, values, and weights must have matching 1D shapes")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(targets)) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(area)):
        raise ValueError("inputs must be finite")
    if np.any(area <= 0.0) or np.any(z.real < 0.0) or np.any(z.real >= 1.0) or np.any(z.imag < 0.0) or np.any(z.imag >= 1.0):
        raise ValueError("points must lie in [0,1)^2 and weights must be positive")
    if np.any(targets.real < 0.0) or np.any(targets.real >= 1.0) or np.any(targets.imag < 0.0) or np.any(targets.imag >= 1.0):
        raise ValueError("target_points must lie in [0,1)^2")
    ny, nx = grid_shape
    if nx < 2 or ny < 2:
        raise ValueError("grid dimensions must be at least two")
    if kernel not in {"cic", "cubic"}:
        raise ValueError("kernel must be 'cic' or 'cubic'")
    grid = np.zeros((ny, nx), dtype=np.complex128)
    cell_area = 1.0 / (nx * ny)
    gx = z.real * nx
    gy = z.imag * ny
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    tx, ty = gx - ix, gy - iy
    deposited = h * area / cell_area
    if kernel == "cic":
        x_stencil = ((0, 1.0 - tx), (1, tx))
        y_stencil = ((0, 1.0 - ty), (1, ty))
        power = 4
    else:
        x_stencil = (
            (-1, (1.0 - tx) ** 3 / 6.0),
            (0, (3.0 * tx**3 - 6.0 * tx**2 + 4.0) / 6.0),
            (1, (-3.0 * tx**3 + 3.0 * tx**2 + 3.0 * tx + 1.0) / 6.0),
            (2, tx**3 / 6.0),
        )
        y_stencil = (
            (-1, (1.0 - ty) ** 3 / 6.0),
            (0, (3.0 * ty**3 - 6.0 * ty**2 + 4.0) / 6.0),
            (1, (-3.0 * ty**3 + 3.0 * ty**2 + 3.0 * ty + 1.0) / 6.0),
            (2, ty**3 / 6.0),
        )
        power = 8
    for ox, wx in x_stencil:
        for oy, wy in y_stencil:
            np.add.at(grid, ((iy + oy) % ny, (ix + ox) % nx), deposited * wx * wy)
    transformed = periodic_beurling_apply(grid)
    if deconvolve:
        # CIC scatter and gather each have the tensor-product linear B-spline
        # window ``sinc^2``.  Compensate their combined ``sinc^4`` response
        # away from the Nyquist edge; the floor prevents noise blow-up there.
        ky = np.fft.fftfreq(ny) * ny
        kx = np.fft.fftfreq(nx) * nx
        window = np.sinc(ky[:, None] / ny) ** power * np.sinc(kx[None, :] / nx) ** power
        transformed_hat = np.fft.fft2(transformed)
        transformed = np.fft.ifft2(
            transformed_hat / np.maximum(window, 0.08)
        )
    target_gx = targets.real * nx
    target_gy = targets.imag * ny
    target_ix = np.floor(target_gx).astype(np.int64)
    target_iy = np.floor(target_gy).astype(np.int64)
    target_tx, target_ty = target_gx - target_ix, target_gy - target_iy
    if kernel == "cic":
        target_x_stencil = ((0, 1.0 - target_tx), (1, target_tx))
        target_y_stencil = ((0, 1.0 - target_ty), (1, target_ty))
    else:
        target_x_stencil = (
            (-1, (1.0 - target_tx) ** 3 / 6.0),
            (0, (3.0 * target_tx**3 - 6.0 * target_tx**2 + 4.0) / 6.0),
            (1, (-3.0 * target_tx**3 + 3.0 * target_tx**2 + 3.0 * target_tx + 1.0) / 6.0),
            (2, target_tx**3 / 6.0),
        )
        target_y_stencil = (
            (-1, (1.0 - target_ty) ** 3 / 6.0),
            (0, (3.0 * target_ty**3 - 6.0 * target_ty**2 + 4.0) / 6.0),
            (1, (-3.0 * target_ty**3 + 3.0 * target_ty**2 + 3.0 * target_ty + 1.0) / 6.0),
            (2, target_ty**3 / 6.0),
        )
    gathered = np.zeros(targets.shape, dtype=np.complex128)
    for ox, wx in target_x_stencil:
        for oy, wy in target_y_stencil:
            gathered += transformed[(target_iy + oy) % ny, (target_ix + ox) % nx] * wx * wy
    return np.ascontiguousarray(gathered)
