"""Structured-grid coarse-to-fine map prolongation."""

from __future__ import annotations

import numpy as np


def prolongate_regular_grid(
    coarse_uv: np.ndarray,
    coarse_nx: int,
    coarse_ny: int,
    fine_nx: int,
    fine_ny: int,
) -> np.ndarray:
    """Bilinearly prolongate vertex values between nested regular grids."""

    values = np.asarray(coarse_uv, dtype=np.float64)
    if coarse_nx < 1 or coarse_ny < 1 or fine_nx < coarse_nx or fine_ny < coarse_ny:
        raise ValueError("invalid coarse/fine grid sizes")
    if fine_nx % coarse_nx or fine_ny % coarse_ny:
        raise ValueError("fine grid must be an integer refinement of coarse grid")
    if values.shape != ((coarse_nx + 1) * (coarse_ny + 1), 2):
        raise ValueError("coarse_uv has the wrong shape")
    coarse = values.reshape(coarse_ny + 1, coarse_nx + 1, 2)
    fine = np.empty(((fine_ny + 1) * (fine_nx + 1), 2), dtype=np.float64)
    for j in range(fine_ny + 1):
        source_y = j * coarse_ny / fine_ny
        j0 = min(int(np.floor(source_y)), coarse_ny - 1)
        ty = source_y - j0
        for i in range(fine_nx + 1):
            source_x = i * coarse_nx / fine_nx
            i0 = min(int(np.floor(source_x)), coarse_nx - 1)
            tx = source_x - i0
            q00 = coarse[j0, i0]
            q10 = coarse[j0, i0 + 1]
            q01 = coarse[j0 + 1, i0]
            q11 = coarse[j0 + 1, i0 + 1]
            fine[j * (fine_nx + 1) + i] = (
                (1.0 - tx) * (1.0 - ty) * q00
                + tx * (1.0 - ty) * q10
                + (1.0 - tx) * ty * q01
                + tx * ty * q11
            )
    return np.ascontiguousarray(fine)


def prolongate_positive_increments(
    coarse_increments: np.ndarray, fine_count: int
) -> np.ndarray:
    """Refine positive 1D increments in log-space while preserving positivity."""

    coarse = np.asarray(coarse_increments, dtype=np.float64)
    if coarse.ndim != 1 or len(coarse) < 1 or np.any(coarse <= 0.0) or not np.all(np.isfinite(coarse)):
        raise ValueError("coarse_increments must be finite and strictly positive")
    if not isinstance(fine_count, (int, np.integer)) or fine_count < len(coarse):
        raise ValueError("fine_count must be an integer at least the coarse count")
    coarse_centers = (np.arange(len(coarse), dtype=np.float64) + 0.5) / len(coarse)
    fine_centers = (np.arange(fine_count, dtype=np.float64) + 0.5) / fine_count
    log_values = np.interp(
        fine_centers,
        np.concatenate(([0.0], coarse_centers, [1.0])),
        np.concatenate(([np.log(coarse[0])], np.log(coarse), [np.log(coarse[-1])])),
    )
    refined = np.exp(log_values)
    refined *= np.sum(coarse) / np.sum(refined)
    return np.ascontiguousarray(refined)
