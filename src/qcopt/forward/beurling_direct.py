"""Reference O(N^2) quadrature for the whole-plane Beurling kernel.

This is deliberately a diagnostic, not a production singular-integral
discretization: the principal-value self term is omitted and supplied
quadrature weights are required.  It makes the cost of leaving a uniform FFT
grid explicit and provides a reference target for a future NUFFT/FMM backend.
"""

from __future__ import annotations

import numpy as np


def direct_beurling_apply(
    points: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    block_size: int = 512,
    target_points: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``-1/pi int values(w)/(z-w)^2 dw`` by blocked quadrature.

    If ``target_points`` is omitted, the operation is interpreted as a
    principal-value diagnostic and the coincident source term is omitted.  A
    separate target set avoids that singular self-term and is useful for
    independent particle-mesh cross-checks.
    """

    z = np.asarray(points, dtype=np.complex128)
    targets = z if target_points is None else np.asarray(target_points, dtype=np.complex128)
    h = np.asarray(values, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    if z.ndim != 1 or targets.ndim != 1 or h.shape != z.shape or area.shape != z.shape:
        raise ValueError("points, values, and weights must have the same 1D shape")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(targets)) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(area)):
        raise ValueError("inputs must be finite")
    if np.any(area <= 0.0) or block_size < 1:
        raise ValueError("weights must be positive and block_size must be positive")
    result = np.zeros(targets.shape, dtype=np.complex128)
    weighted = h * area
    for start in range(0, len(targets), block_size):
        stop = min(start + block_size, len(targets))
        delta = targets[start:stop, None] - z[None, :]
        if target_points is None:
            rows = np.arange(stop - start)
            source_indices = start + rows
            valid = source_indices < len(z)
            delta[rows[valid], source_indices[valid]] = np.inf
        kernel = np.zeros_like(delta)
        nonzero = np.isfinite(delta)
        kernel[nonzero] = -1.0 / (np.pi * delta[nonzero] ** 2)
        result[start:stop] = kernel @ weighted
    return np.ascontiguousarray(result)
