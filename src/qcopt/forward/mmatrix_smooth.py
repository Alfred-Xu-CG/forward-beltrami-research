"""Spatially regularized positive wide-stencil conductance selection.

The local tensor equation ``A w = a`` is underdetermined once a wide
direction dictionary has more than three directions.  This module keeps the
nonnegative cone constraint but selects among exact local candidates using a
deterministic raster continuation penalty.  It is a diagnostic/canonicalizer,
not a global optimality theorem: the scan order is intentionally exposed.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def smooth_positive_directional_conductances(
    tensors: np.ndarray,
    directions: np.ndarray,
    *,
    tolerance: float = 1e-10,
    neighbor_weight: float = 1.0,
    norm_weight: float = 1e-8,
    reverse_rows: bool = False,
    reverse_columns: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose nonnegative local cone fits with a spatial continuity penalty.

    ``tensors`` has shape ``(...,2,2)`` and is interpreted as a raster in its
    first two dimensions.  For each pixel all feasible one-, two-, and
    three-direction active sets are enumerated.  The selected candidate
    minimizes squared distance to the average of the already selected left
    and upper neighbors, plus a small norm tie-break.  The returned residual
    is the tensor reconstruction residual at each pixel.

    The method is deliberately explicit about its scan-order dependence.  It
    is useful for diagnosing whether local active-set nonuniqueness, rather
    than the positive cone itself, is causing a consistency failure.
    """

    matrix = np.asarray(tensors, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if matrix.ndim < 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError("tensors must have at least two raster dimensions and shape (...,2,2)")
    if dirs.ndim != 2 or dirs.shape[1] != 2 or dirs.shape[0] < 1:
        raise ValueError("directions must have shape (m,2)")
    if tolerance <= 0.0 or neighbor_weight < 0.0 or norm_weight < 0.0:
        raise ValueError("tolerance must be positive and weights nonnegative")
    normalized = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    dictionary = np.vstack(
        (normalized[:, 0] ** 2, normalized[:, 0] * normalized[:, 1], normalized[:, 1] ** 2)
    )
    target = np.stack((matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]), axis=-1)
    shape = target.shape[:-1]
    if len(shape) < 2:
        raise ValueError("tensors must have raster dimensions")
    height, width = shape[0], shape[1]
    count = dirs.shape[0]
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for active_count in range(1, min(3, count) + 1):
        for active in combinations(range(count), active_count):
            active_dictionary = dictionary[:, active]
            if np.linalg.matrix_rank(active_dictionary) < active_count:
                continue
            candidates.append((np.asarray(active, dtype=np.int64), np.linalg.pinv(active_dictionary)))
    values = np.zeros(shape + (count,), dtype=np.float64)
    residual = np.full(shape, np.inf, dtype=np.float64)

    row_indices = range(height - 1, -1, -1) if reverse_rows else range(height)
    column_indices = range(width - 1, -1, -1) if reverse_columns else range(width)
    row_step = -1 if reverse_rows else 1
    column_step = -1 if reverse_columns else 1
    for row in row_indices:
        for column in column_indices:
            reference_parts: list[np.ndarray] = []
            previous_column = column - column_step
            previous_row = row - row_step
            if 0 <= previous_column < width:
                reference_parts.append(values[row, previous_column])
            if 0 <= previous_row < height:
                reference_parts.append(values[previous_row, column])
            if reference_parts:
                reference = np.mean(reference_parts, axis=0)
            else:
                reference = np.zeros(count, dtype=np.float64)
            rhs = target[row, column]
            best_score = np.inf
            best_value = None
            best_residual = np.inf
            for active, pseudoinverse in candidates:
                local = pseudoinverse @ rhs
                if np.any(local < -tolerance):
                    continue
                local = np.maximum(local, 0.0)
                local_residual = float(np.linalg.norm(dictionary[:, active] @ local - rhs))
                if local_residual > tolerance:
                    continue
                candidate = np.zeros(count, dtype=np.float64)
                candidate[active] = local
                score = neighbor_weight * float(np.sum((candidate - reference) ** 2))
                score += norm_weight * float(np.sum(candidate * candidate))
                if score < best_score:
                    best_score = score
                    best_value = candidate
                    best_residual = local_residual
            if best_value is None:
                # Keep a finite, explicit fallback for tensors outside the
                # positive cone; callers must inspect the residual certificate.
                all_candidates = []
                for active, pseudoinverse in candidates:
                    local = np.maximum(pseudoinverse @ rhs, 0.0)
                    candidate = np.zeros(count, dtype=np.float64)
                    candidate[active] = local
                    all_candidates.append(candidate)
                if not all_candidates:
                    raise ValueError("direction dictionary has no independent active set")
                best_value = min(all_candidates, key=lambda item: float(np.linalg.norm(dictionary @ item - rhs)))
                best_residual = float(np.linalg.norm(dictionary @ best_value - rhs))
            values[row, column] = best_value
            residual[row, column] = best_residual
    return np.ascontiguousarray(values), np.ascontiguousarray(residual)
