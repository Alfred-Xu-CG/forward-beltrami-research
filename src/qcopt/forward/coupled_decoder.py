"""Composable hard-bijective monotone and shear decoder."""

from __future__ import annotations

import numpy as np

from .monotone import monotone_axis_inverse, monotone_axis_map, separable_monotone_map
from .shear import apply_affine_shear, invert_affine_shear


def coupled_monotone_shear_map(
    points: np.ndarray,
    x_increments: np.ndarray,
    y_increments: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Apply a separable positive map followed by a determinant-one shear."""

    return apply_affine_shear(
        separable_monotone_map(points, x_increments, y_increments), alpha, beta
    )


def coupled_monotone_shear_inverse(
    points: np.ndarray,
    x_increments: np.ndarray,
    y_increments: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Invert :func:`coupled_monotone_shear_map` exactly up to interpolation."""

    values = invert_affine_shear(points, alpha, beta)
    return np.column_stack(
        (
            monotone_axis_inverse(values[:, 0], x_increments),
            monotone_axis_inverse(values[:, 1], y_increments),
        )
    )


def triangular_spatial_monotone_map(
    points: np.ndarray, x_increments: np.ndarray, y_increments: np.ndarray
) -> np.ndarray:
    """Apply a spatially varying triangular monotone rectangle map."""

    values, x_inc, y_inc = _validate_triangular_inputs(points, x_increments, y_increments)
    x = values[:, 0]
    mapped_x = monotone_axis_map(x, x_inc)
    controls = np.linspace(0.0, 1.0, y_inc.shape[0])
    interpolated = np.column_stack([np.interp(x, controls, y_inc[:, j]) for j in range(y_inc.shape[1])])
    mapped_y = _rowwise_monotone_map(values[:, 1], interpolated)
    return np.column_stack((mapped_x, mapped_y))


def triangular_spatial_monotone_inverse(
    points: np.ndarray, x_increments: np.ndarray, y_increments: np.ndarray
) -> np.ndarray:
    """Invert :func:`triangular_spatial_monotone_map` exactly piecewise-linearly."""

    values, x_inc, y_inc = _validate_triangular_inputs(points, x_increments, y_increments)
    original_x = monotone_axis_inverse(values[:, 0], x_inc)
    controls = np.linspace(0.0, 1.0, y_inc.shape[0])
    interpolated = np.column_stack([np.interp(original_x, controls, y_inc[:, j]) for j in range(y_inc.shape[1])])
    original_y = _rowwise_monotone_inverse(values[:, 1], interpolated)
    return np.column_stack((original_x, original_y))


def _rowwise_monotone_map(values: np.ndarray, increments: np.ndarray) -> np.ndarray:
    n_intervals = increments.shape[1]
    normalized = increments / np.sum(increments, axis=1, keepdims=True)
    cumulative = np.concatenate((np.zeros((len(values), 1)), np.cumsum(normalized, axis=1)), axis=1)
    coordinate = np.clip(values, 0.0, 1.0) * n_intervals
    index = np.minimum(np.floor(coordinate).astype(np.int64), n_intervals - 1)
    fraction = coordinate - index
    row = np.arange(len(values))
    return cumulative[row, index] + fraction * normalized[row, index]


def _rowwise_monotone_inverse(values: np.ndarray, increments: np.ndarray) -> np.ndarray:
    n_intervals = increments.shape[1]
    normalized = increments / np.sum(increments, axis=1, keepdims=True)
    cumulative = np.concatenate((np.zeros((len(values), 1)), np.cumsum(normalized, axis=1)), axis=1)
    clipped = np.clip(values, 0.0, 1.0)
    index = np.minimum(np.sum(clipped[:, None] >= cumulative[:, 1:], axis=1).astype(np.int64), n_intervals - 1)
    row = np.arange(len(values))
    fraction = (clipped - cumulative[row, index]) / normalized[row, index]
    return (index + fraction) / n_intervals


def _validate_triangular_inputs(
    points: np.ndarray, x_increments: np.ndarray, y_increments: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    x_inc = np.asarray(x_increments, dtype=np.float64)
    y_inc = np.asarray(y_increments, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must be finite with shape (n,2)")
    if x_inc.ndim != 1 or y_inc.ndim != 2 or y_inc.shape[0] < 2 or y_inc.shape[1] < 1:
        raise ValueError("increments have invalid shapes")
    if np.any(x_inc <= 0.0) or np.any(y_inc <= 0.0) or not np.all(np.isfinite(x_inc)) or not np.all(np.isfinite(y_inc)):
        raise ValueError("all increments must be strictly positive and finite")
    if np.min(values) < -1e-12 or np.max(values) > 1.0 + 1e-12:
        raise ValueError("points must lie in [0,1]^2")
    return values, x_inc, y_inc
