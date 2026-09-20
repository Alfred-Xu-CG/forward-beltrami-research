"""Separable monotone spline-like hard decoder on a rectangular grid."""

from __future__ import annotations

import numpy as np


def monotone_axis_map(values: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """Map normalized coordinates by positive piecewise-linear increments."""

    x = np.asarray(values, dtype=np.float64)
    d = np.asarray(increments, dtype=np.float64)
    if x.ndim != 1 or d.ndim != 1 or len(d) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("values and increments must be finite one-dimensional arrays")
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        raise ValueError("increments must be strictly positive")
    knots = np.linspace(0.0, 1.0, len(d) + 1)
    targets = np.concatenate(([0.0], np.cumsum(d)))
    targets /= targets[-1]
    if np.min(x) < -1e-12 or np.max(x) > 1.0 + 1e-12:
        raise ValueError("values must lie in [0,1]")
    return np.ascontiguousarray(np.interp(x, knots, targets))


def monotone_axis_inverse(values: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """Inverse of :func:`monotone_axis_map` by monotone interpolation."""

    y = np.asarray(values, dtype=np.float64)
    d = np.asarray(increments, dtype=np.float64)
    if y.ndim != 1 or d.ndim != 1 or len(d) < 1 or np.any(d <= 0.0):
        raise ValueError("values and increments must be one-dimensional and positive")
    knots = np.linspace(0.0, 1.0, len(d) + 1)
    targets = np.concatenate(([0.0], np.cumsum(d)))
    targets /= targets[-1]
    if np.min(y) < -1e-12 or np.max(y) > 1.0 + 1e-12:
        raise ValueError("values must lie in [0,1]")
    return np.ascontiguousarray(np.interp(y, targets, knots))


def separable_monotone_map(
    points: np.ndarray, x_increments: np.ndarray, y_increments: np.ndarray
) -> np.ndarray:
    """Apply independent monotone x/y maps, a hard rectangle bijection."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (n,2)")
    return np.column_stack(
        (
            monotone_axis_map(values[:, 0], x_increments),
            monotone_axis_map(values[:, 1], y_increments),
        )
    )
