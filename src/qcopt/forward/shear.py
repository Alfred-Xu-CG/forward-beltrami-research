"""Explicit globally invertible affine shear layers for a mesh-native flow."""

from __future__ import annotations

import numpy as np


def affine_shear_matrix(alpha: float, beta: float) -> np.ndarray:
    """Return ``[[1, alpha], [beta, 1+alpha*beta]]`` with determinant one."""

    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("shear parameters must be finite")
    return np.asarray([[1.0, alpha], [beta, 1.0 + alpha * beta]], dtype=np.float64)


def apply_affine_shear(points: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must have shape (n, 2) and be finite")
    return np.ascontiguousarray(values @ affine_shear_matrix(alpha, beta).T)


def invert_affine_shear(points: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must have shape (n, 2) and be finite")
    matrix = affine_shear_matrix(alpha, beta)
    return np.ascontiguousarray(values @ np.linalg.inv(matrix).T)


def affine_shear_log_abs_det(alpha: float, beta: float) -> float:
    """Analytic log absolute Jacobian determinant (always zero)."""

    affine_shear_matrix(alpha, beta)
    return 0.0
