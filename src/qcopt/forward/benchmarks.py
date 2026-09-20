"""Manufactured maps used to compare all forward Beltrami routes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..beltrami import face_beltrami
from ..mesh import TriMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def affine_map(
    points: FloatArray, matrix: FloatArray, offset: FloatArray
) -> FloatArray:
    """Evaluate an affine map on an ``(n, 2)`` point array."""

    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("matrix and offset must have shapes (2, 2) and (2,)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(matrix)) or not np.all(
        np.isfinite(offset)
    ):
        raise ValueError("affine-map inputs must be finite")
    return np.ascontiguousarray(points @ matrix.T + offset)


def smooth_twist_map(points: FloatArray, amplitude: float = 0.08) -> FloatArray:
    """Apply a smooth shear with unit analytical Jacobian determinant."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    result = points.copy()
    result[:, 0] += float(amplitude) * np.sin(2.0 * np.pi * points[:, 1])
    return np.ascontiguousarray(result)


def manufactured_mu(mesh: TriMesh, uv: FloatArray) -> ComplexArray:
    """Return the exact facewise coefficient of a supplied manufactured map."""

    return np.ascontiguousarray(face_beltrami(mesh, uv))
