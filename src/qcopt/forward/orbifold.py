"""Local hard orbifold cone-chart primitives."""

from __future__ import annotations

import numpy as np


def polar_disk_mesh(n_radial: int, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an outward-oriented polar triangulation of the unit disk."""

    if n_radial < 1 or n_theta < 3:
        raise ValueError("n_radial must be positive and n_theta at least three")
    vertices = [(0.0, 0.0)]
    for radial in range(1, n_radial + 1):
        radius = radial / n_radial
        for angular in range(n_theta):
            angle = 2.0 * np.pi * angular / n_theta
            vertices.append((radius * np.cos(angle), radius * np.sin(angle)))

    def ring(radial: int, angular: int) -> int:
        return 1 + (radial - 1) * n_theta + (angular % n_theta)

    faces: list[tuple[int, int, int]] = []
    for angular in range(n_theta):
        faces.append((0, ring(1, angular), ring(1, angular + 1)))
    for radial in range(1, n_radial):
        for angular in range(n_theta):
            a, b = ring(radial, angular), ring(radial, angular + 1)
            c, d = ring(radial + 1, angular), ring(radial + 1, angular + 1)
            faces.extend(((a, c, d), (a, d, b)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def radial_cone_map(points: np.ndarray, angle_scale: float) -> np.ndarray:
    """Map ``r`` to ``r**angle_scale`` on the unit disk."""

    values = _validate_disk_points(points)
    if not np.isfinite(angle_scale) or angle_scale <= 0.0:
        raise ValueError("angle_scale must be positive and finite")
    radius = np.linalg.norm(values, axis=1)
    result = np.zeros_like(values)
    nonzero = radius > 0.0
    factor = np.zeros_like(radius)
    factor[nonzero] = radius[nonzero] ** (angle_scale - 1.0)
    result[nonzero] = values[nonzero] * factor[nonzero, None]
    return np.ascontiguousarray(result)


def radial_cone_inverse(points: np.ndarray, angle_scale: float) -> np.ndarray:
    """Inverse ``r -> r**angle_scale`` cone-chart map."""

    values = _validate_disk_points(points)
    if not np.isfinite(angle_scale) or angle_scale <= 0.0:
        raise ValueError("angle_scale must be positive and finite")
    radius = np.linalg.norm(values, axis=1)
    result = np.zeros_like(values)
    nonzero = radius > 0.0
    factor = np.zeros_like(radius)
    factor[nonzero] = radius[nonzero] ** (1.0 / angle_scale - 1.0)
    result[nonzero] = values[nonzero] * factor[nonzero, None]
    return np.ascontiguousarray(result)


def _validate_disk_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must be finite with shape (n,2)")
    if np.max(np.linalg.norm(values, axis=1), initial=0.0) > 1.0 + 1e-12:
        raise ValueError("points must lie in the unit disk")
    return values
