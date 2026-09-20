"""Sphere-native mesh and stereographic Möbius truth maps."""

from __future__ import annotations

import numpy as np


def uv_sphere_mesh(n_lon: int, n_lat: int) -> tuple[np.ndarray, np.ndarray]:
    """Create an outward-oriented UV sphere with single north/south poles."""

    if n_lon < 3 or n_lat < 2:
        raise ValueError("n_lon must be >=3 and n_lat must be >=2")
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for j in range(1, n_lat):
        theta = np.pi * j / n_lat
        for i in range(n_lon):
            phi = 2.0 * np.pi * i / n_lon
            vertices.append(
                (
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                )
            )
    south = len(vertices)
    vertices.append((0.0, 0.0, -1.0))

    def ring(j: int, i: int) -> int:
        return 1 + (j - 1) * n_lon + (i % n_lon)

    faces: list[tuple[int, int, int]] = []
    for i in range(n_lon):
        faces.append((0, ring(1, i), ring(1, i + 1)))
    for j in range(1, n_lat - 1):
        for i in range(n_lon):
            upper, upper_next = ring(j, i), ring(j, i + 1)
            lower, lower_next = ring(j + 1, i), ring(j + 1, i + 1)
            faces.append((upper, lower, lower_next))
            faces.append((upper, lower_next, upper_next))
    for i in range(n_lon):
        faces.append((south, ring(n_lat - 1, i + 1), ring(n_lat - 1, i)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def mobius_scale_sphere(vertices: np.ndarray, *, scale: float) -> np.ndarray:
    """Apply the positive real stereographic Möbius map ``z -> scale*z``."""

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must be a finite (n, 3) array")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    norms = np.linalg.norm(points, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-10:
        raise ValueError("vertices must lie on the unit sphere")
    mapped = np.empty_like(points)
    north = np.abs(1.0 - points[:, 2]) <= 1e-14
    mapped[north] = points[north]
    regular = ~north
    z = (points[regular, 0] + 1j * points[regular, 1]) / (1.0 - points[regular, 2])
    w = scale * z
    denominator = 1.0 + np.abs(w) ** 2
    mapped[regular, 0] = 2.0 * w.real / denominator
    mapped[regular, 1] = 2.0 * w.imag / denominator
    mapped[regular, 2] = (np.abs(w) ** 2 - 1.0) / denominator
    return np.ascontiguousarray(mapped)


def mobius_scale_velocity_sphere(vertices: np.ndarray, *, scale: float) -> np.ndarray:
    """Analytic ``d/d(log scale)`` velocity of the stereographic Möbius path."""

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("vertices must be a finite (n, 3) array")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    norms = np.linalg.norm(points, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-10:
        raise ValueError("vertices must lie on the unit sphere")
    velocity = np.zeros_like(points)
    north = np.abs(1.0 - points[:, 2]) <= 1e-14
    regular = ~north
    z = (points[regular, 0] + 1j * points[regular, 1]) / (1.0 - points[regular, 2])
    w = scale * z
    radius = np.abs(w) ** 2
    denominator = 1.0 + radius
    # d/d(log s) of w is w; hence d radius / d(log s)=2 radius.
    velocity[regular, 0] = 2.0 * w.real / denominator - 4.0 * w.real * radius / denominator**2
    velocity[regular, 1] = 2.0 * w.imag / denominator - 4.0 * w.imag * radius / denominator**2
    velocity[regular, 2] = 4.0 * radius / denominator**2
    return np.ascontiguousarray(velocity)


def sphere_face_orientation(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return outward signed area proxy for every spherical face."""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    p0, p1, p2 = (points[triangles[:, i]] for i in range(3))
    cross = np.cross(p1 - p0, p2 - p0)
    centroid = p0 + p1 + p2
    return np.einsum("ij,ij->i", cross, centroid)
