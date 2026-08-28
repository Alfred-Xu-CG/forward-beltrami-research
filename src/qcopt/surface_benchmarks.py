"""Deterministic high-genus benchmark construction and perturbations."""

from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes

from .surface_atlas import SurfacePoint
from .surface_geometry import ScreenedFieldSmoother
from .surface_mesh import SurfaceMesh


def make_double_torus(resolution: int = 24) -> SurfaceMesh:
    """Build a smoothed implicit double torus with exact measured genus two.

    A fixed sub-grid shift avoids exact isosurface/grid coincidences.  Three
    topology-preserving uniform Laplacian geometry steps regularize marching-
    cubes slivers without changing connectivity; topology is re-audited after
    smoothing.
    """

    if resolution < 18:
        raise ValueError("resolution must be at least 18")
    x = np.linspace(-2.7, 2.7, resolution) + 0.013
    y = np.linspace(-1.7, 1.7, resolution) + 0.017
    z = np.linspace(-0.8, 0.8, resolution) + 0.011
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    major_radius, tube_radius = 1.15, 0.33

    def circle_tube(center_x: float) -> np.ndarray:
        radial = np.sqrt((xx - center_x) ** 2 + yy**2)
        return np.sqrt((radial - major_radius) ** 2 + zz**2) - tube_radius

    field = np.minimum(
        circle_tube(-major_radius),
        circle_tube(major_radius),
    )
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    vertices, faces, _, _ = marching_cubes(field, 0.0, spacing=spacing)
    vertices += np.array([x[0], y[0], z[0]])
    adjacency: list[set[int]] = [set() for _ in range(len(vertices))]
    for a, b, c in faces.tolist():
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    for _ in range(3):
        average = np.vstack(
            [vertices[np.asarray(sorted(neighbors))].mean(axis=0) for neighbors in adjacency]
        )
        vertices = 0.5 * vertices + 0.5 * average
    mesh = SurfaceMesh(vertices, faces.astype(np.int64))
    if mesh.topology.genus != 2:
        raise RuntimeError(
            f"double-torus extraction changed topology: measured genus={mesh.topology.genus}"
        )
    return mesh


def deterministic_face_samples(
    mesh: SurfaceMesh, maximum_samples: int
) -> tuple[SurfacePoint, ...]:
    if maximum_samples < 1:
        raise ValueError("maximum_samples must be positive")
    count = min(maximum_samples, mesh.n_faces)
    face_ids = np.unique(np.linspace(0, mesh.n_faces - 1, count, dtype=np.int64))
    return tuple(
        SurfacePoint(int(face), np.full(3, 1.0 / 3.0)) for face in face_ids
    )


def select_farthest_landmarks(
    mesh: SurfaceMesh,
    samples: tuple[SurfacePoint, ...] | list[SurfacePoint],
    count: int,
) -> np.ndarray:
    """Deterministic Euclidean farthest-point landmarks on surface samples."""

    if count < 1 or count > len(samples):
        raise ValueError("landmark count must lie in [1, len(samples)]")
    positions = np.vstack([point.position(mesh) for point in samples])
    center = positions.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(positions - center, axis=1)))]
    minimum_distance = np.linalg.norm(positions - positions[selected[0]], axis=1)
    while len(selected) < count:
        minimum_distance[np.asarray(selected)] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(positions - positions[next_index], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def random_smooth_tangent_field(
    mesh: SurfaceMesh,
    *,
    seed: int,
    relative_amplitude: float = 2.0,
    smoothing: float = 8.0,
) -> np.ndarray:
    """Generate a reproducible smooth perturbation measured in median edges."""

    if relative_amplitude <= 0.0:
        raise ValueError("relative_amplitude must be positive")
    rng = np.random.default_rng(seed)
    anchor_count = min(12, mesh.n_vertices)
    anchors = rng.choice(mesh.n_vertices, size=anchor_count, replace=False)
    coefficients = rng.normal(size=(anchor_count, 3))
    diagonal = max(float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))), 1e-12)
    radius_squared = (0.35 * diagonal) ** 2
    raw = np.zeros((mesh.n_vertices, 3), dtype=np.float64)
    for anchor, coefficient in zip(anchors, coefficients):
        offset = mesh.vertices - mesh.vertices[anchor]
        weight = np.exp(-np.sum(offset * offset, axis=1) / radius_squared)
        raw += weight[:, None] * coefficient
    # Ambient smoothness is the continuity notion used by the PL face tracer;
    # it projects the field to each active face tangent plane.
    field = ScreenedFieldSmoother(mesh, smoothing).smooth(raw)
    maximum = float(np.max(np.linalg.norm(field, axis=1)))
    if maximum <= 0.0:
        raise RuntimeError("random tangent field unexpectedly vanished")
    field *= relative_amplitude * mesh.median_edge_length / maximum
    return field
