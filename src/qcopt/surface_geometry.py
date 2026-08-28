"""Intrinsic descriptors and cached smooth fields on triangular surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .surface_atlas import SurfacePoint
from .surface_mesh import SurfaceMesh


@dataclass(frozen=True)
class CurvatureDescriptors:
    gaussian: np.ndarray
    mean: np.ndarray
    gaussian_raw: np.ndarray
    mean_raw: np.ndarray


def vertex_areas(mesh: SurfaceMesh) -> np.ndarray:
    areas = np.zeros(mesh.n_vertices, dtype=np.float64)
    for local in range(3):
        np.add.at(areas, mesh.faces[:, local], mesh.face_areas / 3.0)
    return areas


def curvature_descriptors(mesh: SurfaceMesh) -> CurvatureDescriptors:
    """Return raw and robust scale-normalized Gaussian/mean curvature."""

    areas = vertex_areas(mesh)
    if np.any(areas <= 0.0):
        raise ValueError("curvature requires every vertex to have positive area")
    angle_sum = np.zeros(mesh.n_vertices, dtype=np.float64)
    cotangent_weights: dict[tuple[int, int], float] = {}

    positions = mesh.vertices[mesh.faces]
    for local in range(3):
        center = positions[:, local]
        first = positions[:, (local + 1) % 3] - center
        second = positions[:, (local + 2) % 3] - center
        cross_norm = np.linalg.norm(np.cross(first, second), axis=1)
        dots = np.sum(first * second, axis=1)
        angles = np.arctan2(cross_norm, dots)
        np.add.at(angle_sum, mesh.faces[:, local], angles)
        cotangents = dots / cross_norm
        opposite_a = mesh.faces[:, (local + 1) % 3]
        opposite_b = mesh.faces[:, (local + 2) % 3]
        for a, b, value in zip(opposite_a, opposite_b, cotangents):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            cotangent_weights[key] = cotangent_weights.get(key, 0.0) + float(value)

    boundary_vertices: set[int] = set()
    if mesh.topology.boundary_edges:
        for face_id, face in enumerate(mesh.faces):
            for local_edge, neighbor in enumerate(mesh.face_neighbors[face_id]):
                if neighbor < 0:
                    boundary_vertices.add(int(face[(local_edge + 1) % 3]))
                    boundary_vertices.add(int(face[(local_edge + 2) % 3]))
    target_angle = np.full(mesh.n_vertices, 2.0 * np.pi)
    if boundary_vertices:
        target_angle[np.fromiter(boundary_vertices, dtype=np.int64)] = np.pi
    gaussian_raw = (target_angle - angle_sum) / areas

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(mesh.n_vertices)
    for (a, b), combined_cotangent in cotangent_weights.items():
        weight = 0.5 * combined_cotangent
        rows.extend((a, b))
        cols.extend((b, a))
        data.extend((-weight, -weight))
        diagonal[a] += weight
        diagonal[b] += weight
    rows.extend(range(mesh.n_vertices))
    cols.extend(range(mesh.n_vertices))
    data.extend(diagonal.tolist())
    laplacian = sp.csr_matrix((data, (rows, cols)), shape=(mesh.n_vertices,) * 2)
    mean_normal = (laplacian @ mesh.vertices) / (2.0 * areas[:, None])
    mean_raw = 0.5 * np.sum(mean_normal * mesh.vertex_normals, axis=1)

    total_area = float(mesh.face_areas.sum())
    gaussian = _robust_standardize(gaussian_raw * total_area)
    mean = _robust_standardize(mean_raw * np.sqrt(total_area))
    for array in (gaussian, mean, gaussian_raw, mean_raw):
        array.setflags(write=False)
    return CurvatureDescriptors(gaussian, mean, gaussian_raw, mean_raw)


class ScreenedFieldSmoother:
    """Cached `(I + alpha L)^{-1}` graph smoother for scalar/vector fields."""

    def __init__(self, mesh: SurfaceMesh, alpha: float = 1.0):
        if alpha < 0.0:
            raise ValueError("alpha must be nonnegative")
        self.mesh = mesh
        self.alpha = float(alpha)
        laplacian = _uniform_graph_laplacian(mesh)
        matrix = sp.eye(mesh.n_vertices, format="csc") + self.alpha * laplacian.tocsc()
        self._solve = spla.factorized(matrix)
        self.factorization_count = 1

    def smooth(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] != self.mesh.n_vertices or values.ndim not in (1, 2):
            raise ValueError("field must have one scalar or vector per mesh vertex")
        if values.ndim == 1:
            return np.asarray(self._solve(values))
        return np.column_stack([self._solve(values[:, index]) for index in range(values.shape[1])])

    def smooth_tangent(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.shape != (self.mesh.n_vertices, 3):
            raise ValueError("vectors must have shape (mesh.n_vertices, 3)")
        smooth = self.smooth(vectors)
        normal_component = np.sum(smooth * self.mesh.vertex_normals, axis=1)
        return smooth - normal_component[:, None] * self.mesh.vertex_normals


def surface_scalar_gradient(mesh: SurfaceMesh, values: np.ndarray) -> np.ndarray:
    """Piecewise-constant intrinsic gradient of a vertex scalar field."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mesh.n_vertices,):
        raise ValueError("values must have shape (mesh.n_vertices,)")
    return np.einsum(
        "fi,fij->fj",
        values[mesh.faces],
        mesh.barycentric_gradients,
    )


def interpolate_vertex_field(
    mesh: SurfaceMesh,
    values: np.ndarray,
    points: list[SurfacePoint] | tuple[SurfacePoint, ...],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != mesh.n_vertices:
        raise ValueError("field must have one value per mesh vertex")
    result_shape = (len(points),) + values.shape[1:]
    result = np.empty(result_shape, dtype=np.float64)
    for index, point in enumerate(points):
        result[index] = np.tensordot(
            point.barycentric,
            values[mesh.faces[point.face]],
            axes=(0, 0),
        )
    return result


def scatter_point_forces(
    mesh: SurfaceMesh,
    points: list[SurfacePoint] | tuple[SurfacePoint, ...],
    forces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(points), 3):
        raise ValueError("forces must have shape (len(points), 3)")
    scattered = np.zeros((mesh.n_vertices, 3), dtype=np.float64)
    weights = np.zeros(mesh.n_vertices, dtype=np.float64)
    for point, force in zip(points, forces):
        vertices = mesh.faces[point.face]
        np.add.at(scattered, vertices, point.barycentric[:, None] * force)
        np.add.at(weights, vertices, point.barycentric)
    return scattered, weights


def _uniform_graph_laplacian(mesh: SurfaceMesh) -> sp.csr_matrix:
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in mesh.faces.tolist():
        for first, second in ((a, b), (b, c), (c, a)):
            edge_set.add((min(first, second), max(first, second)))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros(mesh.n_vertices, dtype=np.float64)
    for first, second in edge_set:
        length = float(np.linalg.norm(mesh.vertices[first] - mesh.vertices[second]))
        weight = float(
            np.clip((mesh.median_edge_length / length) ** 2, 0.1, 1.0e3)
        )
        rows.extend((first, second))
        cols.extend((second, first))
        data.extend((-weight, -weight))
        degree[first] += weight
        degree[second] += weight
    rows.extend(range(mesh.n_vertices))
    cols.extend(range(mesh.n_vertices))
    data.extend(degree.tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=(mesh.n_vertices,) * 2)


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    deviation = float(np.median(np.abs(values - center)))
    if deviation <= 128.0 * np.finfo(np.float64).eps * max(np.max(np.abs(values)), 1.0):
        return np.zeros_like(values)
    standardized = (values - center) / (1.4826 * deviation)
    return np.clip(standardized, -8.0, 8.0)
