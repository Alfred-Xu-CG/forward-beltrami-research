"""Closest-point barycentric location on a triangle surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .surface_atlas import SurfacePoint
from .surface_mesh import SurfaceMesh


@dataclass(frozen=True)
class LocateResult:
    point: SurfacePoint
    distance: float


@dataclass(frozen=True)
class LocateBatch:
    points: tuple[SurfacePoint, ...]
    distances: np.ndarray
    maximum_distance: float


class SurfaceLocator:
    """Reusable exact nearest-triangle locator with KD-tree broad phases.

    Nearest vertices and centroids give a fast initial upper bound.  A second
    centroid-ball query uses the maximum triangle bounding radius: any triangle
    outside that ball has a provable distance lower bound greater than the
    incumbent and therefore cannot be closest.  This avoids silently missing
    long, skinny triangles whose vertices and centroid are all far away.
    """

    def __init__(self, mesh: SurfaceMesh, candidate_count: int = 16):
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        self.mesh = mesh
        self.candidate_count = int(candidate_count)
        self._vertex_tree = cKDTree(mesh.vertices)
        self._face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
        self._face_radii = np.max(
            np.linalg.norm(
                mesh.vertices[mesh.faces] - self._face_centroids[:, None, :],
                axis=2,
            ),
            axis=1,
        )
        self._maximum_face_radius = float(np.max(self._face_radii))
        self._face_tree = cKDTree(self._face_centroids)

    def locate(self, query: np.ndarray) -> LocateResult:
        query = np.asarray(query, dtype=np.float64)
        if query.shape != (3,) or not np.all(np.isfinite(query)):
            raise ValueError("query must be a finite length-3 vector")
        vertex_count = min(self.candidate_count, self.mesh.n_vertices)
        _, nearest_vertices = self._vertex_tree.query(query, k=vertex_count)
        nearest_vertices = np.atleast_1d(nearest_vertices)
        candidates: set[int] = set()
        for vertex in nearest_vertices:
            candidates.update(int(face) for face in self.mesh.vertex_faces[int(vertex)])
        face_count = min(self.candidate_count, self.mesh.n_faces)
        _, nearest_faces = self._face_tree.query(query, k=face_count)
        candidates.update(int(face) for face in np.atleast_1d(nearest_faces))

        best_face = -1
        best_barycentric: np.ndarray | None = None
        best_squared_distance = float("inf")
        def consider(face: int) -> None:
            nonlocal best_face, best_barycentric, best_squared_distance
            triangle = self.mesh.vertices[self.mesh.faces[face]]
            closest, barycentric = _closest_point_triangle(query, triangle)
            squared_distance = float(np.sum((closest - query) ** 2))
            if squared_distance < best_squared_distance:
                best_face = face
                best_barycentric = barycentric
                best_squared_distance = squared_distance

        for face in sorted(candidates):
            consider(face)
        if best_barycentric is None:
            raise RuntimeError("surface locator found no candidate triangle")

        # A triangle lies inside the sphere centered at its centroid with its
        # precomputed radius.  Thus a triangle that beats the incumbent must
        # have centroid distance <= incumbent distance + its radius, which is
        # bounded above by the global maximum face radius.
        incumbent = float(np.sqrt(best_squared_distance))
        broad_radius = np.nextafter(
            incumbent + self._maximum_face_radius, float("inf")
        )
        exact_candidates = self._face_tree.query_ball_point(query, broad_radius)
        for face in sorted(set(int(value) for value in exact_candidates) - candidates):
            consider(face)
        return LocateResult(
            SurfacePoint(best_face, best_barycentric),
            float(np.sqrt(best_squared_distance)),
        )

    def locate_many(self, queries: np.ndarray) -> LocateBatch:
        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries must have shape (n, 3)")
        results = [self.locate(query) for query in queries]
        distances = np.asarray([result.distance for result in results])
        distances.setflags(write=False)
        return LocateBatch(
            tuple(result.point for result in results),
            distances,
            float(np.max(distances, initial=0.0)),
        )


def _closest_point_triangle(
    point: np.ndarray, triangle: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ericson's Voronoi-region closest point with barycentric output."""

    a, b, c = triangle
    ab, ac = b - a, c - a
    ap = point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])

    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        parameter = d1 / (d1 - d3)
        barycentric = np.array([1.0 - parameter, parameter, 0.0])
        return barycentric @ triangle, barycentric

    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        parameter = d2 / (d2 - d6)
        barycentric = np.array([1.0 - parameter, 0.0, parameter])
        return barycentric @ triangle, barycentric

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        barycentric = np.array([0.0, 1.0 - parameter, parameter])
        return barycentric @ triangle, barycentric

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    barycentric = np.array([1.0 - v - w, v, w])
    return barycentric @ triangle, barycentric
