"""Dynamic face-chart transitions for points and tangent steps on PL surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_mesh import SurfaceMesh


@dataclass(frozen=True)
class SurfacePoint:
    """A global point encoded in one local triangle chart."""

    face: int
    barycentric: np.ndarray

    def __post_init__(self) -> None:
        barycentric = np.asarray(self.barycentric, dtype=np.float64)
        if barycentric.shape != (3,) or not np.all(np.isfinite(barycentric)):
            raise ValueError("barycentric coordinates must be a finite length-3 vector")
        if abs(float(barycentric.sum()) - 1.0) > 1e-10:
            raise ValueError("barycentric coordinates must sum to one")
        if np.min(barycentric) < -1e-10:
            raise ValueError("surface point lies outside its face")
        barycentric = np.maximum(barycentric, 0.0)
        barycentric /= barycentric.sum()
        barycentric.setflags(write=False)
        object.__setattr__(self, "face", int(self.face))
        object.__setattr__(self, "barycentric", barycentric)

    def position(self, mesh: SurfaceMesh) -> np.ndarray:
        if self.face < 0 or self.face >= mesh.n_faces:
            raise ValueError("surface point face is out of range")
        return mesh.point_from_barycentric(self.face, self.barycentric)


@dataclass(frozen=True)
class ChartTransition:
    source_face: int
    target_face: int
    edge: tuple[int, int]
    path_fraction: float


@dataclass(frozen=True)
class TraceResult:
    point: SurfacePoint
    transitions: tuple[ChartTransition, ...]
    consumed_fraction: float
    path_length: float


def transport_across_edge(
    mesh: SurfaceMesh,
    face: int,
    local_edge: int,
    vector: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Unfold a tangent vector across one edge into the adjacent face.

    Components parallel to the common edge are preserved.  The component that
    points out of the source triangle becomes the inward component of the
    neighbor.  This is the PL atlas transition induced by unfolding the two
    triangles into a common plane.
    """

    face = int(face)
    local_edge = int(local_edge)
    if face < 0 or face >= mesh.n_faces or local_edge not in (0, 1, 2):
        raise ValueError("invalid face or local edge")
    neighbor = int(mesh.face_neighbors[face, local_edge])
    if neighbor < 0:
        raise RuntimeError("surface trajectory reached a boundary edge")

    source_face = mesh.faces[face]
    edge_vertices = [int(source_face[(local_edge + 1) % 3]), int(source_face[(local_edge + 2) % 3])]
    edge_vertices.sort()
    edge_start, edge_end = mesh.vertices[edge_vertices]
    edge = edge_end - edge_start
    edge_length = float(np.linalg.norm(edge))
    if edge_length <= 0.0:
        raise RuntimeError("surface contains a zero-length transition edge")
    edge_axis = edge / edge_length

    source_opposite = mesh.vertices[int(source_face[local_edge])]
    neighbor_face = mesh.faces[neighbor]
    neighbor_opposite_candidates = [
        int(vertex) for vertex in neighbor_face if int(vertex) not in edge_vertices
    ]
    if len(neighbor_opposite_candidates) != 1:
        raise RuntimeError("invalid manifold adjacency at chart transition")
    neighbor_opposite = mesh.vertices[neighbor_opposite_candidates[0]]

    def inward(opposite: np.ndarray) -> np.ndarray:
        from_edge = opposite - edge_start
        perpendicular = from_edge - float(from_edge @ edge_axis) * edge_axis
        length = float(np.linalg.norm(perpendicular))
        if length <= 0.0:
            raise RuntimeError("degenerate local chart at transition edge")
        return perpendicular / length

    source_inward = inward(source_opposite)
    target_inward = inward(neighbor_opposite)
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("vector must be a finite length-3 vector")
    tangent = vector - float(vector @ mesh.face_normals[face]) * mesh.face_normals[face]
    parallel = float(tangent @ edge_axis)
    source_component = float(tangent @ source_inward)
    transported = parallel * edge_axis - source_component * target_inward
    return neighbor, transported


def trace_tangent_step(
    mesh: SurfaceMesh,
    start: SurfacePoint,
    displacement: np.ndarray,
    *,
    max_crossings: int = 64,
    tolerance: float = 1e-12,
) -> TraceResult:
    """Trace a piecewise-geodesic tangent step through dynamic face charts."""

    if start.face < 0 or start.face >= mesh.n_faces:
        raise ValueError("surface point face is out of range")
    if max_crossings < 0:
        raise ValueError("max_crossings must be nonnegative")
    residual = np.asarray(displacement, dtype=np.float64)
    if residual.shape != (3,) or not np.all(np.isfinite(residual)):
        raise ValueError("displacement must be a finite length-3 vector")
    residual = residual - float(residual @ mesh.face_normals[start.face]) * mesh.face_normals[start.face]
    total_length = float(np.linalg.norm(residual))
    if total_length <= tolerance:
        return TraceResult(start, (), 1.0, 0.0)

    current_face = start.face
    current_barycentric = start.barycentric.copy()
    transitions: list[ChartTransition] = []
    traveled = 0.0

    while True:
        current_position = mesh.point_from_barycentric(current_face, current_barycentric)
        endpoint = current_position + residual
        end_barycentric = mesh.barycentric_from_point(current_face, endpoint)
        if float(np.min(end_barycentric)) >= -tolerance:
            clean = np.maximum(end_barycentric, 0.0)
            clean /= clean.sum()
            traveled += float(np.linalg.norm(residual))
            return TraceResult(
                SurfacePoint(current_face, clean),
                tuple(transitions),
                1.0,
                traveled,
            )

        delta = end_barycentric - current_barycentric
        candidates: list[tuple[float, int]] = []
        for local_edge in range(3):
            if delta[local_edge] < -tolerance:
                fraction = -float(current_barycentric[local_edge]) / float(delta[local_edge])
                if -tolerance <= fraction <= 1.0 + tolerance:
                    candidates.append((max(0.0, fraction), local_edge))
        if not candidates:
            raise RuntimeError("could not resolve the next chart boundary crossing")
        fraction, local_edge = min(candidates, key=lambda item: (item[0], item[1]))
        if len(transitions) >= max_crossings:
            raise RuntimeError("surface trajectory exhausted its crossing budget")

        crossing_barycentric = current_barycentric + fraction * delta
        crossing_barycentric[np.abs(crossing_barycentric) <= 32.0 * tolerance] = 0.0
        crossing_barycentric = np.maximum(crossing_barycentric, 0.0)
        crossing_barycentric /= crossing_barycentric.sum()
        crossing_position = mesh.point_from_barycentric(
            current_face, crossing_barycentric
        )
        traveled += fraction * float(np.linalg.norm(residual))
        residual_after = (1.0 - fraction) * residual
        neighbor, transported = transport_across_edge(
            mesh, current_face, local_edge, residual_after
        )
        edge_vertices = tuple(
            sorted(
                (
                    int(mesh.faces[current_face, (local_edge + 1) % 3]),
                    int(mesh.faces[current_face, (local_edge + 2) % 3]),
                )
            )
        )
        progress = min(1.0, traveled / total_length)
        transitions.append(
            ChartTransition(current_face, neighbor, edge_vertices, progress)
        )
        current_face = neighbor
        current_barycentric = mesh.barycentric_from_point(
            current_face, crossing_position
        )
        current_barycentric[np.abs(current_barycentric) <= 64.0 * tolerance] = 0.0
        current_barycentric = np.maximum(current_barycentric, 0.0)
        current_barycentric /= current_barycentric.sum()
        residual = transported
