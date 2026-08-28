"""Validated triangular surface meshes and their native PL atlas topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SurfaceTopology:
    """Combinatorial properties needed by closed-surface registration."""

    vertices: int
    edges: int
    faces: int
    euler_characteristic: int
    boundary_edges: int
    connected_components: int
    connected: bool
    closed: bool
    oriented: bool
    genus: int | None


@dataclass(frozen=True)
class CommonRefinementRepair:
    merged_vertices: int
    removed_faces: int
    perturbed_source_vertices: int
    perturbed_target_vertices: int
    maximum_perturbation: float
    quantization_step: float


@dataclass(frozen=True)
class CommonRefinement:
    source: "SurfaceMesh"
    target: "SurfaceMesh"
    repair: CommonRefinementRepair


@dataclass(frozen=True)
class SurfaceMesh:
    """A finite, nondegenerate, edge-manifold triangular surface in R3.

    Faces are also the smallest charts of the piecewise-linear atlas.  Local
    edge ``i`` is opposite local vertex ``i``; this convention makes a crossed
    barycentric coordinate directly index :attr:`face_neighbors`.
    """

    vertices: FloatArray
    faces: IntArray
    face_areas: FloatArray = field(init=False, repr=False)
    face_normals: FloatArray = field(init=False, repr=False)
    face_frames: FloatArray = field(init=False, repr=False)
    barycentric_gradients: FloatArray = field(init=False, repr=False)
    face_neighbors: IntArray = field(init=False, repr=False)
    vertex_normals: FloatArray = field(init=False, repr=False)
    vertex_faces: tuple[IntArray, ...] = field(init=False, repr=False)
    minimum_edge_length: float = field(init=False)
    median_edge_length: float = field(init=False)
    topology: SurfaceTopology = field(init=False)

    def __post_init__(self) -> None:
        # Own the input buffers.  Merely marking an array view read-only would
        # unexpectedly freeze the caller's arrays and would still allow the
        # caller to re-enable writes and invalidate all cached geometry.
        vertices = np.array(self.vertices, dtype=np.float64, copy=True, order="C")
        faces = np.array(self.faces, dtype=np.int64, copy=True, order="C")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if len(vertices) < 3 or len(faces) < 1:
            raise ValueError("mesh must contain vertices and faces")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index out of range")
        sorted_faces = np.sort(faces, axis=1)
        if np.any(sorted_faces[:, 1:] == sorted_faces[:, :-1]):
            raise ValueError("each face must contain three distinct vertices")

        p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
        cross = np.cross(p1 - p0, p2 - p0)
        twice_area = np.linalg.norm(cross, axis=1)
        scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
        tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
        if np.any(twice_area <= tolerance):
            raise ValueError("surface contains a degenerate face")
        normals = cross / twice_area[:, None]
        first_axis = (p1 - p0) / np.linalg.norm(p1 - p0, axis=1)[:, None]
        second_axis = np.cross(normals, first_axis)
        frames = np.stack((first_axis, second_axis), axis=1)
        # Gradients of the affine barycentric basis on each supporting plane.
        local_p1 = np.column_stack(
            (np.linalg.norm(p1 - p0, axis=1), np.zeros(len(faces)))
        )
        local_p2 = np.column_stack(
            (
                np.sum((p2 - p0) * first_axis, axis=1),
                np.sum((p2 - p0) * second_axis, axis=1),
            )
        )
        local_basis = np.stack((local_p1, local_p2), axis=2)
        inverse_local_basis = np.linalg.inv(local_basis)
        gradients_12 = np.einsum("fij,fjk->fik", inverse_local_basis, frames)
        barycentric_gradients = np.empty((len(faces), 3, 3), dtype=np.float64)
        barycentric_gradients[:, 1:] = gradients_12
        barycentric_gradients[:, 0] = -gradients_12.sum(axis=1)

        neighbors = np.full((len(faces), 3), -1, dtype=np.int64)
        edge_records: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        incident: list[list[int]] = [[] for _ in range(len(vertices))]
        for face_id, (a, b, c) in enumerate(faces.tolist()):
            for vertex in (a, b, c):
                incident[vertex].append(face_id)
            for local_edge, (start, end) in enumerate(((b, c), (c, a), (a, b))):
                key = (min(start, end), max(start, end))
                edge_records.setdefault(key, []).append(
                    (face_id, local_edge, start, end)
                )
        nonmanifold = [edge for edge, records in edge_records.items() if len(records) > 2]
        if nonmanifold:
            raise ValueError(
                f"non-manifold edge belongs to more than two faces: {nonmanifold[0]}"
            )
        isolated = next(
            (vertex for vertex, values in enumerate(incident) if not values),
            None,
        )
        if isolated is not None:
            raise ValueError(f"surface contains an isolated vertex: {isolated}")

        oriented = True
        for records in edge_records.values():
            if len(records) == 2:
                first, second = records
                neighbors[first[0], first[1]] = second[0]
                neighbors[second[0], second[1]] = first[0]
                if not (first[2] == second[3] and first[3] == second[2]):
                    oriented = False

        component_count = _face_component_count(neighbors)
        boundary_count = sum(len(records) == 1 for records in edge_records.values())
        chi = int(len(vertices) - len(edge_records) + len(faces))
        closed = boundary_count == 0
        connected = component_count == 1
        genus: int | None = None
        if closed and connected and oriented:
            numerator = 2 - chi
            if numerator >= 0 and numerator % 2 == 0:
                genus = numerator // 2

        weighted_normals = np.zeros_like(vertices)
        for local in range(3):
            np.add.at(weighted_normals, faces[:, local], cross)
        normal_lengths = np.linalg.norm(weighted_normals, axis=1)
        used = normal_lengths > tolerance
        weighted_normals[used] /= normal_lengths[used, None]

        vertex_faces = tuple(np.asarray(values, dtype=np.int64) for values in incident)
        topology = SurfaceTopology(
            vertices=len(vertices),
            edges=len(edge_records),
            faces=len(faces),
            euler_characteristic=chi,
            boundary_edges=boundary_count,
            connected_components=component_count,
            connected=connected,
            closed=closed,
            oriented=oriented,
            genus=genus,
        )
        edge_lengths = np.asarray(
            [
                np.linalg.norm(vertices[first] - vertices[second])
                for first, second in edge_records
            ]
        )
        minimum_edge_length = float(np.min(edge_lengths))
        median_edge_length = float(np.median(edge_lengths))
        if minimum_edge_length <= 0.0:
            raise ValueError("surface contains a zero-length edge")

        def immutable(array: np.ndarray) -> np.ndarray:
            result = np.array(array, copy=True, order="C")
            result.setflags(write=False)
            return result

        vertex_faces = tuple(immutable(array) for array in vertex_faces)
        object.__setattr__(self, "vertices", immutable(vertices))
        object.__setattr__(self, "faces", immutable(faces))
        object.__setattr__(self, "face_areas", immutable(0.5 * twice_area))
        object.__setattr__(self, "face_normals", immutable(normals))
        object.__setattr__(self, "face_frames", immutable(frames))
        object.__setattr__(
            self,
            "barycentric_gradients",
            immutable(barycentric_gradients),
        )
        object.__setattr__(self, "face_neighbors", immutable(neighbors))
        object.__setattr__(self, "vertex_normals", immutable(weighted_normals))
        object.__setattr__(self, "vertex_faces", vertex_faces)
        object.__setattr__(self, "minimum_edge_length", minimum_edge_length)
        object.__setattr__(self, "median_edge_length", median_edge_length)
        object.__setattr__(self, "topology", topology)

    @property
    def n_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    def has_same_connectivity(self, other: "SurfaceMesh") -> bool:
        """Return whether two meshes use the identical oriented triangulation."""

        return bool(
            self.faces.shape == other.faces.shape
            and self.n_vertices == other.n_vertices
            and np.array_equal(self.faces, other.faces)
        )

    def point_from_barycentric(self, face: int, barycentric: FloatArray) -> FloatArray:
        barycentric = np.asarray(barycentric, dtype=np.float64)
        if barycentric.shape != (3,):
            raise ValueError("barycentric coordinates must have shape (3,)")
        return barycentric @ self.vertices[self.faces[int(face)]]

    def barycentric_from_point(self, face: int, point: FloatArray) -> FloatArray:
        """Return affine barycentric coordinates in a face's supporting plane."""

        face = int(face)
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("point must have shape (3,)")
        triangle = self.vertices[self.faces[face]]
        barycentric = np.array([1.0, 0.0, 0.0])
        barycentric += self.barycentric_gradients[face] @ (point - triangle[0])
        return barycentric


def load_obj(path: str | Path) -> SurfaceMesh:
    """Load OBJ geometry, ignoring texture/normal indices and triangulating fans."""

    vertices, faces = _read_obj_geometry(Path(path))
    return SurfaceMesh(vertices, faces)


def load_common_refinement(
    source_path: str | Path,
    target_path: str | Path,
    *,
    quantization_step: float = 1e-6,
) -> CommonRefinement:
    """Load paired overlay OBJs and repair only auditable export quantization.

    Published overlay meshes may repeat the same topological point and write
    coordinates with six decimal places.  Exact coincidences in either
    embedding are merged jointly so source and target retain identical
    connectivity.  Remaining collinear faces receive a sub-quantization nudge;
    every such change is reported rather than silently relaxing validation.
    """

    if quantization_step <= 0.0:
        raise ValueError("quantization_step must be positive")
    source_vertices, source_faces = _read_obj_geometry(Path(source_path))
    target_vertices, target_faces = _read_obj_geometry(Path(target_path))
    if source_vertices.shape != target_vertices.shape:
        raise ValueError("common-refinement OBJs must have equal vertex counts")
    if not np.array_equal(source_faces, target_faces):
        raise ValueError("common-refinement OBJs must have identical geometry indices")

    count = len(source_vertices)
    parent = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)

    for embedding in (source_vertices, target_vertices):
        seen: dict[tuple[float, float, float], int] = {}
        for index, coordinate in enumerate(embedding):
            key = tuple(float(value) for value in coordinate)
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    for indices in groups.values():
        if len(indices) == 1:
            continue
        for embedding in (source_vertices, target_vertices):
            diameter = float(
                np.max(np.ptp(embedding[np.asarray(indices)], axis=0))
            )
            if diameter > 2.1 * quantization_step:
                raise ValueError(
                    "coincident overlay vertices disagree beyond OBJ quantization"
                )

    remap = np.empty(count, dtype=np.int64)
    source_merged: list[np.ndarray] = []
    target_merged: list[np.ndarray] = []
    for new_index, root in enumerate(sorted(groups)):
        indices = np.asarray(groups[root], dtype=np.int64)
        remap[indices] = new_index
        source_merged.append(np.mean(source_vertices[indices], axis=0))
        target_merged.append(np.mean(target_vertices[indices], axis=0))
    common_faces = remap[source_faces]
    distinct = np.all(
        np.sort(common_faces, axis=1)[:, 1:]
        != np.sort(common_faces, axis=1)[:, :-1],
        axis=1,
    )
    filtered = common_faces[distinct]
    unique_faces: list[np.ndarray] = []
    seen_faces: set[tuple[int, int, int]] = set()
    for face in filtered:
        key = tuple(sorted(int(value) for value in face))
        if key not in seen_faces:
            seen_faces.add(key)
            unique_faces.append(face)
    common_faces = np.asarray(unique_faces, dtype=np.int64)

    source_array = np.asarray(source_merged, dtype=np.float64)
    target_array = np.asarray(target_merged, dtype=np.float64)
    source_repaired, source_count, source_max = _repair_quantized_degeneracies(
        source_array, common_faces, quantization_step
    )
    target_repaired, target_count, target_max = _repair_quantized_degeneracies(
        target_array, common_faces, quantization_step
    )
    source = SurfaceMesh(source_repaired, common_faces)
    target = SurfaceMesh(target_repaired, common_faces)
    if source.topology.genus != target.topology.genus or not source.has_same_connectivity(target):
        raise ValueError("repaired common refinement does not preserve topology")
    return CommonRefinement(
        source,
        target,
        CommonRefinementRepair(
            merged_vertices=count - len(source_array),
            removed_faces=len(source_faces) - len(common_faces),
            perturbed_source_vertices=source_count,
            perturbed_target_vertices=target_count,
            maximum_perturbation=max(source_max, target_max),
            quantization_step=float(quantization_step),
        ),
    )


def _read_obj_geometry(path: Path) -> tuple[FloatArray, IntArray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"invalid vertex at OBJ line {line_number}")
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"invalid face at OBJ line {line_number}")
                polygon: list[int] = []
                for token in fields[1:]:
                    geometry_index = token.split("/", 1)[0]
                    if not geometry_index:
                        raise ValueError(f"missing geometry index at OBJ line {line_number}")
                    raw_index = int(geometry_index)
                    index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"face index out of range at OBJ line {line_number}")
                    polygon.append(index)
                for offset in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"OBJ contains no usable triangular surface: {path}")
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def _repair_quantized_degeneracies(
    vertices: FloatArray,
    faces: IntArray,
    quantization_step: float,
) -> tuple[FloatArray, int, float]:
    repaired = vertices.copy()
    original = vertices.copy()
    moved: set[int] = set()
    for _ in range(8):
        triangles = repaired[faces]
        twice_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        scale = max(float(np.ptp(repaired, axis=0).max()), 1.0)
        tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
        bad_faces = np.flatnonzero(twice_area <= tolerance)
        if len(bad_faces) == 0:
            displacement = np.linalg.norm(repaired - original, axis=1)
            return repaired, len(moved), float(np.max(displacement, initial=0.0))
        for face_id in bad_faces:
            face = faces[int(face_id)]
            points = repaired[face]
            selected = 2
            for local in range(3):
                other = [index for index in range(3) if index != local]
                edge = points[other[1]] - points[other[0]]
                denominator = float(edge @ edge)
                if denominator == 0.0:
                    continue
                parameter = float((points[local] - points[other[0]]) @ edge / denominator)
                if 0.0 <= parameter <= 1.0:
                    selected = local
                    break
            other = [index for index in range(3) if index != selected]
            edge = points[other[1]] - points[other[0]]
            edge_axis = edge / np.linalg.norm(edge)
            coordinate_axis = np.eye(3)[int(np.argmin(np.abs(edge_axis)))]
            direction = np.cross(edge_axis, coordinate_axis)
            direction /= np.linalg.norm(direction)
            vertex = int(face[selected])
            repaired[vertex] += 0.25 * quantization_step * direction
            moved.add(vertex)
    raise ValueError("could not repair OBJ quantization degeneracies")


def _face_component_count(neighbors: IntArray) -> int:
    unseen = set(range(len(neighbors)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                neighbor = int(neighbor)
                if neighbor >= 0 and neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return components
