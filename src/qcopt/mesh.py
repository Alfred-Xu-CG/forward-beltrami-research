"""Validated planar triangular meshes and piecewise-linear geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TriMesh:
    """A positively oriented, manifold planar triangle mesh."""

    vertices: FloatArray
    faces: IntArray
    areas: FloatArray = field(init=False, repr=False)
    gradients: FloatArray = field(init=False, repr=False)
    boundary_loops: tuple[IntArray, ...] = field(init=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (n, 2)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if len(vertices) < 3 or len(faces) < 1:
            raise ValueError("mesh must contain vertices and faces")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index out of range")
        if np.any(np.sort(faces, axis=1)[:, 1:] == np.sort(faces, axis=1)[:, :-1]):
            raise ValueError("each face must contain three distinct vertices")

        p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
        twice_area = _cross2(p1 - p0, p2 - p0)
        scale = max(float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])), 1.0)
        tolerance = 64.0 * np.finfo(np.float64).eps * scale * scale
        if np.any(twice_area <= tolerance):
            raise ValueError("all faces must have strict positive orientation")

        gradients = np.empty((len(faces), 3, 2), dtype=np.float64)
        gradients[:, 0, 0] = (p1[:, 1] - p2[:, 1]) / twice_area
        gradients[:, 0, 1] = (p2[:, 0] - p1[:, 0]) / twice_area
        gradients[:, 1, 0] = (p2[:, 1] - p0[:, 1]) / twice_area
        gradients[:, 1, 1] = (p0[:, 0] - p2[:, 0]) / twice_area
        gradients[:, 2, 0] = (p0[:, 1] - p1[:, 1]) / twice_area
        gradients[:, 2, 1] = (p1[:, 0] - p0[:, 0]) / twice_area

        vertices = np.ascontiguousarray(vertices)
        faces = np.ascontiguousarray(faces)
        areas = np.ascontiguousarray(0.5 * twice_area)
        gradients = np.ascontiguousarray(gradients)
        loops = _boundary_loops(faces)
        for array in (vertices, faces, areas, gradients, *loops):
            array.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "areas", areas)
        object.__setattr__(self, "gradients", gradients)
        object.__setattr__(self, "boundary_loops", loops)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def structured_rectangle(nx: int, ny: int) -> TriMesh:
    """Create a unit rectangle with ``nx * ny`` consistently split cells."""

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack((xx.ravel(), yy.ravel()))
    faces: list[tuple[int, int, int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            v00 = j * stride + i
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return TriMesh(vertices, np.asarray(faces, dtype=np.int64))


def _cross2(a: FloatArray, b: FloatArray) -> FloatArray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _boundary_loops(faces: IntArray) -> tuple[IntArray, ...]:
    oriented: dict[tuple[int, int], tuple[int, int]] = {}
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces.tolist():
        for start, end in ((a, b), (b, c), (c, a)):
            key = (min(start, end), max(start, end))
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > 2:
                raise ValueError("non-manifold edge belongs to more than two faces")
            oriented[key] = (start, end)
    boundary_edges = [oriented[key] for key, count in counts.items() if count == 1]
    if not boundary_edges:
        return ()
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for start, end in boundary_edges:
        if start in successor or end in predecessor:
            raise ValueError("boundary is not a disjoint union of simple vertex loops")
        successor[start] = end
        predecessor[end] = start
    if set(successor) != set(predecessor):
        raise ValueError("boundary edge orientation is inconsistent")
    remaining = set(successor)
    loops: list[IntArray] = []
    while remaining:
        start = min(remaining)
        loop = [start]
        current = successor[start]
        while current != start:
            if current not in remaining:
                raise ValueError("boundary traversal repeated a vertex")
            loop.append(current)
            current = successor[current]
        remaining.difference_update(loop)
        loops.append(np.asarray(loop, dtype=np.int64))
    return tuple(loops)
