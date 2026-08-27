"""Sparse linear constraints for planar QC maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .mesh import TriMesh

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LinearConstraints:
    C: sparse.csr_matrix
    d: FloatArray

    def __post_init__(self) -> None:
        matrix = sparse.csr_matrix(self.C, dtype=np.float64)
        rhs = np.asarray(self.d, dtype=np.float64).reshape(-1)
        if matrix.shape[0] != rhs.size:
            raise ValueError("constraint row count must match d")
        object.__setattr__(self, "C", matrix)
        object.__setattr__(self, "d", rhs)

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.C.toarray()))

    def stack(self, other: "LinearConstraints") -> "LinearConstraints":
        if self.C.shape[1] != other.C.shape[1]:
            raise ValueError("stacked constraints must have equal column counts")
        return LinearConstraints(
            sparse.vstack((self.C, other.C), format="csr"),
            np.concatenate((self.d, other.d)),
        )


def two_pin_constraints(
    n_vertices: int, vertices: list[int] | NDArray[np.int64], targets: FloatArray
) -> LinearConstraints:
    indices = np.asarray(vertices, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64)
    if len(indices) != 2 or targets.shape != (2, 2):
        raise ValueError("two pins and two two-dimensional targets are required")
    if indices[0] == indices[1]:
        raise ValueError("pin vertices must be distinct")
    return fixed_vertex_constraints(n_vertices, indices, targets)


def fixed_vertex_constraints(
    n_vertices: int, vertices: NDArray[np.int64], targets: FloatArray
) -> LinearConstraints:
    indices = np.asarray(vertices, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (len(indices), 2):
        raise ValueError("targets must have shape (len(vertices), 2)")
    if np.any(indices < 0) or np.any(indices >= n_vertices):
        raise ValueError("fixed vertex index out of range")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("fixed vertices must be unique")
    rows = np.repeat(np.arange(2 * len(indices)), 1)
    columns = np.empty(2 * len(indices), dtype=np.int64)
    values = np.ones(2 * len(indices), dtype=np.float64)
    rhs = np.empty(2 * len(indices), dtype=np.float64)
    for position, vertex in enumerate(indices):
        columns[2 * position] = vertex
        columns[2 * position + 1] = n_vertices + vertex
        rhs[2 * position : 2 * position + 2] = targets[position]
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(2 * len(indices), 2 * n_vertices)
    ).tocsr()
    return LinearConstraints(matrix, rhs)


def rectangle_sliding_constraints(mesh: TriMesh, tol: float = 1e-12) -> LinearConstraints:
    """Constrain boundary normals while leaving side tangents free."""

    xy = mesh.vertices
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    absolute_tol = tol * span
    entries: list[tuple[int, float]] = []
    for vertex, (x, y) in enumerate(xy):
        if abs(x - x_min) <= absolute_tol:
            entries.append((vertex, 0.0))
        elif abs(x - x_max) <= absolute_tol:
            entries.append((vertex, 1.0))
        if abs(y - y_min) <= absolute_tol:
            entries.append((mesh.n_vertices + vertex, 0.0))
        elif abs(y - y_max) <= absolute_tol:
            entries.append((mesh.n_vertices + vertex, 1.0))
    rows = np.arange(len(entries), dtype=np.int64)
    columns = np.asarray([column for column, _ in entries], dtype=np.int64)
    data = np.ones(len(entries), dtype=np.float64)
    rhs = np.asarray([value for _, value in entries], dtype=np.float64)
    matrix = sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(entries), 2 * mesh.n_vertices)
    ).tocsr()
    return LinearConstraints(matrix, rhs)
