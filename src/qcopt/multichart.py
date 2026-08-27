"""Matched atlases and transition-aware overlap compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse

from .constraints import LinearConstraints
from .mesh import TriMesh


@dataclass(frozen=True)
class Chart:
    name: str
    mesh: TriMesh


@dataclass(frozen=True)
class AffineTransition:
    source_chart: str
    target_chart: str
    matrix: np.ndarray
    offset: np.ndarray

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64).reshape(-1)
        if matrix.shape != (2, 2) or offset.shape != (2,):
            raise ValueError("an affine chart transition requires a 2x2 matrix and 2-vector")
        if abs(np.linalg.det(matrix)) <= 1e-14:
            raise ValueError("chart transition matrix must be invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.matrix.T + self.offset

    def inverse(self) -> "AffineTransition":
        inverse_matrix = np.linalg.inv(self.matrix)
        return AffineTransition(
            self.target_chart,
            self.source_chart,
            inverse_matrix,
            -inverse_matrix @ self.offset,
        )


@dataclass(frozen=True)
class Overlap:
    chart_c: str
    chart_d: str
    indices_c: np.ndarray
    indices_d: np.ndarray
    source_transition: AffineTransition
    target_transition: AffineTransition

    def __post_init__(self):
        indices_c = np.asarray(self.indices_c, dtype=np.int64).reshape(-1)
        indices_d = np.asarray(self.indices_d, dtype=np.int64).reshape(-1)
        if len(indices_c) == 0 or len(indices_c) != len(indices_d):
            raise ValueError("overlap index arrays must be nonempty and equally sized")
        if (
            self.source_transition.source_chart != self.chart_c
            or self.source_transition.target_chart != self.chart_d
            or self.target_transition.source_chart != self.chart_c
            or self.target_transition.target_chart != self.chart_d
        ):
            raise ValueError("transition directions must match the overlap chart order")
        object.__setattr__(self, "indices_c", indices_c)
        object.__setattr__(self, "indices_d", indices_d)


@dataclass(frozen=True)
class Atlas:
    charts: tuple[Chart, ...]
    overlaps: tuple[Overlap, ...]

    def __post_init__(self):
        names = [chart.name for chart in self.charts]
        if len(names) == 0 or len(set(names)) != len(names):
            raise ValueError("atlas chart names must be nonempty and unique")
        lookup = {chart.name: chart for chart in self.charts}
        for overlap in self.overlaps:
            if overlap.chart_c not in lookup or overlap.chart_d not in lookup:
                raise ValueError("overlap references an unknown chart")
            chart_c, chart_d = lookup[overlap.chart_c], lookup[overlap.chart_d]
            if np.any(overlap.indices_c >= chart_c.mesh.n_vertices) or np.any(
                overlap.indices_d >= chart_d.mesh.n_vertices
            ):
                raise ValueError("overlap vertex index out of range")
            source_points = chart_c.mesh.vertices[overlap.indices_c]
            target_points = chart_d.mesh.vertices[overlap.indices_d]
            if not np.allclose(
                overlap.source_transition.apply(source_points), target_points, atol=1e-10
            ):
                raise ValueError("source overlap points violate their chart transition")

    @property
    def n_coordinates(self) -> int:
        return sum(2 * chart.mesh.n_vertices for chart in self.charts)

    def chart(self, name: str) -> Chart:
        for chart in self.charts:
            if chart.name == name:
                return chart
        raise KeyError(name)

    def coordinate_offset(self, name: str) -> int:
        offset = 0
        for chart in self.charts:
            if chart.name == name:
                return offset
            offset += 2 * chart.mesh.n_vertices
        raise KeyError(name)

    def pack_maps(self, maps: dict[str, np.ndarray]) -> np.ndarray:
        vectors = []
        for chart in self.charts:
            values = np.asarray(maps[chart.name], dtype=np.float64)
            if values.shape != (chart.mesh.n_vertices, 2):
                raise ValueError("chart map has incompatible shape")
            vectors.append(values.T.reshape(-1))
        return np.concatenate(vectors)

    def unpack_maps(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if len(values) != self.n_coordinates:
            raise ValueError("global coordinate vector has incompatible length")
        result: dict[str, np.ndarray] = {}
        offset = 0
        for chart in self.charts:
            count = 2 * chart.mesh.n_vertices
            local = values[offset : offset + count]
            result[chart.name] = np.column_stack(
                (local[: chart.mesh.n_vertices], local[chart.mesh.n_vertices :])
            )
            offset += count
        return result

    def embed_chart_constraints(
        self, chart_name: str, constraints: LinearConstraints
    ) -> LinearConstraints:
        chart = self.chart(chart_name)
        if constraints.C.shape[1] != 2 * chart.mesh.n_vertices:
            raise ValueError("local constraint columns do not match chart")
        offset = self.coordinate_offset(chart_name)
        local = constraints.C.tocoo()
        global_matrix = sparse.coo_matrix(
            (local.data, (local.row, local.col + offset)),
            shape=(local.shape[0], self.n_coordinates),
        ).tocsr()
        return LinearConstraints(global_matrix, constraints.d)


def build_affine_compatibility(atlas: Atlas) -> LinearConstraints:
    return _compatibility_constraints(atlas, use_target_transition=True)


def raw_coordinate_equality_constraints(atlas: Atlas) -> LinearConstraints:
    return _compatibility_constraints(atlas, use_target_transition=False)


def _compatibility_constraints(
    atlas: Atlas, *, use_target_transition: bool
) -> LinearConstraints:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs: list[float] = []
    row = 0
    for overlap in atlas.overlaps:
        chart_c, chart_d = atlas.chart(overlap.chart_c), atlas.chart(overlap.chart_d)
        offset_c, offset_d = atlas.coordinate_offset(overlap.chart_c), atlas.coordinate_offset(overlap.chart_d)
        matrix = overlap.target_transition.matrix if use_target_transition else np.eye(2)
        offset = overlap.target_transition.offset if use_target_transition else np.zeros(2)
        for index_c, index_d in zip(overlap.indices_c, overlap.indices_d, strict=True):
            for output_coordinate in range(2):
                rows.append(row)
                columns.append(offset_d + output_coordinate * chart_d.mesh.n_vertices + int(index_d))
                values.append(1.0)
                for input_coordinate in range(2):
                    rows.append(row)
                    columns.append(offset_c + input_coordinate * chart_c.mesh.n_vertices + int(index_c))
                    values.append(-float(matrix[output_coordinate, input_coordinate]))
                rhs.append(float(offset[output_coordinate]))
                row += 1
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(row, atlas.n_coordinates)
    ).tocsr()
    return LinearConstraints(matrix, np.asarray(rhs))


def nonlinear_compatibility_residual_jacobian(
    source_values: np.ndarray,
    target_values: np.ndarray,
    transition: Callable[[np.ndarray], np.ndarray],
    transition_jacobian: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target overlap values must both have shape (n, 2)")
    residual = (target - transition(source)).reshape(-1)
    jacobians = np.asarray(transition_jacobian(source), dtype=np.float64)
    if jacobians.shape != (len(source), 2, 2):
        raise ValueError("transition Jacobian must have shape (n, 2, 2)")
    dimension = source.size
    matrix = np.zeros((dimension, 2 * dimension), dtype=np.float64)
    for index in range(len(source)):
        rows = slice(2 * index, 2 * index + 2)
        cols = slice(2 * index, 2 * index + 2)
        matrix[rows, cols] = -jacobians[index]
        matrix[rows, dimension + 2 * index : dimension + 2 * index + 2] = np.eye(2)
    return residual, matrix


def conformal_beltrami_transition(
    mu: np.ndarray, source_transition_derivative: complex
) -> np.ndarray:
    derivative = complex(source_transition_derivative)
    if abs(derivative) <= 1e-15:
        raise ValueError("conformal transition derivative must be nonzero")
    return np.asarray(mu, dtype=np.complex128) * derivative / np.conjugate(derivative)
