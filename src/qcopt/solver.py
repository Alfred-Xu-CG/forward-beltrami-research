"""Shared sparse solve result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

FloatArray = NDArray[np.float64]


@dataclass
class SparseSolveState:
    system: sparse.csc_matrix
    rhs: FloatArray
    solution: FloatArray
    factor: Any
    coordinate_slice: slice
    residual_slice: slice | None = None
    multiplier_slice: slice | None = None


@dataclass
class ReducedSolveState:
    """State of a hard-pin-eliminated coordinate solve.

    ``system`` is the free-coordinate block and ``factor`` is deliberately
    retained so the exact same numeric factorization can serve the transpose
    adjoint solve.
    """

    system: sparse.csc_matrix
    rhs: FloatArray
    solution: FloatArray
    factor: Any
    coordinates: FloatArray
    free_indices: NDArray[np.int64]
    pinned_indices: NDArray[np.int64]
    pinned_values: FloatArray


@dataclass
class SolveResult:
    uv: FloatArray
    primal_residual: float
    constraint_residual: float
    state: SparseSolveState | ReducedSolveState


def vector_to_uv(vector: FloatArray, n_vertices: int) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (2 * n_vertices,):
        raise ValueError("coordinate vector has incorrect size")
    return np.column_stack((vector[:n_vertices], vector[n_vertices:]))
