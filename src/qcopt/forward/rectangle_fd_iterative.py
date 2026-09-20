"""Reusable ILU-preconditioned GMRES reference for the rectangle FD BVP."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import LinearOperator, gmres, spilu


@dataclass(frozen=True)
class RectangleFDIterativeResult:
    map: np.ndarray
    equation_residual: float
    info: int
    iterations: int
    assembly_seconds: float
    preconditioner_seconds: float
    solve_seconds: float
    transpose_iterations: int
    transpose_info: int
    transpose_seconds: float
    transpose_residual: float


def _assemble(mu: np.ndarray):
    coefficients = np.asarray(mu, dtype=np.complex128)
    ny, nx = coefficients.shape
    unknowns = -np.ones((ny, nx), dtype=np.int64)
    unknowns[1:-1, 1:-1] = np.arange((ny - 2) * (nx - 2), dtype=np.int64).reshape(ny - 2, nx - 2)
    matrix = lil_matrix(((ny - 2) * (nx - 2), (ny - 2) * (nx - 2)), dtype=np.complex128)
    boundary_terms: list[tuple[int, int, int, complex]] = []
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    row = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            coefficient = coefficients[j, i]
            xcoef = (1.0 - coefficient) / (4.0 * dx)
            ycoef = 1j * (1.0 + coefficient) / (4.0 * dy)
            for jj, ii, value in ((j, i + 1, xcoef), (j, i - 1, -xcoef), (j + 1, i, ycoef), (j - 1, i, -ycoef)):
                column = unknowns[jj, ii]
                if column >= 0:
                    matrix[row, column] += value
                else:
                    boundary_terms.append((row, jj, ii, value))
            row += 1
    return matrix.tocsr(), unknowns, boundary_terms


def _residual(values: np.ndarray, mu: np.ndarray) -> float:
    ny, nx = values.shape
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    fx = (values[1:-1, 2:] - values[1:-1, :-2]) / (2.0 * dx)
    fy = (values[2:, 1:-1] - values[:-2, 1:-1]) / (2.0 * dy)
    dbar = 0.5 * (fx + 1j * fy)
    dz = 0.5 * (fx - 1j * fy)
    return float(np.max(np.abs(dbar - mu[1:-1, 1:-1] * dz)))


def solve_rectangle_fd_gmres(
    mu: np.ndarray,
    boundary_values: np.ndarray,
    *,
    rtol: float = 1e-10,
    restart: int = 100,
    maxiter: int = 1000,
    drop_tol: float = 1e-4,
    fill_factor: float = 10.0,
    transpose_rhs: np.ndarray | None = None,
) -> RectangleFDIterativeResult:
    """Solve the sparse rectangle system with ILU-preconditioned GMRES.

    The returned transpose iteration statistics are an implicit-layer control:
    callers can provide a cotangent RHS and solve the conjugate-transpose
    system without unrolling forward iterations.
    """

    coefficients = np.asarray(mu, dtype=np.complex128)
    boundary = np.asarray(boundary_values, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 3 or boundary.shape != coefficients.shape:
        raise ValueError("mu and boundary_values must be matching 2D arrays")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(boundary)):
        raise ValueError("inputs must be finite")
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    started = time.perf_counter()
    matrix, unknowns, boundary_terms = _assemble(coefficients)
    assembly_seconds = time.perf_counter() - started
    rhs = np.zeros(matrix.shape[0], dtype=np.complex128)
    for row, j, i, value in boundary_terms:
        rhs[row] -= value * boundary[j, i]
    precondition_started = time.perf_counter()
    try:
        ilu = spilu(matrix.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor, diag_pivot_thresh=0.0)
    except RuntimeError:
        # The centered complex stencil has zero diagonal and can produce an
        # exact ILU pivot breakdown at tighter drop tolerances.  A looser,
        # higher-fill incomplete factor is still a valid preconditioner and
        # avoids changing the solved operator.
        ilu = spilu(
            matrix.tocsc(), drop_tol=max(drop_tol, 1e-2),
            fill_factor=max(fill_factor, 20.0), diag_pivot_thresh=0.0,
        )
    preconditioner_seconds = time.perf_counter() - precondition_started
    precondition = LinearOperator(matrix.shape, matvec=ilu.solve, dtype=np.complex128)
    forward_history: list[float] = []
    solve_started = time.perf_counter()
    solution, info = gmres(
        matrix, rhs, M=precondition, restart=restart, maxiter=maxiter,
        rtol=rtol, atol=0.0, callback=forward_history.append, callback_type="pr_norm"
    )
    solve_seconds = time.perf_counter() - solve_started
    values = boundary.copy()
    values[1:-1, 1:-1] = solution.reshape((coefficients.shape[0] - 2, coefficients.shape[1] - 2))
    if transpose_rhs is None:
        transpose_iterations, transpose_info, transpose_seconds, transpose_residual = 0, 0, 0.0, 0.0
    else:
        cotangent = np.asarray(transpose_rhs, dtype=np.complex128)
        if cotangent.shape != coefficients.shape:
            raise ValueError("transpose_rhs must match mu shape")
        transpose_started = time.perf_counter()
        try:
            ilu_h = spilu(matrix.conj().T.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor, diag_pivot_thresh=0.0)
        except RuntimeError:
            ilu_h = spilu(
                matrix.conj().T.tocsc(), drop_tol=max(drop_tol, 1e-2),
                fill_factor=max(fill_factor, 20.0), diag_pivot_thresh=0.0,
            )
        transpose_precondition = LinearOperator(matrix.shape, matvec=ilu_h.solve, dtype=np.complex128)
        transpose_history: list[float] = []
        transpose_solution, transpose_info = gmres(
            matrix.conj().T, cotangent[1:-1, 1:-1].ravel(), M=transpose_precondition,
            restart=restart, maxiter=maxiter, rtol=rtol, atol=0.0,
            callback=transpose_history.append, callback_type="pr_norm"
        )
        transpose_iterations = len(transpose_history)
        transpose_seconds = time.perf_counter() - transpose_started
        transpose_residual = float(np.linalg.norm(matrix.conj().T @ transpose_solution - cotangent[1:-1, 1:-1].ravel()) / max(np.linalg.norm(cotangent[1:-1, 1:-1]), 1e-30))
    return RectangleFDIterativeResult(
        np.ascontiguousarray(values), _residual(values, coefficients), int(info),
        len(forward_history), assembly_seconds, preconditioner_seconds, solve_seconds,
        transpose_iterations, int(transpose_info), transpose_seconds, transpose_residual,
    )
