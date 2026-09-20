"""A boundary-conditioned finite-difference rectangle Beltrami reference solve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr, spsolve


@dataclass(frozen=True)
class RectangleFDSolveResult:
    map: np.ndarray
    max_equation_residual: float
    iterations: int
    converged: bool


class RectangleFDFactorization:
    """Reusable sparse operator for repeated rectangle boundary solves."""

    def __init__(self, mu: np.ndarray) -> None:
        coefficients = np.asarray(mu, dtype=np.complex128)
        if coefficients.ndim != 2 or min(coefficients.shape) < 3:
            raise ValueError("mu must be a 2D grid with both axes at least three")
        if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) >= 1.0:
            raise ValueError("mu must be finite and lie strictly inside the unit disk")
        self.mu = np.ascontiguousarray(coefficients)
        self.ny, self.nx = coefficients.shape
        self.dx, self.dy = 1.0 / (self.nx - 1), 1.0 / (self.ny - 1)
        self.unknowns = {
            (j, i): k
            for k, (j, i) in enumerate(
                ( (j, i) for j in range(1, self.ny - 1) for i in range(1, self.nx - 1) )
            )
        }
        self.matrix = lil_matrix((len(self.unknowns), len(self.unknowns)), dtype=np.complex128)
        self.boundary_terms: list[tuple[int, int, int, complex]] = []
        row = 0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                coeff = self.mu[j, i]
                xcoef = (1.0 - coeff) / (4.0 * self.dx)
                ycoef = 1j * (1.0 + coeff) / (4.0 * self.dy)
                for jj, ii, value in ((j, i + 1, xcoef), (j, i - 1, -xcoef), (j + 1, i, ycoef), (j - 1, i, -ycoef)):
                    if (jj, ii) in self.unknowns:
                        self.matrix[row, self.unknowns[(jj, ii)]] += value
                    else:
                        self.boundary_terms.append((row, jj, ii, value))
                row += 1
        from scipy.sparse.linalg import factorized

        self._solve = factorized(self.matrix.tocsc())

    def solve(self, boundary_values: np.ndarray) -> RectangleFDSolveResult:
        boundary = np.asarray(boundary_values, dtype=np.complex128)
        if boundary.shape != self.mu.shape or not np.all(np.isfinite(boundary)):
            raise ValueError("boundary_values must be finite and match mu shape")
        rhs = np.zeros(len(self.unknowns), dtype=np.complex128)
        for row, j, i, value in self.boundary_terms:
            rhs[row] -= value * boundary[j, i]
        solution = self._solve(rhs)
        values = boundary.copy()
        for (j, i), column in self.unknowns.items():
            values[j, i] = solution[column]
        residual = _equation_residual(values, self.mu, self.dx, self.dy)
        return RectangleFDSolveResult(np.ascontiguousarray(values), residual, 0, True)


def rectangle_beltrami_fd(
    mu: np.ndarray,
    boundary_values: np.ndarray,
    *,
    atol: float = 1e-14,
    btol: float = 1e-14,
    iter_lim: int | None = None,
    method: str = "direct",
) -> RectangleFDSolveResult:
    """Solve ``dbar f = mu dz f`` with supplied full-grid Dirichlet boundary.

    Central differences are used only at interior nodes and the resulting
    complex overdetermined sparse system is solved by LSQR. This explicitly
    removes the periodic FFT assumption, but it is a reference BVP solver, not
    a boundary-free forward decoder.
    """

    coefficients = np.asarray(mu, dtype=np.complex128)
    boundary = np.asarray(boundary_values, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 3:
        raise ValueError("mu must be a 2D grid with both axes at least three")
    if boundary.shape != coefficients.shape:
        raise ValueError("boundary_values shape must match mu shape")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(boundary)):
        raise ValueError("inputs must be finite")
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    if method not in {"direct", "lsqr"}:
        raise ValueError("method must be 'direct' or 'lsqr'")
    ny, nx = coefficients.shape
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    unknowns: dict[tuple[int, int], int] = {}
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            unknowns[(j, i)] = len(unknowns)
    rows = (ny - 2) * (nx - 2)
    matrix = lil_matrix((rows, len(unknowns)), dtype=np.complex128)
    rhs = np.zeros(rows, dtype=np.complex128)

    def add_term(row: int, j: int, i: int, value: complex) -> None:
        if (j, i) in unknowns:
            matrix[row, unknowns[(j, i)]] += value
        else:
            rhs[row] -= value * boundary[j, i]

    row = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            coefficient = coefficients[j, i]
            xcoef = (1.0 - coefficient) / (4.0 * dx)
            ycoef = 1j * (1.0 + coefficient) / (4.0 * dy)
            add_term(row, j, i + 1, xcoef)
            add_term(row, j, i - 1, -xcoef)
            add_term(row, j + 1, i, ycoef)
            add_term(row, j - 1, i, -ycoef)
            row += 1
    sparse_matrix = matrix.tocsr()
    if method == "direct":
        solution = spsolve(sparse_matrix, rhs)
        iterations, converged = 0, True
    else:
        solved = lsqr(sparse_matrix, rhs, atol=atol, btol=btol, iter_lim=iter_lim)
        solution = solved[0]
        iterations, converged = int(solved[2]), bool(solved[1] in (1, 2))
    values = boundary.copy()
    for (j, i), column in unknowns.items():
        values[j, i] = solution[column]
    residual = _equation_residual(values, coefficients, dx, dy)
    return RectangleFDSolveResult(
        np.ascontiguousarray(values), residual, iterations, converged
    )


def _equation_residual(values: np.ndarray, mu: np.ndarray, dx: float, dy: float) -> float:
    fx = (values[1:-1, 2:] - values[1:-1, :-2]) / (2.0 * dx)
    fy = (values[2:, 1:-1] - values[:-2, 1:-1]) / (2.0 * dy)
    dbar = 0.5 * (fx + 1j * fy)
    dz = 0.5 * (fx - 1j * fy)
    return float(np.max(np.abs(dbar - mu[1:-1, 1:-1] * dz)))
