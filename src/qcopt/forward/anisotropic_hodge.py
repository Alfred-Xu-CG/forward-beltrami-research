"""Small positive-stencil audit for anisotropic primal-dual conductances.

The four directions are e_x, e_y, e_+=(1,1), and e_-=(1,-1).  The returned
coefficients satisfy

    A = c_x e_x e_x^T + c_y e_y e_y^T
        + c_+ e_+ e_+^T + c_- e_- e_-^T.

This is an algebraic representability test, not a complete discrete exterior
calculus implementation.  It makes the positivity obstruction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiamondStencilResult:
    conductances: np.ndarray
    reconstructed: np.ndarray
    feasible: bool
    diagonal_mass: float
    positive_feasible_interval: tuple[float, float]

    @property
    def parameter_interval(self) -> tuple[float, float]:
        """Backward-compatible alias for the positive-conductance interval."""
        return self.positive_feasible_interval


def diamond_stencil_decomposition(
    tensor: np.ndarray,
    *,
    diagonal_mass: float | None = None,
    positivity_tol: float = 1e-12,
    symmetry_tol: float = 1e-12,
) -> DiamondStencilResult:
    """Return one member of the four-direction decomposition family.

    The coefficient order is ``(c_x, c_y, c_plus, c_minus)``.  The matrix is
    required to be finite, symmetric, and positive definite.  Negative
    coefficients are retained in the result so callers can distinguish exact
    algebraic representation from a valid positive conductance network.  The
    exact algebraic family allows every finite real value of
    ``diagonal_mass = c_plus + c_minus``.  Nonnegative conductances are
    possible only on the interval
    ``abs(A[0, 1]) <= diagonal_mass <= min(A[0, 0], A[1, 1])``.
    """
    matrix = np.asarray(tensor, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("tensor must be a finite 2x2 matrix")
    if not np.allclose(matrix, matrix.T, atol=symmetry_tol, rtol=0.0):
        raise ValueError("tensor must be symmetric")
    # Avoid a platform-dependent LAPACK eigensolver for a 2x2 SPD check.
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or np.linalg.det(matrix) <= 0.0:
        raise ValueError("tensor must be positive definite")
    off = float(matrix[0, 1])
    minimum_mass = abs(off)
    maximum_mass = min(float(matrix[0, 0]), float(matrix[1, 1]))
    if diagonal_mass is None:
        mass = minimum_mass
    else:
        mass = float(diagonal_mass)
        if not np.isfinite(mass):
            raise ValueError("diagonal_mass must be finite")
    coefficients = np.array(
        [matrix[0, 0] - mass, matrix[1, 1] - mass, 0.5 * (mass + off), 0.5 * (mass - off)],
        dtype=np.float64,
    )
    directions = np.array(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)))
    reconstructed = np.einsum("k,ki,kj->ij", coefficients, directions, directions)
    feasible = bool(np.min(coefficients) >= -positivity_tol)
    return DiamondStencilResult(
        coefficients,
        reconstructed,
        feasible,
        mass,
        (minimum_mass, maximum_mass),
    )
