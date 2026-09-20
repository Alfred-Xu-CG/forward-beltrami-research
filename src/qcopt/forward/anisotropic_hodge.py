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


def diamond_stencil_decomposition(
    tensor: np.ndarray,
    *,
    positivity_tol: float = 1e-12,
    symmetry_tol: float = 1e-12,
) -> DiamondStencilResult:
    """Return the unique four-direction decomposition and its positivity flag.

    The coefficient order is ``(c_x, c_y, c_plus, c_minus)``.  The matrix is
    required to be finite, symmetric, and positive definite.  Negative
    coefficients are retained in the result so callers can distinguish exact
    algebraic representation from a valid positive conductance network.
    """
    matrix = np.asarray(tensor, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("tensor must be a finite 2x2 matrix")
    if not np.allclose(matrix, matrix.T, atol=symmetry_tol, rtol=0.0):
        raise ValueError("tensor must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError("tensor must be positive definite")
    off = float(matrix[0, 1])
    magnitude = abs(off)
    coefficients = np.array(
        [matrix[0, 0] - magnitude, matrix[1, 1] - magnitude, max(off, 0.0), max(-off, 0.0)],
        dtype=np.float64,
    )
    directions = np.array(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)))
    reconstructed = np.einsum("k,ki,kj->ij", coefficients, directions, directions)
    feasible = bool(np.min(coefficients) >= -positivity_tol)
    return DiamondStencilResult(coefficients, reconstructed, feasible)
