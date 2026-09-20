"""Adjoint VJP for the sparse rectangle finite-difference BVP."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .rectangle_fd import rectangle_beltrami_fd


def rectangle_fd_boundary_vjp(
    mu: np.ndarray, boundary_values: np.ndarray, grad_map: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the real-inner-product VJP with respect to Dirichlet data."""

    coefficients = np.asarray(mu, dtype=np.complex128)
    boundary = np.asarray(boundary_values, dtype=np.complex128)
    cotangent = np.asarray(grad_map, dtype=np.complex128)
    if coefficients.ndim != 2 or boundary.shape != coefficients.shape or cotangent.shape != coefficients.shape:
        raise ValueError("mu, boundary_values, and grad_map must have matching 2D shapes")
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    result = rectangle_beltrami_fd(coefficients, boundary, method="direct")
    ny, nx = coefficients.shape
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    unknowns = {(j, i): k for k, (j, i) in enumerate(
        ( (j, i) for j in range(1, ny - 1) for i in range(1, nx - 1) )
    )}
    size = len(unknowns)
    matrix = lil_matrix((size, size), dtype=np.complex128)
    boundary_terms: list[tuple[int, int, int, complex]] = []
    row = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            coeff = coefficients[j, i]
            xcoef = (1.0 - coeff) / (4.0 * dx)
            ycoef = 1j * (1.0 + coeff) / (4.0 * dy)
            for jj, ii, value in ((j, i + 1, xcoef), (j, i - 1, -xcoef), (j + 1, i, ycoef), (j - 1, i, -ycoef)):
                if (jj, ii) in unknowns:
                    matrix[row, unknowns[(jj, ii)]] += value
                else:
                    boundary_terms.append((row, jj, ii, value))
            row += 1
    adjoint = spsolve(matrix.tocsr().conj().T, cotangent[1:-1, 1:-1].ravel())
    gradient = cotangent.copy()
    for row, j, i, value in boundary_terms:
        gradient[j, i] -= np.conjugate(value) * adjoint[row]
    residual = float(np.max(np.abs(matrix.tocsr().conj().T @ adjoint - cotangent[1:-1, 1:-1].ravel())))
    return np.ascontiguousarray(gradient), residual


def rectangle_fd_coefficient_vjp(
    mu: np.ndarray, boundary_values: np.ndarray, grad_map: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the real-inner-product VJP with respect to interior ``mu``."""

    coefficients = np.asarray(mu, dtype=np.complex128)
    boundary = np.asarray(boundary_values, dtype=np.complex128)
    cotangent = np.asarray(grad_map, dtype=np.complex128)
    if coefficients.ndim != 2 or boundary.shape != coefficients.shape or cotangent.shape != coefficients.shape:
        raise ValueError("mu, boundary_values, and grad_map must have matching 2D shapes")
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")
    result = rectangle_beltrami_fd(coefficients, boundary, method="direct")
    ny, nx = coefficients.shape
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    unknowns = {(j, i): k for k, (j, i) in enumerate(
        ((j, i) for j in range(1, ny - 1) for i in range(1, nx - 1))
    )}
    size = len(unknowns)
    matrix = lil_matrix((size, size), dtype=np.complex128)
    row = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            coeff = coefficients[j, i]
            xcoef = (1.0 - coeff) / (4.0 * dx)
            ycoef = 1j * (1.0 + coeff) / (4.0 * dy)
            for jj, ii, value in ((j, i + 1, xcoef), (j, i - 1, -xcoef), (j + 1, i, ycoef), (j - 1, i, -ycoef)):
                if (jj, ii) in unknowns:
                    matrix[row, unknowns[(jj, ii)]] += value
            row += 1
    sparse_matrix = matrix.tocsr()
    adjoint = spsolve(sparse_matrix.conj().T, cotangent[1:-1, 1:-1].ravel())
    fx = (result.map[1:-1, 2:] - result.map[1:-1, :-2]) / (2.0 * dx)
    fy = (result.map[2:, 1:-1] - result.map[:-2, 1:-1]) / (2.0 * dy)
    dz = 0.5 * (fx - 1j * fy)
    gradient = np.zeros_like(coefficients)
    gradient[1:-1, 1:-1] = adjoint.reshape((ny - 2, nx - 2)) * np.conjugate(dz)
    residual = float(np.max(np.abs(sparse_matrix.conj().T @ adjoint - cotangent[1:-1, 1:-1].ravel())))
    return np.ascontiguousarray(gradient), residual
