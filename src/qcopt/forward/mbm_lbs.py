"""Reference mixed-boundary modulus Beltrami solver on a structured rectangle.

This module is deliberately a reference/control implementation. It uses two P1
sparse solves (primary and complementary) and exposes the residuals needed to
decide whether the mixed-boundary construction can become a hard-bijective
neural layer. It does not claim a global homeomorphism theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class MBMLBSResult:
    u: np.ndarray
    v: np.ndarray
    complementary: np.ndarray
    modulus: float
    primary_energy: float
    complementary_energy: float
    conjugacy_residual: float
    min_triangle_determinant: float
    left_monotone: bool
    right_monotone: bool


def solve_mbm_lbs(mu: np.ndarray, *, monotonicity_tol: float = 1e-10) -> MBMLBSResult:
    """Solve the primary and complementary mixed conductivity problems."""
    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 3:
        raise ValueError("mu must be a 2D grid with both axes at least three")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("mu must be finite")
    if np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must lie strictly inside the unit disk")

    ny, nx = coefficients.shape
    triangles, gradients, areas, face_mu = _structured_geometry(coefficients)
    face_A = np.stack([_conductivity_tensor(value) for value in face_mu])
    n_vertices = nx * ny
    stiffness = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
    for face_index, face in enumerate(triangles):
        local = areas[face_index] * (
            gradients[face_index] @ face_A[face_index] @ gradients[face_index].T
        )
        for i_local, i_global in enumerate(face):
            for j_local, j_global in enumerate(face):
                stiffness[i_global, j_global] += local[i_local, j_local]
    stiffness = stiffness.tocsr()

    left = np.arange(0, n_vertices, nx)
    right = np.arange(nx - 1, n_vertices, nx)
    bottom = np.arange(nx)
    top = np.arange((ny - 1) * nx, ny * nx)

    u = _solve_dirichlet(
        stiffness, np.concatenate((left, right)), np.r_[np.zeros(ny), np.ones(ny)]
    )
    w = _solve_dirichlet(
        stiffness, np.concatenate((bottom, top)), np.r_[np.zeros(nx), np.ones(nx)]
    )

    primary_energy = float(u @ (stiffness @ u))
    complementary_energy = float(w @ (stiffness @ w))
    if complementary_energy <= 0.0:
        raise ValueError("complementary energy must be positive")
    modulus = 1.0 / complementary_energy
    v = modulus * w

    conjugacy = []
    determinants = []
    for face_index, face in enumerate(triangles):
        grad_u = u[list(face)] @ gradients[face_index]
        grad_v = v[list(face)] @ gradients[face_index]
        target = np.array(
            [
                -float((face_A[face_index] @ grad_u)[1]),
                float((face_A[face_index] @ grad_u)[0]),
            ]
        )
        conjugacy.append(float(np.linalg.norm(grad_v - target)))
        determinants.append(float(grad_u[0] * grad_v[1] - grad_u[1] * grad_v[0]))

    u_grid = u.reshape(ny, nx)
    v_grid = v.reshape(ny, nx)
    left_increments = np.diff(v_grid[:, 0])
    right_increments = np.diff(v_grid[:, -1])
    return MBMLBSResult(
        u=u_grid,
        v=v_grid,
        complementary=w.reshape(ny, nx),
        modulus=modulus,
        primary_energy=primary_energy,
        complementary_energy=complementary_energy,
        conjugacy_residual=float(max(conjugacy)),
        min_triangle_determinant=float(min(determinants)),
        left_monotone=bool(np.min(left_increments) >= -monotonicity_tol),
        right_monotone=bool(np.min(right_increments) >= -monotonicity_tol),
    )


def _solve_dirichlet(matrix, fixed_indices: np.ndarray, fixed_values: np.ndarray) -> np.ndarray:
    fixed_indices = np.asarray(fixed_indices, dtype=np.int64)
    fixed_values = np.asarray(fixed_values, dtype=np.float64)
    if fixed_indices.size != fixed_values.size:
        raise ValueError("fixed indices and values must have the same length")
    n = matrix.shape[0]
    values = np.zeros(n, dtype=np.float64)
    values[fixed_indices] = fixed_values
    fixed_mask = np.zeros(n, dtype=bool)
    fixed_mask[fixed_indices] = True
    free = np.flatnonzero(~fixed_mask)
    if free.size:
        values[free] = spsolve(
            matrix[free][:, free],
            -matrix[free][:, fixed_indices] @ values[fixed_indices],
        )
    return values


def _conductivity_tensor(coefficient: complex) -> np.ndarray:
    a, b = float(coefficient.real), float(coefficient.imag)
    denominator = 1.0 - a * a - b * b
    return np.array(
        [
            [(1.0 - a) ** 2 + b * b, -2.0 * b],
            [-2.0 * b, (1.0 + a) ** 2 + b * b],
        ],
        dtype=np.float64,
    ) / denominator


def _structured_geometry(coefficients: np.ndarray):
    ny, nx = coefficients.shape
    hx, hy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    triangles: list[tuple[int, int, int]] = []
    gradients = []
    areas = []
    face_mu = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            triangles.extend(((n00, n10, n11), (n00, n11, n01)))
    for face in triangles:
        points = np.array(
            [
                (face[0] % nx * hx, face[0] // nx * hy),
                (face[1] % nx * hx, face[1] // nx * hy),
                (face[2] % nx * hx, face[2] // nx * hy),
            ],
            dtype=np.float64,
        )
        twice_area = (
            (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
            - (points[1, 1] - points[0, 1]) * (points[2, 0] - points[0, 0])
        )
        gradients.append(
            np.array(
                [
                    [(points[1, 1] - points[2, 1]) / twice_area,
                     (points[2, 0] - points[1, 0]) / twice_area],
                    [(points[2, 1] - points[0, 1]) / twice_area,
                     (points[0, 0] - points[2, 0]) / twice_area],
                    [(points[0, 1] - points[1, 1]) / twice_area,
                     (points[1, 0] - points[0, 0]) / twice_area],
                ],
                dtype=np.float64,
            )
        )
        areas.append(0.5 * twice_area)
        rows = [vertex // nx for vertex in face]
        cols = [vertex % nx for vertex in face]
        face_mu.append(np.mean(coefficients[rows, cols]))
    return triangles, np.asarray(gradients), np.asarray(areas), np.asarray(face_mu)
