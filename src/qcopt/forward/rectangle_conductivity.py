"""Boundary-adapted conductivity solve for a rectangular Beltrami BVP.

The collocated first-order central-difference equation has an odd-even
checkerboard nullspace.  This module uses the classical conductivity
reduction instead: solve an SPD finite-element equation for ``u=Re f`` and
recover ``v=Im f`` from ``grad(v)=J A(mu) grad(u)`` in a weak projection.
It is a boundary-conditioned reference/control, not a boundary-free forward
decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class RectangleConductivityResult:
    map: np.ndarray
    equation_residual: float
    boundary_imag_mismatch: float
    min_triangle_determinant: float
    converged: bool


def rectangle_beltrami_conductivity(
    mu: np.ndarray, boundary_values: np.ndarray
) -> RectangleConductivityResult:
    """Solve a boundary-conditioned Beltrami equation by conductivity FEM.

    ``mu`` and ``boundary_values`` are nodal rectangular arrays with axis 0
    corresponding to ``y`` and axis 1 to ``x``.  The real boundary data are
    imposed in the SPD conductivity solve; the supplied imaginary boundary
    data are imposed in the weak recovery solve.  The returned residual is
    evaluated elementwise on the same two-triangle Q1/P1 mesh.
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

    ny, nx = coefficients.shape
    hx, hy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    n_vertices = nx * ny
    triangles: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            triangles.extend(((n00, n10, n11), (n00, n11, n01)))

    gradients = np.empty((len(triangles), 3, 2), dtype=np.float64)
    areas = np.full(len(triangles), 0.5 * hx * hy, dtype=np.float64)
    tri_mu = np.empty(len(triangles), dtype=np.complex128)
    for face_index, face in enumerate(triangles):
        points = np.array(
            [
                (face[0] % nx * hx, face[0] // nx * hy),
                (face[1] % nx * hx, face[1] // nx * hy),
                (face[2] % nx * hx, face[2] // nx * hy),
            ],
            dtype=np.float64,
        )
        twice_area = (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1]) - (
            points[1, 1] - points[0, 1]
        ) * (points[2, 0] - points[0, 0])
        gradients[face_index, 0] = (
            (points[1, 1] - points[2, 1]) / twice_area,
            (points[2, 0] - points[1, 0]) / twice_area,
        )
        gradients[face_index, 1] = (
            (points[2, 1] - points[0, 1]) / twice_area,
            (points[0, 0] - points[2, 0]) / twice_area,
        )
        gradients[face_index, 2] = (
            (points[0, 1] - points[1, 1]) / twice_area,
            (points[1, 0] - points[0, 0]) / twice_area,
        )
        tri_mu[face_index] = np.mean(
            coefficients[[face[0] // nx, face[1] // nx, face[2] // nx],
                         [face[0] % nx, face[1] % nx, face[2] % nx]]
        )

    conductivity = np.empty((len(triangles), 2, 2), dtype=np.float64)
    for index, coefficient in enumerate(tri_mu):
        a, b = float(coefficient.real), float(coefficient.imag)
        denominator = 1.0 - a * a - b * b
        conductivity[index] = np.array(
            [
                [(1.0 - a) ** 2 + b * b, -2.0 * b],
                [-2.0 * b, (1.0 + a) ** 2 + b * b],
            ]
        ) / denominator

    stiffness = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
    laplace = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
    rhs_v = np.zeros(n_vertices, dtype=np.float64)
    # Only boundary entries are data. Interior entries may be arbitrary and
    # must not leak into either the conductivity solve or flux recovery.
    u_boundary = np.zeros(n_vertices, dtype=np.float64)
    v_boundary = np.zeros(n_vertices, dtype=np.float64)
    for face_index, face in enumerate(triangles):
        grad = gradients[face_index]
        local_laplace = areas[face_index] * (grad @ grad.T)
        local_stiffness = areas[face_index] * (grad @ conductivity[face_index] @ grad.T)
        for i_local, i_global in enumerate(face):
            for j_local, j_global in enumerate(face):
                stiffness[i_global, j_global] += local_stiffness[i_local, j_local]
                laplace[i_global, j_global] += local_laplace[i_local, j_local]

    boundary_mask = np.zeros(n_vertices, dtype=bool)
    boundary_mask[:nx] = True
    boundary_mask[-nx:] = True
    boundary_mask[::nx] = True
    boundary_mask[nx - 1 :: nx] = True
    interior = np.flatnonzero(~boundary_mask)
    boundary_indices = np.flatnonzero(boundary_mask)
    supplied_real = boundary.real.ravel()
    supplied_imag = boundary.imag.ravel()
    u_boundary[boundary_indices] = supplied_real[boundary_indices]
    v_boundary[boundary_indices] = supplied_imag[boundary_indices]
    stiffness_csr = stiffness.tocsr()
    u = u_boundary.copy()
    u_rhs = -stiffness_csr[interior][:, boundary_indices] @ u_boundary[boundary_indices]
    u[interior] = spsolve(stiffness_csr[interior][:, interior], u_rhs)

    # Recover the conjugate coordinate from the solved conductivity field,
    # rather than from caller-provided interior values.
    for face_index, face in enumerate(triangles):
        grad = gradients[face_index]
        grad_u = u[list(face)] @ grad
        A = conductivity[face_index]
        # J A grad(u), where J rotates a vector counter-clockwise.
        target = np.array([-float((A @ grad_u)[1]), float((A @ grad_u)[0])])
        rhs_v[list(face)] += areas[face_index] * (grad @ target)

    laplace_csr = laplace.tocsr()
    v = v_boundary.copy()
    v_rhs = rhs_v[interior] - laplace_csr[interior][:, boundary_indices] @ v_boundary[boundary_indices]
    v[interior] = spsolve(laplace_csr[interior][:, interior], v_rhs)
    values = (u + 1j * v).reshape(ny, nx)

    residuals = []
    determinants = []
    for face_index, face in enumerate(triangles):
        grad = gradients[face_index]
        local = values.ravel()[list(face)]
        fx, fy = local @ grad
        dbar = 0.5 * (fx + 1j * fy)
        dz = 0.5 * (fx - 1j * fy)
        residuals.append(abs(dbar - tri_mu[face_index] * dz))
        determinants.append(float(np.imag(np.conjugate(fx) * fy)))

    boundary_mismatch = _boundary_tangential_mismatch(
        values.ravel(), triangles, gradients, areas, conductivity, boundary_mask, nx, ny, hx, hy
    )
    return RectangleConductivityResult(
        map=np.ascontiguousarray(values),
        equation_residual=float(np.max(residuals)),
        boundary_imag_mismatch=boundary_mismatch,
        min_triangle_determinant=float(np.min(determinants)),
        converged=True,
    )


def _boundary_tangential_mismatch(
    values: np.ndarray,
    triangles: list[tuple[int, int, int]],
    gradients: np.ndarray,
    areas: np.ndarray,
    conductivity: np.ndarray,
    boundary_mask: np.ndarray,
    nx: int,
    ny: int,
    hx: float,
    hy: float,
) -> float:
    del areas, ny
    edge_count: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(triangles):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((a, b)))
            edge_count.setdefault(key, []).append(face_index)
    face_targets: dict[int, np.ndarray] = {}
    for face_index, face in enumerate(triangles):
        grad_u = values[list(face)].real @ gradients[face_index]
        flux = conductivity[face_index] @ grad_u
        face_targets[face_index] = np.array([-flux[1], flux[0]])
    mismatch = 0.0
    for (a, b), incident in edge_count.items():
        if len(incident) != 1 or not (boundary_mask[a] and boundary_mask[b]):
            continue
        dx = (b % nx - a % nx) * hx
        dy = (b // nx - a // nx) * hy
        edge = np.array([dx, dy])
        predicted = float(np.dot(face_targets[incident[0]], edge))
        mismatch = max(mismatch, abs((values[b].imag - values[a].imag) - predicted))
    return float(mismatch)
