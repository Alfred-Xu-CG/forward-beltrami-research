"""Discrete operators shared by forward Beltrami route experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..beltrami import face_beltrami, face_jacobians
from ..injectivity import audit_injectivity
from ..mesh import TriMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ForwardMetrics:
    """Solver-independent metrics for a single discrete map."""

    topology_certified: bool
    flipped_faces: tuple[int, ...]
    min_det: float
    mu_l2: float
    mu_linf: float
    equation_linf: float


def face_jacobian_determinants(mesh: TriMesh, uv: FloatArray) -> FloatArray:
    """Return the signed P1 Jacobian determinant on each face."""

    return np.ascontiguousarray(np.linalg.det(face_jacobians(mesh, uv)))


def beltrami_residual(
    mesh: TriMesh, uv: FloatArray, mu: ComplexArray
) -> FloatArray:
    """Return ``|f_bar - mu*f_z|`` on each face."""

    jac = face_jacobians(mesh, uv)
    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.shape != (mesh.n_faces,):
        raise ValueError("mu must have shape (mesh.n_faces,)")
    ux, uy = jac[:, 0, 0], jac[:, 0, 1]
    vx, vy = jac[:, 1, 0], jac[:, 1, 1]
    fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
    fbar = 0.5 * ((ux - vy) + 1j * (vx + uy))
    return np.ascontiguousarray(np.abs(fbar - coefficients * fz))


def evaluate_forward_metrics(
    mesh: TriMesh,
    uv: FloatArray,
    target_mu: ComplexArray | None = None,
) -> ForwardMetrics:
    """Evaluate geometry, topology, and optional target-BC errors independently."""

    uv = np.asarray(uv, dtype=np.float64)
    measured_mu = face_beltrami(mesh, uv)
    report = audit_injectivity(mesh, uv)
    min_det = float(np.min(face_jacobian_determinants(mesh, uv)))
    if target_mu is None:
        mu_l2 = mu_linf = equation_linf = float("nan")
    else:
        target = np.asarray(target_mu, dtype=np.complex128)
        if target.shape != (mesh.n_faces,):
            raise ValueError("target_mu must have shape (mesh.n_faces,)")
        difference = measured_mu - target
        weights = mesh.areas / float(np.sum(mesh.areas))
        mu_l2 = float(np.sqrt(np.sum(weights * np.abs(difference) ** 2)))
        mu_linf = float(np.max(np.abs(difference)))
        equation_linf = float(np.max(beltrami_residual(mesh, uv, target)))
    return ForwardMetrics(
        topology_certified=bool(report.certified),
        flipped_faces=report.flipped_faces,
        min_det=min_det,
        mu_l2=mu_l2,
        mu_linf=mu_linf,
        equation_linf=equation_linf,
    )


def periodic_grid_edges(nx: int, ny: int) -> IntArray:
    """Return unique horizontal/vertical edges of an ``nx`` by ``ny`` torus grid."""

    if not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
        raise TypeError("nx and ny must be integers")
    if nx < 2 or ny < 2:
        raise ValueError("periodic grid dimensions must both be at least two")
    edges: list[tuple[int, int]] = []

    def index(i: int, j: int) -> int:
        return (j % ny) * nx + (i % nx)

    for j in range(ny):
        for i in range(nx):
            here = index(i, j)
            edges.append((here, index(i + 1, j)))
            edges.append((here, index(i, j + 1)))
    result = np.asarray(edges, dtype=np.int64)
    canonical = np.sort(result, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    return np.ascontiguousarray(result[np.sort(unique_indices)])
