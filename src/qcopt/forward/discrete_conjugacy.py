"""Small structure-preserving P1 conductivity/conjugacy primitives.

These routines deliberately expose the compatibility test instead of claiming
that two independent finite-element solves automatically form a QC map.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from ..mesh import TriMesh


def _validate_inputs(
    mesh: TriMesh,
    face_a: np.ndarray,
    *,
    require_unit_determinant: bool = False,
) -> np.ndarray:
    tensors = np.asarray(face_a, dtype=np.float64)
    if tensors.shape != (mesh.n_faces, 2, 2) or not np.all(np.isfinite(tensors)):
        raise ValueError("face_a must have shape (n_faces, 2, 2) and be finite")
    if not np.allclose(tensors, tensors.transpose(0, 2, 1), atol=1e-12, rtol=0.0):
        raise ValueError("face tensors must be symmetric")
    if any(tensor[0, 0] <= 0.0 or tensor[1, 1] <= 0.0 or np.linalg.det(tensor) <= 0.0 for tensor in tensors):
        raise ValueError("face tensors must be positive definite")
    if require_unit_determinant and not np.allclose(
        np.linalg.det(tensors), 1.0, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("conjugacy tensors must have determinant one")
    return tensors


def assemble_facewise_conductivity(mesh: TriMesh, face_a: np.ndarray) -> csr_matrix:
    """Assemble the standard facewise-constant P1 conductivity matrix."""

    tensors = _validate_inputs(mesh, face_a)
    matrix = lil_matrix((mesh.n_vertices, mesh.n_vertices), dtype=np.float64)
    for face_index, face in enumerate(mesh.faces):
        gradients = mesh.gradients[face_index]
        local = mesh.areas[face_index] * gradients @ tensors[face_index] @ gradients.T
        for i_local, i_global in enumerate(face):
            for j_local, j_global in enumerate(face):
                matrix[int(i_global), int(j_global)] += local[i_local, j_local]
    return matrix.tocsr()


def p1_conjugacy_residual(
    mesh: TriMesh,
    u: np.ndarray,
    v: np.ndarray,
    face_a: np.ndarray,
) -> float:
    """Return the largest facewise residual of ``grad(v)=J A grad(u)``."""

    tensors = _validate_inputs(mesh, face_a, require_unit_determinant=True)
    values_u = np.asarray(u, dtype=np.float64)
    values_v = np.asarray(v, dtype=np.float64)
    if values_u.shape != (mesh.n_vertices,) or values_v.shape != (mesh.n_vertices,):
        raise ValueError("u and v must have one value per mesh vertex")
    if not np.all(np.isfinite(values_u)) or not np.all(np.isfinite(values_v)):
        raise ValueError("u and v must be finite")
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    residuals = []
    for face_index, face in enumerate(mesh.faces):
        gradients = mesh.gradients[face_index]
        grad_u = values_u[face] @ gradients
        grad_v = values_v[face] @ gradients
        difference = grad_v - rotation @ tensors[face_index] @ grad_u
        residuals.append(float(np.sqrt(np.sum(difference * difference))))
    return float(max(residuals, default=0.0))
