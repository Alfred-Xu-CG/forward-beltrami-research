"""Small structure-preserving P1 conductivity/conjugacy primitives.

These routines deliberately expose the compatibility test instead of claiming
that two independent finite-element solves automatically form a QC map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from ..mesh import TriMesh


@dataclass(frozen=True)
class DiscreteStreamResult:
    """Edge-integrated stream coordinate and compatibility diagnostics."""

    values: np.ndarray
    edge_inconsistency: float
    cycle_residual: float
    exact: bool


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


def integrate_stream_from_face_flux(
    mesh: TriMesh,
    u: np.ndarray,
    face_a: np.ndarray,
    *,
    gauge_vertex: int = 0,
    tolerance: float = 1e-10,
) -> DiscreteStreamResult:
    """Integrate ``r_T = J A_T grad(u)|_T`` along primal edges.

    For a compatible piecewise-affine map, the edge line integral computed from
    either incident face agrees and the resulting edge 1-form is exact. The
    routine chooses the mean of the incident predictions, integrates a
    spanning tree, and reports both the face-to-face mismatch and the residual
    on all remaining edges. Boundary singleton edges contribute zero to the
    mismatch statistic; the cycle residual is informative only on non-tree
    edges because tree-edge residuals vanish by construction. It is a
    diagnostic/structure-preserving prototype,
    not a replacement for a general mixed finite-element Hodge star.
    """
    tensors = _validate_inputs(mesh, face_a, require_unit_determinant=True)
    values_u = np.asarray(u, dtype=np.float64)
    if values_u.shape != (mesh.n_vertices,) or not np.all(np.isfinite(values_u)):
        raise ValueError("u must have one finite value per mesh vertex")
    if not 0 <= gauge_vertex < mesh.n_vertices:
        raise ValueError("gauge_vertex is out of range")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")

    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    edge_predictions: dict[tuple[int, int], list[float]] = {}
    for face_index, face in enumerate(mesh.faces):
        grad_u = values_u[face] @ mesh.gradients[face_index]
        stream_grad = rotation @ tensors[face_index] @ grad_u
        for local in range(3):
            start = int(face[local])
            end = int(face[(local + 1) % 3])
            key = (min(start, end), max(start, end))
            displacement = mesh.vertices[end] - mesh.vertices[start]
            delta = float(stream_grad @ displacement)
            if (start, end) != key:
                delta = -delta
            edge_predictions.setdefault(key, []).append(delta)

    edge_data = {
        key: float(np.mean(predictions))
        for key, predictions in edge_predictions.items()
    }
    edge_inconsistency = max(
        (float(np.ptp(predictions)) for predictions in edge_predictions.values()),
        default=0.0,
    )
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(mesh.n_vertices)]
    for (start, end), delta in edge_data.items():
        adjacency[start].append((end, delta))
        adjacency[end].append((start, -delta))

    values = np.full(mesh.n_vertices, np.nan, dtype=np.float64)
    values[gauge_vertex] = 0.0
    queue = [gauge_vertex]
    while queue:
        vertex = queue.pop()
        for neighbor, delta in adjacency[vertex]:
            candidate = values[vertex] + delta
            if np.isnan(values[neighbor]):
                values[neighbor] = candidate
                queue.append(neighbor)
    if np.any(np.isnan(values)):
        raise ValueError("mesh edge graph is disconnected")
    cycle_residual = max(
        (
            abs(values[end] - values[start] - delta)
            for (start, end), delta in edge_data.items()
        ),
        default=0.0,
    )
    return DiscreteStreamResult(
        values=values,
        edge_inconsistency=edge_inconsistency,
        cycle_residual=float(cycle_residual),
        exact=bool(max(edge_inconsistency, cycle_residual) <= tolerance),
    )
