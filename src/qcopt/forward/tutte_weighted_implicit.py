"""Implicit derivatives for positive weighted Tutte embeddings."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

from ..mesh import TriMesh
from .tutte import _is_counterclockwise_convex


def weighted_tutte_vjp(
    mesh: TriMesh,
    target_boundary: np.ndarray,
    edge_weights: dict[tuple[int, int], float],
    output_gradient: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Return a weighted Tutte map and its VJP with respect to edge weights.

    Each interior row uses ``p_ij=w_ij/sum_j w_ij``.  The transpose solve
    computes the adjoint, after which every edge derivative is assembled from
    the local residual ``(map[j]-map[i])/sum_j w_ij``.  The returned dictionary
    combines contributions from both endpoints of an undirected edge.
    """

    if len(mesh.boundary_loops) != 1:
        raise ValueError("weighted Tutte requires one boundary loop")
    loop = np.asarray(mesh.boundary_loops[0], dtype=np.int64)
    target = np.asarray(target_boundary, dtype=np.float64)
    grad = np.asarray(output_gradient, dtype=np.float64)
    if target.shape != (len(loop), 2) or not np.all(np.isfinite(target)):
        raise ValueError("target_boundary has the wrong shape or non-finite values")
    if not _is_counterclockwise_convex(target):
        raise ValueError("target boundary must be counter-clockwise convex")
    if grad.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(grad)):
        raise ValueError("output_gradient must have shape (n_vertices, 2)")

    boundary_set = set(loop.tolist())
    interior = [v for v in range(mesh.n_vertices) if v not in boundary_set]
    index = {v: k for k, v in enumerate(interior)}
    boundary_index = {v: k for k, v in enumerate(loop.tolist())}
    adjacency = [set() for _ in range(mesh.n_vertices)]
    for a, b, c in mesh.faces.tolist():
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    system = lil_matrix((len(interior), len(interior)), dtype=np.float64)
    coupling = lil_matrix((len(interior), len(loop)), dtype=np.float64)
    row_data: list[list[tuple[int, float]]] = []
    for vertex in interior:
        values = []
        for neighbor in sorted(adjacency[vertex]):
            key = (min(vertex, neighbor), max(vertex, neighbor))
            value = edge_weights.get(key)
            if value is None or not np.isfinite(value) or value <= 0.0:
                raise ValueError("every mesh edge needs a finite strictly positive weight")
            values.append((neighbor, float(value)))
        total = sum(value for _, value in values)
        row = index[vertex]
        system[row, row] = 1.0
        for neighbor, value in values:
            p = value / total
            if neighbor in index:
                system[row, index[neighbor]] -= p
            else:
                coupling[row, boundary_index[neighbor]] += p
        row_data.append(values)

    result = mesh.vertices.copy()
    result[loop] = target
    if not interior:
        return result, {key: 0.0 for key in edge_weights}
    matrix = system.tocsr()
    solve = factorized(matrix.tocsc())
    solve_t = factorized(matrix.T.tocsc())
    result[np.asarray(interior)] = solve(coupling.tocsr() @ target)
    adjoint = solve_t(grad[np.asarray(interior)])
    edge_gradient = {key: 0.0 for key in edge_weights}
    for row, vertex in enumerate(interior):
        total = sum(value for _, value in row_data[row])
        for neighbor, _ in row_data[row]:
            key = (min(vertex, neighbor), max(vertex, neighbor))
            local = (result[neighbor] - result[vertex]) / total
            edge_gradient[key] += float(np.dot(adjoint[row], local))
    return np.ascontiguousarray(result), edge_gradient

