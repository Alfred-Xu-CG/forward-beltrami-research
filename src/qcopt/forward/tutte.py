"""Positive-weight Tutte/Floater convex-combination decoder."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.typing import NDArray

from ..mesh import TriMesh

FloatArray = NDArray[np.float64]


def tutte_embedding(
    mesh: TriMesh, target_boundary: FloatArray | None = None
) -> FloatArray:
    """Solve uniform positive barycentric coordinates for interior vertices.

    The target boundary must be a counter-clockwise convex polygon, allowing
    collinear samples along its edges. Under the usual Tutte graph hypotheses,
    this positive-combination solve is an injective planar embedding.
    """

    if len(mesh.boundary_loops) != 1:
        raise ValueError("Tutte embedding requires exactly one boundary loop")
    loop = mesh.boundary_loops[0]
    if target_boundary is None:
        target = mesh.vertices[loop].copy()
    else:
        target = np.asarray(target_boundary, dtype=np.float64)
    if target.shape != (len(loop), 2) or not np.all(np.isfinite(target)):
        raise ValueError("target_boundary must have shape (boundary_vertices, 2)")
    if not _is_counterclockwise_convex(target):
        raise ValueError("target boundary must be counter-clockwise convex")

    boundary = set(loop.tolist())
    interior, system, boundary_weights = _interior_system(mesh, loop)
    result = mesh.vertices.copy()
    result[loop] = target
    if not interior:
        return result
    rhs = boundary_weights @ target
    solved = spsolve(system, rhs)
    result[np.asarray(interior, dtype=np.int64)] = np.asarray(solved, dtype=np.float64)
    return np.ascontiguousarray(result)


def tutte_weights(mesh: TriMesh) -> FloatArray:
    """Return dense boundary barycentric weights for small/medium meshes.

    The dense matrix is convenient for explicit differentiation and tests. For
    high-resolution layers, solve the same sparse system with an adjoint or a
    matrix-free Jacobian-vector product instead of materializing this matrix.
    """

    if len(mesh.boundary_loops) != 1:
        raise ValueError("Tutte embedding requires exactly one boundary loop")
    loop = mesh.boundary_loops[0]
    interior, system, boundary_weights = _interior_system(mesh, loop)
    result = np.zeros((mesh.n_vertices, len(loop)), dtype=np.float64)
    result[loop, np.arange(len(loop))] = 1.0
    if interior:
        result[np.asarray(interior, dtype=np.int64)] = np.asarray(
            spsolve(system, boundary_weights.toarray()), dtype=np.float64
        )
    return np.ascontiguousarray(result)


def tutte_embedding_weighted(
    mesh: TriMesh,
    target_boundary: FloatArray,
    edge_weights: dict[tuple[int, int], float],
) -> FloatArray:
    """Solve a positive nonuniform barycentric embedding.

    ``edge_weights`` supplies a strictly positive symmetric weight for every
    mesh edge. Positivity, rather than uniformity, is the hard-injectivity
    ingredient; the routine is intentionally dictionary/sparse based so it can
    be driven by a learned edge field without a dense all-pairs matrix.
    """

    if len(mesh.boundary_loops) != 1:
        raise ValueError("weighted Tutte embedding requires exactly one boundary loop")
    loop = mesh.boundary_loops[0]
    target = np.asarray(target_boundary, dtype=np.float64)
    if target.shape != (len(loop), 2) or not np.all(np.isfinite(target)):
        raise ValueError("target_boundary must have shape (boundary_vertices, 2)")
    if not _is_counterclockwise_convex(target):
        raise ValueError("target boundary must be counter-clockwise convex")
    adjacency = [set() for _ in range(mesh.n_vertices)]
    for a, b, c in mesh.faces.tolist():
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    boundary_set = set(loop.tolist())
    interior = [v for v in range(mesh.n_vertices) if v not in boundary_set]
    index = {v: k for k, v in enumerate(interior)}
    boundary_index = {v: k for k, v in enumerate(loop.tolist())}
    system = lil_matrix((len(interior), len(interior)), dtype=np.float64)
    coupling = lil_matrix((len(interior), len(loop)), dtype=np.float64)
    for vertex in interior:
        row = index[vertex]
        neighbors = sorted(adjacency[vertex])
        values = []
        for neighbor in neighbors:
            key = (min(vertex, neighbor), max(vertex, neighbor))
            if key not in edge_weights or not np.isfinite(edge_weights[key]) or edge_weights[key] <= 0.0:
                raise ValueError("every mesh edge needs a finite strictly positive weight")
            values.append((neighbor, float(edge_weights[key])))
        total = sum(value for _, value in values)
        system[row, row] = 1.0
        for neighbor, value in values:
            normalized = value / total
            if neighbor in index:
                system[row, index[neighbor]] -= normalized
            else:
                coupling[row, boundary_index[neighbor]] += normalized
    result = mesh.vertices.copy()
    result[loop] = target
    if interior:
        solved = spsolve(system.tocsr(), coupling.tocsr() @ target)
        result[np.asarray(interior, dtype=np.int64)] = solved
    return np.ascontiguousarray(result)


def _interior_system(mesh: TriMesh, loop: NDArray[np.int64]):
    boundary = set(loop.tolist())
    interior = [vertex for vertex in range(mesh.n_vertices) if vertex not in boundary]
    index = {vertex: row for row, vertex in enumerate(interior)}
    adjacency = [set() for _ in range(mesh.n_vertices)]
    for a, b, c in mesh.faces.tolist():
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    system = lil_matrix((len(interior), len(interior)), dtype=np.float64)
    boundary_weights = lil_matrix(
        (len(interior), len(loop)), dtype=np.float64
    )
    boundary_index = {vertex: column for column, vertex in enumerate(loop.tolist())}
    for vertex in interior:
        row = index[vertex]
        neighbors = sorted(adjacency[vertex])
        if not neighbors:
            raise ValueError("interior vertex has no graph neighbors")
        weight = 1.0 / len(neighbors)
        system[row, row] = 1.0
        for neighbor in neighbors:
            if neighbor in index:
                system[row, index[neighbor]] -= weight
            else:
                boundary_weights[row, boundary_index[neighbor]] += weight
    return interior, system.tocsr(), boundary_weights.tocsr()


def _is_counterclockwise_convex(polygon: FloatArray, tolerance: float = 1e-12) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
        edges[:, 0], -1
    )
    area_twice = float(
        np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    )
    positive = crosses > tolerance
    nonnegative = crosses >= -tolerance
    return bool(area_twice > tolerance and np.all(nonnegative) and np.any(positive))
