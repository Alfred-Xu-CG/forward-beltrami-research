"""Positive-weight periodic harmonic embedding on a rectangular torus grid."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def periodic_positive_graph_embedding(
    nx: int,
    ny: int,
    offsets: np.ndarray,
    edge_weights: np.ndarray,
    *,
    period_x: np.ndarray | None = None,
    period_y: np.ndarray | None = None,
) -> np.ndarray:
    """Solve a positive quasi-periodic graph embedding with arbitrary offsets.

    ``offsets`` lists unoriented primitive graph directions.  For every offset
    ``(di,dj)``, ``edge_weights[k,j,i]`` belongs to the undirected edge from
    ``(i,j)`` to ``(i+di,j+dj)``; the reverse row reuses the weight stored at
    the neighboring base vertex.  Seam crossings carry the corresponding
    integer combination of ``period_x`` and ``period_y``.  Positive weights
    retain the graph maximum-principle structure while diagonal offsets add
    cross-direction coupling absent from the axis-only decoder.
    """

    if nx < 3 or ny < 3:
        raise ValueError("torus grid dimensions must be at least three")
    dirs = np.asarray(offsets, dtype=np.int64)
    weights = np.asarray(edge_weights, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 2 or len(dirs) < 2:
        raise ValueError("offsets must have shape (m,2) with m >= 2")
    if np.any(np.all(dirs == 0, axis=1)):
        raise ValueError("offsets must be nonzero")
    if len({(int(a), int(b)) for a, b in dirs.tolist()}) != len(dirs):
        raise ValueError("offsets must be unique")
    if weights.shape != (len(dirs), ny, nx) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("edge_weights must have shape (m,ny,nx) and be positive")
    px = np.array([1.0, 0.0]) if period_x is None else np.asarray(period_x, dtype=np.float64)
    py = np.array([0.0, 1.0]) if period_y is None else np.asarray(period_y, dtype=np.float64)
    if px.shape != (2,) or py.shape != (2,) or not np.all(np.isfinite(np.r_[px, py])):
        raise ValueError("period vectors must be finite 2-vectors")
    n = nx * ny
    anchor = 0
    unknown = {v: k for k, v in enumerate(v for v in range(n) if v != anchor)}
    matrix = lil_matrix((n - 1, n - 1), dtype=np.float64)
    rhs = np.zeros((n - 1, 2), dtype=np.float64)

    def vid(i: int, j: int) -> int:
        return (j % ny) * nx + (i % nx)

    def shift(i: int, j: int) -> np.ndarray:
        return (i // nx) * px + (j // ny) * py

    for j in range(ny):
        for i in range(nx):
            v = vid(i, j)
            if v == anchor:
                continue
            row = unknown[v]
            neighbors: list[tuple[int, float, np.ndarray]] = []
            for k, (di, dj) in enumerate(dirs.tolist()):
                ni, nj = i + int(di), j + int(dj)
                neighbors.append((vid(ni, nj), float(weights[k, j, i]), shift(ni, nj)))
                bi, bj = i - int(di), j - int(dj)
                # The reverse traversal uses the same undirected edge weight.
                neighbors.append((vid(bi, bj), float(weights[k, bj % ny, bi % nx]), -shift(i, j) + shift(bi, bj)))
            total = sum(weight for _, weight, _ in neighbors)
            matrix[row, row] = 1.0
            for neighbor, weight, seam in neighbors:
                p = weight / total
                if neighbor == anchor:
                    rhs[row] += p * seam
                else:
                    matrix[row, unknown[neighbor]] -= p
                    rhs[row] += p * seam
    solved = spsolve(matrix.tocsr(), rhs)
    result = np.zeros((n, 2), dtype=np.float64)
    for vertex, column in unknown.items():
        result[vertex] = solved[column]
    return np.ascontiguousarray(result)


def periodic_tutte_embedding(
    nx: int,
    ny: int,
    horizontal_weights: np.ndarray,
    vertical_weights: np.ndarray,
    *,
    period_x: np.ndarray | None = None,
    period_y: np.ndarray | None = None,
) -> np.ndarray:
    """Solve positive barycentric equations with quasi-periodic seam shifts.

    Horizontal weights have shape ``(ny,nx)`` and attach the edge from
    ``(i,j)`` to ``(i+1 mod nx,j)``; vertical weights analogously attach the
    edge to ``(i,j+1 mod ny)``.  The returned vertex coordinates use the
    fundamental cell. Seam consistency is represented by the supplied period
    vectors, not by identifying Euclidean seam endpoints.
    """

    if nx < 3 or ny < 3:
        raise ValueError("torus grid dimensions must be at least three")
    wx = np.asarray(horizontal_weights, dtype=np.float64)
    wy = np.asarray(vertical_weights, dtype=np.float64)
    if wx.shape != (ny, nx) or wy.shape != (ny, nx):
        raise ValueError("weight arrays must have shape (ny,nx)")
    if not np.all(np.isfinite(wx)) or not np.all(np.isfinite(wy)) or np.any(wx <= 0.0) or np.any(wy <= 0.0):
        raise ValueError("all periodic edge weights must be strictly positive")
    px = np.array([1.0, 0.0]) if period_x is None else np.asarray(period_x, dtype=np.float64)
    py = np.array([0.0, 1.0]) if period_y is None else np.asarray(period_y, dtype=np.float64)
    if px.shape != (2,) or py.shape != (2,) or not np.all(np.isfinite(np.r_[px, py])):
        raise ValueError("period vectors must be finite 2-vectors")
    n = nx * ny
    anchor = 0
    unknown = {v: k for k, v in enumerate(v for v in range(n) if v != anchor)}
    matrix = lil_matrix((n - 1, n - 1), dtype=np.float64)
    rhs = np.zeros((n - 1, 2), dtype=np.float64)

    def vid(i: int, j: int) -> int:
        return (j % ny) * nx + (i % nx)

    for j in range(ny):
        for i in range(nx):
            v = vid(i, j)
            if v == anchor:
                continue
            row = unknown[v]
            neighbors = [
                (vid(i + 1, j), wx[j, i], px if i == nx - 1 else np.zeros(2)),
                (vid(i - 1, j), wx[j, (i - 1) % nx], -px if i == 0 else np.zeros(2)),
                (vid(i, j + 1), wy[j, i], py if j == ny - 1 else np.zeros(2)),
                (vid(i, j - 1), wy[(j - 1) % ny, i], -py if j == 0 else np.zeros(2)),
            ]
            total = sum(weight for _, weight, _ in neighbors)
            matrix[row, row] = 1.0
            for neighbor, weight, shift in neighbors:
                p = weight / total
                if neighbor == anchor:
                    rhs[row] += p * shift
                else:
                    matrix[row, unknown[neighbor]] -= p
                    rhs[row] += p * shift
    solved = spsolve(matrix.tocsr(), rhs)
    result = np.zeros((n, 2), dtype=np.float64)
    result[anchor] = 0.0
    for vertex, column in unknown.items():
        result[vertex] = solved[column]
    return np.ascontiguousarray(result)


def periodic_torus_face_determinants(
    values: np.ndarray,
    nx: int,
    ny: int,
    period_x: np.ndarray | None = None,
    period_y: np.ndarray | None = None,
) -> np.ndarray:
    """Return lifted oriented triangle determinants for a periodic embedding."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (nx * ny, 2):
        raise ValueError("values must have shape (nx*ny,2)")
    px = np.array([1.0, 0.0]) if period_x is None else np.asarray(period_x, dtype=np.float64)
    py = np.array([0.0, 1.0]) if period_y is None else np.asarray(period_y, dtype=np.float64)
    dets = []
    for j in range(ny):
        for i in range(nx):
            p00 = values[j * nx + i]
            p10 = values[j * nx + ((i + 1) % nx)] + (px if i == nx - 1 else 0.0)
            p01 = values[((j + 1) % ny) * nx + i] + (py if j == ny - 1 else 0.0)
            p11 = values[((j + 1) % ny) * nx + ((i + 1) % nx)] + (px if i == nx - 1 else 0.0) + (py if j == ny - 1 else 0.0)
            for a, b, c in ((p00, p10, p11), (p00, p11, p01)):
                dets.append(float(np.cross(b - a, c - a)))
    return np.asarray(dets, dtype=np.float64)
