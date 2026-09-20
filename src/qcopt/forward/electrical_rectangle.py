"""Exact isotropic electrical rectangle reference on a rectangular grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class ElectricalRectangleResult:
    primal_potential: np.ndarray
    dual_potential: np.ndarray
    modulus: float
    minimum_cell_width: float
    minimum_cell_height: float
    tiling_area: float
    target_area: float


def solve_isotropic_electrical_rectangle(
    nx: int, ny: int, *, conductance: float = 1.0
) -> ElectricalRectangleResult:
    """Solve the unit-conductance grid and integrate its dual potential.

    The graph has nx by ny primal vertices, left/right Dirichlet values 0/1,
    and natural top/bottom boundaries.  This is the exact mu=0 unit test; it is
    not an anisotropic solver.
    """

    if nx < 2 or ny < 2 or conductance <= 0.0:
        raise ValueError("nx, ny must be at least two and conductance positive")
    n_vertices = nx * ny
    laplacian = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            vertex = j * nx + i
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii >= nx or jj >= ny:
                    continue
                neighbor = jj * nx + ii
                laplacian[vertex, vertex] += conductance
                laplacian[neighbor, neighbor] += conductance
                laplacian[vertex, neighbor] -= conductance
                laplacian[neighbor, vertex] -= conductance
    laplacian = laplacian.tocsr()

    left = np.arange(0, n_vertices, nx)
    right = np.arange(nx - 1, n_vertices, nx)
    fixed = np.concatenate((left, right))
    values = np.zeros(n_vertices, dtype=np.float64)
    values[right] = 1.0
    free = np.setdiff1d(np.arange(n_vertices), fixed)
    values[free] = spsolve(
        laplacian[free][:, free],
        -laplacian[free][:, fixed] @ values[fixed],
    )
    primal = values.reshape(ny, nx)
    horizontal_current = conductance * np.diff(primal, axis=1)
    if np.max(np.abs(horizontal_current - horizontal_current.mean())) > 1e-10:
        raise ValueError("the isotropic grid current is not path-independent")

    dual = np.zeros((ny, nx - 1), dtype=np.float64)
    mean_current = float(horizontal_current.mean())
    for row in range(1, ny):
        dual[row] = dual[row - 1] + mean_current
    cell_width = horizontal_current[:-1]
    cell_height = np.diff(dual, axis=0)
    modulus = float(dual[-1, 0])
    tiling_area = float(np.sum(cell_width * cell_height))
    return ElectricalRectangleResult(
        primal_potential=primal,
        dual_potential=dual,
        modulus=modulus,
        minimum_cell_width=float(np.min(cell_width)),
        minimum_cell_height=float(np.min(cell_height)),
        tiling_area=tiling_area,
        target_area=modulus,
    )
