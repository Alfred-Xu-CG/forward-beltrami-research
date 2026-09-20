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
    graph_energy: float
    right_flux: float
    minimum_cell_width: float
    minimum_cell_height: float
    tiling_area: float


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
    horizontal = np.full((ny, nx - 1), float(conductance), dtype=np.float64)
    vertical = np.full((ny - 1, nx), float(conductance), dtype=np.float64)
    return solve_weighted_electrical_rectangle(horizontal, vertical)


def solve_weighted_electrical_rectangle(
    horizontal_conductance: np.ndarray,
    vertical_conductance: np.ndarray,
) -> ElectricalRectangleResult:
    """Solve a positive rectangular conductance grid and audit its tiling.

    The horizontal array has shape ``(ny, nx-1)`` and the vertical array has
    shape ``(ny-1, nx)``.  The dual-strip reconstruction is deliberately
    conservative: every horizontal row must have path-independent current. A
    general network that violates this condition is rejected rather than being
    silently presented as a rectangle tiling.
    """
    horizontal_conductance = np.asarray(horizontal_conductance, dtype=np.float64)
    vertical_conductance = np.asarray(vertical_conductance, dtype=np.float64)
    if horizontal_conductance.ndim != 2 or horizontal_conductance.shape[0] < 2 or horizontal_conductance.shape[1] < 1:
        raise ValueError("horizontal_conductance must have shape (ny, nx-1) with ny >= 2")
    ny, horizontal_edges = horizontal_conductance.shape
    nx = horizontal_edges + 1
    if vertical_conductance.shape != (ny - 1, nx):
        raise ValueError("vertical_conductance must have shape (ny-1, nx)")
    if not np.all(np.isfinite(horizontal_conductance)) or not np.all(np.isfinite(vertical_conductance)):
        raise ValueError("conductances must be finite")
    if np.min(horizontal_conductance) <= 0.0 or np.min(vertical_conductance) <= 0.0:
        raise ValueError("conductances must be positive")
    n_vertices = nx * ny
    laplacian = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            vertex = j * nx + i
            if i + 1 < nx:
                neighbor = vertex + 1
                edge_conductance = horizontal_conductance[j, i]
                laplacian[vertex, vertex] += edge_conductance
                laplacian[neighbor, neighbor] += edge_conductance
                laplacian[vertex, neighbor] -= edge_conductance
                laplacian[neighbor, vertex] -= edge_conductance
            if j + 1 < ny:
                neighbor = vertex + nx
                edge_conductance = vertical_conductance[j, i]
                laplacian[vertex, vertex] += edge_conductance
                laplacian[neighbor, neighbor] += edge_conductance
                laplacian[vertex, neighbor] -= edge_conductance
                laplacian[neighbor, vertex] -= edge_conductance
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
    horizontal_difference = np.diff(primal, axis=1)
    horizontal_current = horizontal_conductance * horizontal_difference
    row_currents = horizontal_current.mean(axis=1)
    if np.max(np.abs(horizontal_current - row_currents[:, None])) > 1e-10:
        raise ValueError("the isotropic grid current is not path-independent")
    vertical_difference = np.diff(primal, axis=0)
    vertical_current = vertical_conductance * vertical_difference
    graph_energy = float(
        np.sum(horizontal_current * horizontal_difference)
        + np.sum(vertical_current * vertical_difference)
    )
    right_flux = float(np.sum(horizontal_current[:, -1]))
    if abs(graph_energy - right_flux) > 1e-10:
        raise ValueError("graph energy and right boundary flux disagree")

    dual_levels = np.r_[0.0, np.cumsum(row_currents)]
    dual = np.repeat(dual_levels[:, None], nx - 1, axis=1)
    cell_width = horizontal_difference
    cell_height = np.repeat(row_currents[:, None], nx - 1, axis=1)
    modulus = float(dual_levels[-1])
    tiling_area = float(np.sum(cell_width * cell_height))
    return ElectricalRectangleResult(
        primal_potential=primal,
        dual_potential=dual,
        modulus=modulus,
        graph_energy=graph_energy,
        right_flux=right_flux,
        minimum_cell_width=float(np.min(cell_width)),
        minimum_cell_height=float(np.min(cell_height)),
        tiling_area=tiling_area,
    )
