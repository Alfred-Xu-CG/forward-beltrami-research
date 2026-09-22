"""Rank of a physically broad edge update is not latent parameter count."""

from __future__ import annotations

import numpy as np

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def _patch_incidence_rank(side: int, x0: int, y0: int, patch_cells: int) -> int:
    reference = MatrixFreeSymmetricTutteLayer(structured_rectangle(side - 1, side - 1))
    edges = reference.active_edges
    interior = reference.interior_vertices
    interior_index = np.full(side**2, -1, dtype=np.int64)
    interior_index[interior] = np.arange(len(interior))
    interior_vertices_in_patch = [
        vertex for vertex in interior
        if x0 <= vertex % side <= x0 + patch_cells and y0 <= vertex // side <= y0 + patch_cells
    ]
    reduced_col = {int(vertex): i for i, vertex in enumerate(interior_vertices_in_patch)}
    rows = []
    for a, b in edges:
        ax, ay = int(a % side), int(a // side)
        bx, by = int(b % side), int(b // side)
        if not (x0 <= ax <= x0 + patch_cells and y0 <= ay <= y0 + patch_cells):
            continue
        if not (x0 <= bx <= x0 + patch_cells and y0 <= by <= y0 + patch_cells):
            continue
        row = np.zeros(len(interior_vertices_in_patch))
        if int(a) in reduced_col:
            row[reduced_col[int(a)]] += 1.0
        if int(b) in reduced_col:
            row[reduced_col[int(b)]] -= 1.0
        rows.append(row)
    return int(np.linalg.matrix_rank(np.stack(rows)))


def test_one_interior_patch_latent_produces_many_matrix_rank_directions() -> None:
    # A 4-by-4-cell interior patch has 25 vertices and no Dirichlet vertex.
    assert _patch_incidence_rank(17, 4, 4, 4) == 24


def test_boundary_patch_rank_counts_only_unfixed_vertices() -> None:
    # The 5 vertices on x=0 are fixed, leaving 20 interior unknowns.
    assert _patch_incidence_rank(17, 0, 4, 4) == 20
