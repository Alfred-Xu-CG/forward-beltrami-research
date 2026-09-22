"""Incidence-rank audit for distributed positive conductance updates."""

from __future__ import annotations

import json

import numpy as np

from qcopt.mesh import structured_rectangle


def _rank_and_graph_formula(edges: np.ndarray, interior: np.ndarray) -> tuple[int, int, int]:
    used = np.unique(edges)
    parent = {int(v): int(v) for v in used}

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for a, b in edges:
        parent[root(int(a))] = root(int(b))
    groups: dict[int, list[int]] = {}
    for vertex in used:
        groups.setdefault(root(int(vertex)), []).append(int(vertex))
    interior_set = set(int(v) for v in interior)
    expected = 0
    for group in groups.values():
        inside = sum(v in interior_set for v in group)
        touching_boundary = inside < len(group)
        expected += inside if touching_boundary else max(inside - 1, 0)
    rows = {int(vertex): index for index, vertex in enumerate(interior)}
    incidence = np.zeros((len(interior), len(edges)), dtype=np.float64)
    for column, (a, b) in enumerate(edges):
        if int(a) in rows:
            incidence[rows[int(a)], column] = 1.0
        if int(b) in rows:
            incidence[rows[int(b)], column] = -1.0
    numeric = int(np.linalg.matrix_rank(incidence))
    return numeric, expected, len(groups)


def main() -> None:
    side = 17
    mesh = structured_rectangle(side - 1, side - 1)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]])), axis=1), axis=0)
    index = np.arange(side**2)
    x, y = index % side, index // side
    interior = index[(x > 0) & (x < side - 1) & (y > 0) & (y < side - 1)]
    active = edges[np.isin(edges[:, 0], interior) | np.isin(edges[:, 1], interior)]
    patch = active[
        ((active[:, 0] % side >= 6) & (active[:, 0] % side <= 10)
         & (active[:, 0] // side >= 6) & (active[:, 0] // side <= 10)
         & (active[:, 1] % side >= 6) & (active[:, 1] % side <= 10)
         & (active[:, 1] // side >= 6) & (active[:, 1] // side <= 10))
    ]
    cut = active[
        ((active[:, 0] // side == 8) & (active[:, 1] // side == 9)
         & (active[:, 1] % side == active[:, 0] % side + 1))
    ]
    sets = {"one_edge": active[:1], "sixteen_edge_cut": cut, "central_5x5_patch": patch, "all_active_edges_one_scalar": active}
    results = {}
    for name, selected in sets.items():
        numeric, formula, components = _rank_and_graph_formula(selected, interior)
        results[name] = {
            "changed_edges": len(selected),
            "affected_vertices": len(np.unique(selected)),
            "support_components": components,
            "numeric_dirichlet_update_rank": numeric,
            "graph_formula_rank": formula,
        }
    print(json.dumps({"control_side": side, "interior_vertices": len(interior), "cases": results}, sort_keys=True))


if __name__ == "__main__":
    main()
