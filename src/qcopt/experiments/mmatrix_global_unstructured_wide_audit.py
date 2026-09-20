"""Wide-ring positive M-matrix decoder on a realistic unstructured mesh.

The experiment keeps the original Delaunay faces for the topology audit but
allows the decoder graph to use one- or two-ring directions.  This tests
whether enlarging the positive stencil improves the local Beltrami tensor cone
without silently relaxing the hard positive-weight solve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from qcopt.forward.mmatrix import beltrami_conductivity, edge_direction_conductances

from .mmatrix_global_unstructured_audit import _orient_faces, _square_points


def _one_ring_graph(faces: np.ndarray, n: int) -> list[set[int]]:
    graph = [set() for _ in range(n)]
    for face in faces:
        a, b, c = (int(v) for v in face)
        graph[a].update((b, c))
        graph[b].update((a, c))
        graph[c].update((a, b))
    return graph


def _ring_vertices(graph: list[set[int]], vertex: int, rings: int) -> list[int]:
    visited = {vertex}
    frontier = {vertex}
    for _ in range(rings):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(graph[node])
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    return sorted(visited - {vertex})


def _mu_field(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    amplitude = 0.72 * np.tanh(1.4 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y))
    phase = 0.8 * np.sin(2.0 * np.pi * y) + 0.25 * np.cos(2.0 * np.pi * x)
    return amplitude * np.exp(1j * phase)


def _decode(
    points: np.ndarray,
    faces: np.ndarray,
    boundary: np.ndarray,
    graph: list[set[int]],
    rings: int,
    weight_floor: float,
) -> dict:
    n = len(points)
    boundary_set = set(int(v) for v in boundary)
    interior_vertices = [v for v in range(n) if v not in boundary_set]
    local_weights: dict[int, dict[int, float]] = {}
    residuals: list[float] = []
    support_sizes: list[int] = []
    coefficients = _mu_field(points)
    started = time.perf_counter()
    for vertex in range(n):
        neighbors = _ring_vertices(graph, vertex, rings)
        support_sizes.append(len(neighbors))
        if len(neighbors) < 3:
            local_weights[vertex] = {neighbor: 1.0 for neighbor in neighbors}
            continue
        directions = points[neighbors] - points[vertex]
        fit = edge_direction_conductances(beltrami_conductivity(complex(coefficients[vertex])), directions)
        local_weights[vertex] = {
            neighbor: float(max(value, weight_floor))
            for neighbor, value in zip(neighbors, fit.values)
        }
        residuals.append(fit.residual)
    fit_seconds = time.perf_counter() - started

    edges: set[tuple[int, int]] = set()
    for vertex in range(n):
        for neighbor in local_weights[vertex]:
            edge = (vertex, neighbor) if vertex < neighbor else (neighbor, vertex)
            if edge[0] != edge[1]:
                edges.add(edge)
    edge_weights = {
        edge: max(weight_floor, 0.5 * (local_weights[edge[0]].get(edge[1], weight_floor) + local_weights[edge[1]].get(edge[0], weight_floor)))
        for edge in edges
    }
    unknown = {vertex: row for row, vertex in enumerate(interior_vertices)}
    matrix = lil_matrix((len(interior_vertices), len(interior_vertices)), dtype=np.float64)
    rhs = np.zeros((len(interior_vertices), 2), dtype=np.float64)
    for vertex in interior_vertices:
        row = unknown[vertex]
        neighbors = local_weights[vertex]
        total = sum(edge_weights[(min(vertex, nb), max(vertex, nb))] for nb in neighbors)
        matrix[row, row] = 1.0
        for neighbor in neighbors:
            weight = edge_weights[(min(vertex, neighbor), max(vertex, neighbor))] / total
            if neighbor in unknown:
                matrix[row, unknown[neighbor]] -= weight
            else:
                rhs[row] += weight * points[neighbor]
    solve_started = time.perf_counter()
    solved = spsolve(matrix.tocsr(), rhs)
    solve_seconds = time.perf_counter() - solve_started
    mapped = points.copy()
    mapped[np.asarray(interior_vertices)] = solved
    target_cross = np.cross(mapped[faces[:, 1]] - mapped[faces[:, 0]], mapped[faces[:, 2]] - mapped[faces[:, 0]])
    return {
        "rings": rings,
        "vertices": n,
        "boundary_vertices": len(boundary),
        "interior_vertices": len(interior_vertices),
        "triangles": int(len(faces)),
        "decoder_edges": len(edges),
        "mean_support": float(np.mean(support_sizes)),
        "p95_support": float(np.quantile(support_sizes, 0.95)),
        "fit_residual_mean": float(np.mean(residuals)),
        "fit_residual_p95": float(np.quantile(residuals, 0.95)),
        "edge_weight_min": float(min(edge_weights.values())),
        "edge_weight_max": float(max(edge_weights.values())),
        "fit_seconds": fit_seconds,
        "solve_seconds": solve_seconds,
        "min_target_face_determinant": float(np.min(target_cross)),
        "flipped_faces": int(np.sum(target_cross <= 0.0)),
        "boundary_exact_max_error": float(np.max(np.abs(mapped[boundary] - points[boundary]))),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(target_cross))),
    }


def run(output_dir: Path, interior: int = 12000, side: int = 96, weight_floor: float = 1e-4, rings: tuple[int, ...] = (1, 2)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    points, boundary = _square_points(rng, interior, side)
    faces = _orient_faces(points, Delaunay(points).simplices)
    graph = _one_ring_graph(faces, len(points))
    records = [_decode(points, faces, boundary, graph, int(ring), weight_floor) for ring in rings]
    result = {
        "schema": "forward-beltrami-mmatrix-unstructured-wide-v1",
        "interior_requested": interior,
        "boundary_side": side,
        "records": records,
        "scope": "realistic unstructured Delaunay positive-weight decoder with one-/two-ring direction dictionaries",
        "interpretation": "Increasing rings enlarges the positive edge-direction cone; residual and original-face flip audits measure whether this improves QC tensor fidelity without losing hard orientation evidence.",
        "limitation": "Wide decoder graphs are not planar Delaunay graphs, so positive row convexity alone is not a global Tutte theorem; this remains a numerical cone/decoder audit.",
    }
    (output_dir / "mmatrix_global_unstructured_wide_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interior", type=int, default=12000)
    parser.add_argument("--side", type=int, default=96)
    parser.add_argument("--weight-floor", type=float, default=1e-4)
    parser.add_argument("--rings", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.interior, args.side, args.weight_floor, tuple(args.rings)), indent=2))


if __name__ == "__main__":
    main()
