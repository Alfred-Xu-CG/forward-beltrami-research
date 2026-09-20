"""Nonnegative M-matrix cone audit on an unstructured Delaunay mesh."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from qcopt.forward.mmatrix import beltrami_conductivity, edge_direction_conductances


def _adjacency(simplices: np.ndarray, n: int) -> list[set[int]]:
    graph = [set() for _ in range(n)]
    for triangle in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(triangle[i]), int(triangle[j])
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _directions(points: np.ndarray, graph: list[set[int]], vertex: int, rings: int) -> np.ndarray:
    visited = {vertex}
    frontier = {vertex}
    for _ in range(rings):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph[node])
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    neighbors = sorted(visited - {vertex})
    return points[neighbors] - points[vertex]


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(20260919)
    points = rng.random((4096, 2))
    triangulation = Delaunay(points)
    graph = _adjacency(triangulation.simplices, len(points))
    records = []
    for rings in (1, 2):
        residuals = []
        supports = []
        positive = []
        for vertex in range(len(points)):
            x, y = points[vertex]
            mu = 0.82 * np.tanh(1.3 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)) * np.exp(
                1j * (0.8 * np.sin(2.0 * np.pi * y) + 0.25 * np.cos(2.0 * np.pi * x))
            )
            directions = _directions(points, graph, vertex, rings)
            fit = edge_direction_conductances(beltrami_conductivity(complex(mu)), directions)
            residuals.append(fit.residual)
            supports.append(len(directions))
            positive.append(bool(np.all(fit.values >= -1e-12)))
        residuals_array = np.asarray(residuals)
        records.append({
            "rings": rings,
            "vertices": len(points),
            "mean_support": float(np.mean(supports)),
            "max_support": int(np.max(supports)),
            "mean_residual": float(np.mean(residuals_array)),
            "median_residual": float(np.median(residuals_array)),
            "p95_residual": float(np.quantile(residuals_array, 0.95)),
            "max_residual": float(np.max(residuals_array)),
            "all_nonnegative": bool(all(positive)),
            "exact_fit_fraction_1e-8": float(np.mean(residuals_array <= 1e-8)),
        })
    result = {
        "points": len(points),
        "triangles": int(len(triangulation.simplices)),
        "records": records,
        "scope": "nonuniform Delaunay mesh; local one-ring versus two-ring edge-direction tensor cones",
        "limitation": "cone-fit evidence, not yet a globally assembled unstructured M-matrix QC decoder or injectivity theorem",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path("D:/QC_optimization/artifacts/mmatrix_unstructured_stencil_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mmatrix_unstructured_stencil_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
