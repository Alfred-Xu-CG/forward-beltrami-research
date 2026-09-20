"""Global positive-weight unstructured harmonic decoder audit."""

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


def _square_points(rng: np.random.Generator, interior: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, side, endpoint=False)
    boundary = np.vstack(
        [
            np.column_stack((t, np.zeros_like(t))),
            np.column_stack((np.ones_like(t), t)),
            np.column_stack((1.0 - t, np.ones_like(t))),
            np.column_stack((np.zeros_like(t), 1.0 - t)),
        ]
    )
    points = np.vstack((boundary, 0.05 + 0.90 * rng.random((interior, 2))))
    return points, np.arange(len(boundary), dtype=np.int64)


def _orient_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    oriented = np.asarray(faces, dtype=np.int64).copy()
    cross = np.cross(points[oriented[:, 1]] - points[oriented[:, 0]], points[oriented[:, 2]] - points[oriented[:, 0]])
    oriented[cross < 0.0, 1], oriented[cross < 0.0, 2] = (
        oriented[cross < 0.0, 2],
        oriented[cross < 0.0, 1],
    )
    return oriented


def run(output_dir: Path, interior: int = 3800, side: int = 64, weight_floor: float = 1e-4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    points, boundary = _square_points(rng, interior, side)
    triangulation = Delaunay(points)
    faces = _orient_faces(points, triangulation.simplices)
    n = len(points)
    graph = [set() for _ in range(n)]
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            u, v = (a, b) if a < b else (b, a)
            edges.add((u, v))
            graph[a].add(b)
            graph[b].add(a)
    boundary_set = set(int(v) for v in boundary)
    interior_vertices = [v for v in range(n) if v not in boundary_set]
    # Fit a positive local edge cone at each vertex, then symmetrize each graph
    # edge. The solve itself only uses these strictly positive edge weights.
    local_weights: dict[int, dict[int, float]] = {}
    fit_residuals = []
    for vertex in range(n):
        neighbors = sorted(graph[vertex])
        directions = points[neighbors] - points[vertex]
        if directions.shape[0] < 3:
            # Convex-hull corner vertices can have only two incident graph
            # directions; their rows are Dirichlet and do not need an
            # anisotropic cone fit. Keep strictly positive fallback edge
            # weights so adjacent interior rows remain well-defined.
            local_weights[vertex] = {neighbor: 1.0 for neighbor in neighbors}
            continue
        x, y = points[vertex]
        mu = 0.72 * np.tanh(1.4 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)) * np.exp(
            1j * (0.8 * np.sin(2.0 * np.pi * y) + 0.25 * np.cos(2.0 * np.pi * x))
        )
        fit = edge_direction_conductances(beltrami_conductivity(complex(mu)), directions)
        local_weights[vertex] = {neighbor: float(max(weight, weight_floor)) for neighbor, weight in zip(neighbors, fit.values)}
        fit_residuals.append(fit.residual)
    edge_weights = {
        edge: max(weight_floor, 0.5 * (local_weights[edge[0]][edge[1]] + local_weights[edge[1]][edge[0]]))
        for edge in edges
    }
    unknown = {vertex: row for row, vertex in enumerate(interior_vertices)}
    matrix = lil_matrix((len(interior_vertices), len(interior_vertices)), dtype=np.float64)
    rhs = np.zeros((len(interior_vertices), 2), dtype=np.float64)
    for vertex in interior_vertices:
        row = unknown[vertex]
        neighbors = sorted(graph[vertex])
        total = sum(edge_weights[(min(vertex, nb), max(vertex, nb))] for nb in neighbors)
        matrix[row, row] = 1.0
        for nb in neighbors:
            weight = edge_weights[(min(vertex, nb), max(vertex, nb))] / total
            if nb in unknown:
                matrix[row, unknown[nb]] -= weight
            else:
                rhs[row] += weight * points[nb]
    solved_interior = spsolve(matrix.tocsr(), rhs)
    mapped = points.copy()
    mapped[np.asarray(interior_vertices)] = solved_interior
    target_cross = np.cross(mapped[faces[:, 1]] - mapped[faces[:, 0]], mapped[faces[:, 2]] - mapped[faces[:, 0]])
    result = {
        "vertices": n,
        "boundary_vertices": len(boundary),
        "interior_vertices": len(interior_vertices),
        "triangles": int(len(faces)),
        "edges": len(edges),
        "mean_degree": float(np.mean([len(g) for g in graph])),
        "fit_residual_mean": float(np.mean(fit_residuals)),
        "fit_residual_p95": float(np.quantile(fit_residuals, 0.95)),
        "edge_weight_min": float(min(edge_weights.values())),
        "edge_weight_max": float(max(edge_weights.values())),
        "edge_weight_floor": weight_floor,
        "min_target_face_determinant": float(np.min(target_cross)),
        "flipped_faces": int(np.sum(target_cross <= 0.0)),
        "boundary_exact_max_error": float(np.max(np.abs(mapped[boundary] - points[boundary]))),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(target_cross))),
        "scope": "globally assembled positive-edge unstructured harmonic decoder with convex square boundary",
        "interpretation": "global positivity and convex boundary give a hard-injective harmonic control, while local Beltrami tensor fit residual quantifies anisotropic inconsistency",
        "limitation": "this is not an exact arbitrary-mu discretization; edge-cone consistency, approximation order, and a general QC decoder remain open",
    }
    (output_dir / "mmatrix_global_unstructured_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interior", type=int, default=3800)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--weight-floor", type=float, default=1e-4)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(args.output_dir, args.interior, args.side, args.weight_floor)
    result["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "mmatrix_global_unstructured_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
