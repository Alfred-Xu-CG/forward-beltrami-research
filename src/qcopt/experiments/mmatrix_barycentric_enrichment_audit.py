"""Planar barycentric enrichment for positive M-matrix QC decoding.

Unlike a two-ring graph, barycentric subdivision adds directions while keeping
the decoder graph planar.  Original boundary vertices remain fixed; face-center
vertices are solved as interior unknowns with positive row weights.
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


def _mu_field(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    amplitude = 0.72 * np.tanh(1.4 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y))
    phase = 0.8 * np.sin(2.0 * np.pi * y) + 0.25 * np.cos(2.0 * np.pi * x)
    return amplitude * np.exp(1j * phase)


def run(
    output_dir: Path,
    interior: int = 8000,
    side: int = 96,
    weight_floor: float = 1e-4,
    anisotropic_centers: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    points, boundary = _square_points(rng, interior, side)
    original_faces = _orient_faces(points, Delaunay(points).simplices)
    original_vertices = len(points)
    centroids = points[original_faces].mean(axis=1)
    centroid_offset = original_vertices
    vertices = np.vstack((points, centroids))
    centroid_ids = np.arange(centroid_offset, centroid_offset + len(centroids), dtype=np.int64)
    refined_faces = np.vstack(
        (
            np.column_stack((original_faces[:, 0], original_faces[:, 1], centroid_ids)),
            np.column_stack((original_faces[:, 1], original_faces[:, 2], centroid_ids)),
            np.column_stack((original_faces[:, 2], original_faces[:, 0], centroid_ids)),
        )
    )
    # Face orientation is inherited from the oriented parent triangle.
    n_vertices = len(vertices)
    graph = [set() for _ in range(n_vertices)]
    for face in refined_faces:
        a, b, c = (int(v) for v in face)
        graph[a].update((b, c))
        graph[b].update((a, c))
        graph[c].update((a, b))
    boundary_set = set(int(v) for v in boundary)
    interior_vertices = [v for v in range(n_vertices) if v not in boundary_set]
    coefficients = _mu_field(vertices)
    local_weights: dict[int, dict[int, float]] = {}
    residuals: list[float] = []
    original_residuals: list[float] = []
    center_residuals: list[float] = []
    fit_started = time.perf_counter()
    for vertex in range(n_vertices):
        neighbors = sorted(graph[vertex])
        if vertex in boundary_set or (vertex >= centroid_offset and not anisotropic_centers) or len(neighbors) < 3:
            local_weights[vertex] = {neighbor: 1.0 for neighbor in neighbors}
            continue
        directions = vertices[neighbors] - vertices[vertex]
        fit = edge_direction_conductances(beltrami_conductivity(complex(coefficients[vertex])), directions)
        local_weights[vertex] = {
            neighbor: float(max(value, weight_floor))
            for neighbor, value in zip(neighbors, fit.values)
        }
        residuals.append(fit.residual)
        (center_residuals if vertex >= centroid_offset else original_residuals).append(fit.residual)
    fit_seconds = time.perf_counter() - fit_started
    edges: set[tuple[int, int]] = set()
    for vertex, neighbors in enumerate(local_weights):
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
                rhs[row] += weight * vertices[neighbor]
    solve_started = time.perf_counter()
    solved = spsolve(matrix.tocsr(), rhs)
    solve_seconds = time.perf_counter() - solve_started
    mapped = vertices.copy()
    mapped[np.asarray(interior_vertices)] = solved
    p0, p1, p2 = (mapped[refined_faces[:, i]] for i in range(3))
    refined_cross = np.cross(p1 - p0, p2 - p0)
    result = {
        "schema": "forward-beltrami-mmatrix-barycentric-enrichment-v1",
        "original_vertices": original_vertices,
        "face_centers": int(len(centroids)),
        "vertices": n_vertices,
        "boundary_vertices": len(boundary),
        "original_triangles": int(len(original_faces)),
        "refined_triangles": int(len(refined_faces)),
        "decoder_edges": len(edges),
        "mean_original_support": float(np.mean([len(graph[v]) for v in range(original_vertices)])),
        "fit_residual_mean": float(np.mean(residuals)),
        "fit_residual_p95": float(np.quantile(residuals, 0.95)),
        "original_fit_residual_mean": float(np.mean(original_residuals)),
        "original_fit_residual_p95": float(np.quantile(original_residuals, 0.95)),
        "center_fit_residual_mean": float(np.mean(center_residuals)) if center_residuals else 0.0,
        "center_fit_residual_p95": float(np.quantile(center_residuals, 0.95)) if center_residuals else 0.0,
        "edge_weight_min": float(min(edge_weights.values())),
        "edge_weight_max": float(max(edge_weights.values())),
        "anisotropic_centers": bool(anisotropic_centers),
        "fit_seconds": fit_seconds,
        "solve_seconds": solve_seconds,
        "min_refined_face_determinant": float(np.min(refined_cross)),
        "flipped_refined_faces": int(np.sum(refined_cross <= 0.0)),
        "boundary_exact_max_error": float(np.max(np.abs(mapped[boundary] - vertices[boundary]))),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(refined_cross))),
        "scope": "planar barycentric subdivision with positive decoder weights and fixed convex square boundary",
        "interpretation": "Planar enrichment tests whether added face-center directions improve local QC tensor cones without the crossing-induced flips of a nonplanar two-ring graph.",
        "limitation": "This is a planar hard-decoder control, not an arbitrary-mu theorem; anisotropic face-center rows can introduce their own cone-fit residual.",
    }
    (output_dir / "mmatrix_barycentric_enrichment_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interior", type=int, default=8000)
    parser.add_argument("--side", type=int, default=96)
    parser.add_argument("--weight-floor", type=float, default=1e-4)
    parser.add_argument("--anisotropic-centers", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.interior, args.side, args.weight_floor, args.anisotropic_centers), indent=2))


if __name__ == "__main__":
    main()
