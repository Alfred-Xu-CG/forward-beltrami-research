"""Dual-holonomy and primal edge-closure audit for facewise Beltrami fields."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.mesh import structured_rectangle


def _edge_records(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    records: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(faces.tolist()):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(start, end), max(start, end))
            records.setdefault(key, []).append((face_index, start, end))
    return records


def _propagate_face_scales(vertices: np.ndarray, faces: np.ndarray, mu: np.ndarray):
    edges = _edge_records(faces)
    adjacency: list[list[tuple[int, complex]]] = [[] for _ in range(len(faces))]
    for records in edges.values():
        if len(records) != 2:
            continue
        first, second = records
        e = complex(*(vertices[first[2]] - vertices[first[1]]))
        t, u = first[0], second[0]
        ratio_t_to_u = (e + mu[t] * np.conjugate(e)) / (e + mu[u] * np.conjugate(e))
        adjacency[t].append((u, ratio_t_to_u))
        adjacency[u].append((t, 1.0 / ratio_t_to_u))
    scales = np.full(len(faces), np.nan + 1j * np.nan, dtype=np.complex128)
    scales[0] = 1.0 + 0.0j
    queue: deque[int] = deque([0])
    cycle_residuals: list[float] = []
    while queue:
        face = queue.popleft()
        for neighbor, ratio in adjacency[face]:
            candidate = scales[face] * ratio
            if np.isnan(scales[neighbor].real):
                scales[neighbor] = candidate
                queue.append(neighbor)
            else:
                cycle_residuals.append(float(abs(candidate - scales[neighbor]) / max(abs(scales[neighbor]), 1e-15)))
    return scales, np.asarray(cycle_residuals, dtype=np.float64), edges


def _integrate_edges(vertices: np.ndarray, faces: np.ndarray, mu: np.ndarray, scales: np.ndarray, edges):
    graph: list[list[tuple[int, complex]]] = [[] for _ in range(len(vertices))]
    for (u, v), records in edges.items():
        face = records[0][0]
        e = complex(*(vertices[v] - vertices[u]))
        image_edge = scales[face] * (e + mu[face] * np.conjugate(e))
        graph[u].append((v, image_edge))
        graph[v].append((u, -image_edge))
    recovered = np.full(len(vertices), np.nan + 1j * np.nan, dtype=np.complex128)
    recovered[0] = 0.0j
    queue: deque[int] = deque([0])
    closure: list[float] = []
    while queue:
        vertex = queue.popleft()
        for neighbor, edge in graph[vertex]:
            candidate = recovered[vertex] + edge
            if np.isnan(recovered[neighbor].real):
                recovered[neighbor] = candidate
                queue.append(neighbor)
            else:
                closure.append(float(abs(candidate - recovered[neighbor])))
    return recovered, np.asarray(closure, dtype=np.float64)


def _align_error(recovered: np.ndarray, target: np.ndarray) -> float:
    z = recovered
    w = target[:, 0] + 1j * target[:, 1]
    zc = z - np.mean(z)
    wc = w - np.mean(w)
    alpha = np.sum(zc * np.conjugate(wc)) / max(np.sum(np.abs(wc) ** 2), 1e-15)
    beta = np.mean(z) - alpha * np.mean(w)
    return float(np.linalg.norm(z - (alpha * w + beta)) / max(np.linalg.norm(z), 1e-15))


def _case(mesh, target: np.ndarray, mu: np.ndarray) -> dict:
    vertices = mesh.vertices
    scales, holonomy, edges = _propagate_face_scales(vertices, mesh.faces, mu)
    recovered, closure = _integrate_edges(vertices, mesh.faces, mu, scales, edges)
    return {
        "mu_max_abs": float(np.max(np.abs(mu))),
        "face_scale_finite_fraction": float(np.mean(np.isfinite(scales))),
        "dual_cycle_count": int(len(holonomy)),
        "dual_holonomy_max_relative": float(np.max(holonomy)) if len(holonomy) else 0.0,
        "dual_holonomy_p95_relative": float(np.quantile(holonomy, 0.95)) if len(holonomy) else 0.0,
        "primal_edge_cycle_count": int(len(closure)),
        "primal_edge_closure_max": float(np.max(closure)) if len(closure) else 0.0,
        "primal_edge_closure_p95": float(np.quantile(closure, 0.95)) if len(closure) else 0.0,
        "aligned_reconstruction_relative_l2": _align_error(recovered, target),
        "finite": bool(np.all(np.isfinite(recovered))),
    }


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh = structured_rectangle(n, n)
    target = mesh.vertices.copy()
    target[:, 0] += 0.06 * np.sin(2.0 * np.pi * target[:, 1]) * np.sin(np.pi * target[:, 0])
    target[:, 1] += 0.05 * np.sin(2.0 * np.pi * target[:, 0]) * np.sin(np.pi * target[:, 1])
    manufactured = face_beltrami(mesh, target)
    rng = np.random.default_rng(20260919)
    raw = rng.normal(size=(mesh.n_faces, 2))
    random_mu = 0.62 * (raw[:, 0] + 1j * raw[:, 1]) / np.maximum(1.0, np.linalg.norm(raw, axis=1))
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "manufactured_map": _case(mesh, target, manufactured),
        "random_bounded_mu": _case(mesh, target, random_mu),
        "scope": "dual-edge holonomy propagation and primal image-edge reconstruction for compatible PL Beltrami fields",
        "interpretation": "a realizable facewise field has near-zero cycle holonomy and reconstructs a map up to complex similarity; bounded random fields need not satisfy these constraints",
        "limitation": "numerical characterization rather than a formal holonomy theorem; rectangle boundary submanifold and projection differentiation remain open",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "holonomy_reconstruction_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
