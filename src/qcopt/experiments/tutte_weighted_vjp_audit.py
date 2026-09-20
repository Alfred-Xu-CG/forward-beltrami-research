"""Realistic sparse weighted-Tutte implicit VJP audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.tutte_weighted_implicit import weighted_tutte_vjp
from qcopt.mesh import structured_rectangle


def _weights(mesh):
    edges = set()
    for face in mesh.faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edges.add(tuple(sorted((int(face[i]), int(face[j])))))
    return {edge: 0.25 + 1.75 * (1.0 + np.sin(0.17 * edge[0] + 0.11 * edge[1])) / 2.0 for edge in edges}


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 256
    mesh = structured_rectangle(n, n)
    loop = mesh.boundary_loops[0]
    theta = np.linspace(0.0, 2.0 * np.pi, len(loop), endpoint=False)
    target = np.column_stack((np.cos(theta), np.sin(theta)))
    weights = _weights(mesh)
    rng = np.random.default_rng(31)
    gradient = rng.normal(size=(mesh.n_vertices, 2))
    t0 = time.perf_counter()
    result, edge_gradient = weighted_tutte_vjp(mesh, target, weights, gradient)
    elapsed = time.perf_counter() - t0
    values = np.asarray(list(edge_gradient.values()))
    audit = {
        "grid_cells_per_axis": n,
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "edges": len(weights),
        "elapsed_seconds": elapsed,
        "min_output_norm": float(np.min(np.linalg.norm(result, axis=1))),
        "max_abs_edge_vjp": float(np.max(np.abs(values))),
        "finite": bool(np.all(np.isfinite(result)) and np.all(np.isfinite(values))),
        "scope": "positive weighted Tutte solve plus sparse transpose VJP with respect to edge weights",
        "limitation": "the VJP is a local differentiability certificate; it does not make arbitrary signed weights safe",
    }
    (output_dir / "tutte_weighted_vjp_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

