"""Realistic mesh audit of the piecewise-affine BHF quadrature reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import triangle_bhf_variation


def regular_triangles(n: int) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, n + 1)
    vertices = (grid[:, None] + 1j * grid[None, :]).reshape(-1)
    faces = []
    for i in range(n):
        for j in range(n):
            p = i * (n + 1) + j
            q = (i + 1) * (n + 1) + j
            faces.extend(((p, q, p + 1), (q, q + 1, p + 1)))
    return vertices[np.asarray(faces, dtype=np.int64)]


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    b = 0.2 + 0.1j
    a = 1.0 - b
    for n, n_eval in ((64, 1024), (128, 2048)):
        tri = regular_triangles(n)
        tri_mu = 0.18 * np.exp(-np.abs(np.mean(tri, axis=1) - 0.15 - 0.2j) ** 2 / 0.5)
        tri_fz = np.full(tri.shape[0], a, dtype=np.complex128)
        tri_image = a * tri + b * np.conjugate(tri)
        rng = np.random.default_rng(700 + n)
        evaluation = (rng.random(n_eval) + 1j * rng.random(n_eval)) * 1.8 - 0.9
        evaluation_image = a * evaluation + b * np.conjugate(evaluation)
        t0 = time.perf_counter()
        velocity = triangle_bhf_variation(
            evaluation,
            evaluation_image,
            tri,
            tri_image,
            tri_fz,
            tri_mu,
            block_size=128,
        )
        records.append(
            {
                "grid_cells_per_axis": n,
                "triangles": int(tri.shape[0]),
                "quadrature_source_points": int(7 * tri.shape[0]),
                "evaluation_points": n_eval,
                "elapsed_seconds": time.perf_counter() - t0,
                "pair_evaluations": int(7 * tri.shape[0] * n_eval),
                "max_velocity": float(np.max(np.abs(velocity))),
                "finite": bool(np.all(np.isfinite(velocity))),
            }
        )
    result = {
        "records": records,
        "scope": "piecewise-affine current map with degree-five triangle quadrature",
        "limitations": [
            "direct O(Q*M) pair cost, where Q=7 times number of triangles",
            "fixed quadrature is a reference PV regularization, not exact triangle integration",
            "nonlinear BHF time integration and sphere overlap gluing remain open",
        ],
    }
    (output_dir / "bhf_triangle_mesh_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

