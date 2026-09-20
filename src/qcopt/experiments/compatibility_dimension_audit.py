"""Small/medium dimension audit for the realizable facewise-BC kernel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.forward.compatibility import compatibility_matrix, compatibility_nullity
from qcopt.mesh import structured_rectangle


def run(output_dir: Path, sizes: tuple[int, ...] = (4, 5, 6)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n in sizes:
        mesh = structured_rectangle(n, n)
        uv = mesh.vertices.copy()
        uv[:, 0] += 0.05 * np.sin(2.0 * np.pi * uv[:, 1])
        uv[:, 1] += 0.04 * np.sin(2.0 * np.pi * uv[:, 0])
        mu = face_beltrami(mesh, uv)
        t0 = time.perf_counter()
        matrix = compatibility_matrix(mesh, mu)
        nullity_manufactured = compatibility_nullity(matrix, tolerance=1e-8)
        elapsed = time.perf_counter() - t0
        rng = np.random.default_rng(100 + n)
        raw = rng.normal(size=(mesh.n_faces, 2))
        random_mu = 0.6 * (raw[:, 0] + 1j * raw[:, 1]) / np.maximum(
            1.0, np.linalg.norm(raw, axis=1)
        )
        random_matrix = compatibility_matrix(mesh, random_mu)
        nullity_random = compatibility_nullity(random_matrix, tolerance=1e-8)
        records.append(
            {
                "n": n,
                "vertices": mesh.n_vertices,
                "faces": mesh.n_faces,
                "shared_edge_rows": matrix.shape[0],
                "manufactured_nullity": nullity_manufactured,
                "random_nullity": nullity_random,
                "assembly_and_dense_svd_seconds": elapsed,
            }
        )
    result = {"records": records, "tolerance": 1e-8}
    (output_dir / "compatibility_dimension_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
