"""Realistic positive nonuniform Tutte-weight audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.tutte import tutte_embedding_weighted
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def run(output_dir: Path, nx: int = 512, ny: int = 512) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(nx, ny)
    loop = mesh.boundary_loops[0]
    target = 1.15 * mesh.vertices[loop] + np.asarray([0.12, -0.04])
    edges: set[tuple[int, int]] = set()
    for a, b, c in mesh.faces.tolist():
        edges.update((min(u, v), max(u, v)) for u, v in ((a, b), (b, c), (c, a)))
    weights = {
        edge: float(np.exp(-1.5 * np.linalg.norm(mesh.vertices[edge[0]] - mesh.vertices[edge[1]])))
        for edge in edges
    }
    t0 = time.perf_counter()
    embedding = tutte_embedding_weighted(mesh, target, weights)
    elapsed = time.perf_counter() - t0
    report = audit_injectivity(mesh, embedding)
    p0, p1, p2 = (embedding[mesh.faces[:, i]] for i in range(3))
    determinants = np.linalg.det(np.stack((p1 - p0, p2 - p0), axis=-1))
    result = {
        "nx": nx,
        "ny": ny,
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "edges": len(edges),
        "elapsed_seconds": elapsed,
        "min_face_determinant": float(np.min(determinants)),
        "flipped_faces": int(np.count_nonzero(determinants <= 0.0)),
        "injectivity_certified": report.certified,
    }
    (output_dir / "tutte_weighted_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.nx, args.ny), indent=2))


if __name__ == "__main__":
    main()
