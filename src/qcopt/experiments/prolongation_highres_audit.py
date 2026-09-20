"""High-resolution coarse-to-fine prolongation benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.benchmarks import smooth_twist_map
from qcopt.forward.prolongation import prolongate_regular_grid
from qcopt.mesh import structured_rectangle


def _face_dets(mesh, values):
    p0, p1, p2 = (values[mesh.faces[:, i]] for i in range(3))
    return (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for coarse_n, fine_n in ((64, 512), (128, 1024)):
        coarse = structured_rectangle(coarse_n, coarse_n)
        fine = structured_rectangle(fine_n, fine_n)
        coarse_map = smooth_twist_map(coarse.vertices, amplitude=0.08)
        t0 = time.perf_counter()
        fine_map = prolongate_regular_grid(coarse_map, coarse_n, coarse_n, fine_n, fine_n)
        det = _face_dets(fine, fine_map)
        records.append({
            "coarse_grid": coarse_n,
            "fine_grid": fine_n,
            "coarse_vertices": coarse.n_vertices,
            "fine_vertices": fine.n_vertices,
            "fine_faces": fine.n_faces,
            "elapsed_seconds": time.perf_counter() - t0,
            "min_determinant": float(np.min(det)),
            "flipped_faces": int(np.sum(det <= 0.0)),
            "finite": bool(np.all(np.isfinite(fine_map))),
        })
    result = {
        "records": records,
        "scope": "bilinear coarse-to-fine prolongation of a smooth positive map",
        "limitation": "positive coarse faces alone do not prevent fine-grid folds; this audit uses an independent fine-grid determinant check",
    }
    (output_dir / "prolongation_highres_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

