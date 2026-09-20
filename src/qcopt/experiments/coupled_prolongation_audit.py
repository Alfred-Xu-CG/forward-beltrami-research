"""High-resolution injective coarse-to-fine coupled decoder audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.coupled_decoder import coupled_monotone_shear_inverse, coupled_monotone_shear_map
from qcopt.forward.prolongation import prolongate_positive_increments
from qcopt.mesh import structured_rectangle


def _face_dets(mesh, values):
    p0, p1, p2 = (values[mesh.faces[:, i]] for i in range(3))
    return (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for coarse_n, fine_n in ((32, 512), (64, 1024)):
        coarse_x = np.exp(0.35 * np.sin(2.0 * np.pi * (np.arange(coarse_n) + 0.5) / coarse_n))
        coarse_y = np.exp(0.25 * np.cos(2.0 * np.pi * (np.arange(coarse_n) + 0.5) / coarse_n))
        fine_x = prolongate_positive_increments(coarse_x, fine_n)
        fine_y = prolongate_positive_increments(coarse_y, fine_n)
        mesh = structured_rectangle(fine_n, fine_n)
        alpha, beta = 0.22, -0.17
        t0 = time.perf_counter()
        mapped = coupled_monotone_shear_map(mesh.vertices, fine_x, fine_y, alpha, beta)
        inverse = coupled_monotone_shear_inverse(mapped, fine_x, fine_y, alpha, beta)
        det = _face_dets(mesh, mapped)
        records.append(
            {
                "coarse_intervals": coarse_n,
                "fine_intervals": fine_n,
                "fine_vertices": int(mesh.n_vertices),
                "fine_faces": int(mesh.n_faces),
                "elapsed_seconds": time.perf_counter() - t0,
                "min_determinant": float(np.min(det)),
                "flipped_faces": int(np.sum(det <= 0.0)),
                "max_roundtrip_error": float(np.max(np.linalg.norm(inverse - mesh.vertices, axis=1))),
                "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(inverse))),
            }
        )
    result = {
        "records": records,
        "scope": "positive-increment prolongation followed by a coupled determinant-one shear",
        "limitation": "the coupling is affine and the result is not an arbitrary spatially varying QC decoder; no diffusion training is included",
    }
    (output_dir / "coupled_prolongation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
