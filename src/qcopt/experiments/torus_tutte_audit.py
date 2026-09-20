"""High-resolution positive-weight periodic Tutte embedding audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.torus_tutte import periodic_torus_face_determinants, periodic_tutte_embedding


def run(output_dir: Path, n: int = 512) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    i = np.arange(n, dtype=np.float64)[None, :]
    j = np.arange(n, dtype=np.float64)[:, None]
    wx = 1.0 + 0.35 * np.sin(2.0 * np.pi * (i + 0.31 * j) / n) + 0.08 * np.cos(4.0 * np.pi * j / n)
    wy = 1.0 + 0.35 * np.cos(2.0 * np.pi * (j + 0.23 * i) / n) + 0.08 * np.sin(4.0 * np.pi * i / n)
    t0 = time.perf_counter()
    values = periodic_tutte_embedding(n, n, wx, wy)
    elapsed = time.perf_counter() - t0
    det = periodic_torus_face_determinants(values, n, n)
    result = {
        "grid": f"{n}x{n} periodic vertices",
        "vertices": n * n,
        "periodic_faces": 2 * n * n,
        "elapsed_seconds": elapsed,
        "min_lifted_face_determinant": float(np.min(det)),
        "max_lifted_face_determinant": float(np.max(det)),
        "flipped_faces": int(np.sum(det <= 0.0)),
        "finite": bool(np.all(np.isfinite(values)) and np.all(np.isfinite(det))),
        "weight_min": float(min(np.min(wx), np.min(wy))),
        "weight_max": float(max(np.max(wx), np.max(wy))),
        "scope": "positive-weight quasi-periodic harmonic torus embedding",
        "limitation": "this is a rectangular periodic graph theorem/control, not arbitrary surface triangulation or spatially varying Beltrami inversion",
    }
    (output_dir / "torus_tutte_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()

