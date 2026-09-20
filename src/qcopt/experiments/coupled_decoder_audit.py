"""High-resolution coupled monotone/shear decoder audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.coupled_decoder import coupled_monotone_shear_inverse, coupled_monotone_shear_map


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 1024
    axis = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    dx = np.array([0.3, 0.8, 1.5, 0.6, 1.2, 0.45, 0.9, 1.4])
    dy = np.array([1.1, 0.4, 0.9, 1.8, 0.7, 1.3, 0.5, 1.6])
    t0 = time.perf_counter()
    mapped = coupled_monotone_shear_map(points, dx, dy, 0.17, -0.11)
    recovered = coupled_monotone_shear_inverse(mapped, dx, dy, 0.17, -0.11)
    result = {
        "grid": f"{n}x{n}",
        "vertices": int(points.shape[0]),
        "elapsed_seconds": time.perf_counter() - t0,
        "max_roundtrip_error": float(np.max(np.abs(recovered - points))),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(recovered))),
        "scope": "separable positive monotone map composed with determinant-one affine shear",
        "limitation": "affine shear remains a restricted coupling; this is not a general spatially varying QC decoder",
    }
    (output_dir / "coupled_decoder_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

