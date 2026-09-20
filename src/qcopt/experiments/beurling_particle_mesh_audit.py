"""Realistic-resolution particle-mesh Beurling accuracy/cost audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_particle_mesh import particle_mesh_beurling_apply


def run(output_dir: Path, n: int = 1024) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    offset_x, offset_y = 0.23, 0.37
    axis = (np.arange(n, dtype=np.float64) + offset_x) / n
    axis_y = (np.arange(n, dtype=np.float64) + offset_y) / n
    xx, yy = np.meshgrid(axis, axis_y, indexing="xy")
    points = (xx + 1j * yy).ravel()
    kx, ky = 3, -2
    values = np.exp(2j * np.pi * (kx * points.real + ky * points.imag))
    weights = np.full(points.shape, 1.0 / points.size)
    expected = ((kx - 1j * ky) / (kx + 1j * ky)) * values
    records: list[dict] = []
    for grid_n in (max(16, n // 4), max(32, n // 2), n):
        t0 = time.perf_counter()
        result = particle_mesh_beurling_apply(
            points, values, weights, grid_shape=(grid_n, grid_n)
        )
        t1 = time.perf_counter()
        corrected = particle_mesh_beurling_apply(
            points,
            values,
            weights,
            grid_shape=(grid_n, grid_n),
            deconvolve=True,
        )
        t2 = time.perf_counter()
        cubic = particle_mesh_beurling_apply(
            points,
            values,
            weights,
            grid_shape=(grid_n, grid_n),
            deconvolve=True,
            kernel="cubic",
        )
        records.append(
            {
                "grid_n": grid_n,
                "elapsed_seconds": time.perf_counter() - t0,
                "mean_abs_error": float(np.mean(np.abs(result - expected))),
                "max_abs_error": float(np.max(np.abs(result - expected))),
                "deconvolved_elapsed_seconds": time.perf_counter() - t1,
                "deconvolved_mean_abs_error": float(np.mean(np.abs(corrected - expected))),
                "deconvolved_max_abs_error": float(np.max(np.abs(corrected - expected))),
                "cubic_deconvolved_elapsed_seconds": time.perf_counter() - t2,
                "cubic_deconvolved_mean_abs_error": float(np.mean(np.abs(cubic - expected))),
                "cubic_deconvolved_max_abs_error": float(np.max(np.abs(cubic - expected))),
            }
        )
    result = {
        "particle_points": int(points.size),
        "records": records,
        "truth": "periodic Fourier mode with exact Beurling multiplier",
        "scope": "CIC/cubic B-spline particle-mesh approximation with window deconvolution; not a proof for arbitrary singular data",
    }
    (output_dir / "beurling_particle_mesh_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
