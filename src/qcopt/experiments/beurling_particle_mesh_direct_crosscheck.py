"""Independent direct-quadrature cross-check for particle-mesh Beurling gridding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_particle_mesh import particle_mesh_beurling_apply


def run(output_dir: Path, points_count: int = 4096) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    side = int(round(np.sqrt(points_count)))
    if side * side != points_count:
        raise ValueError("points_count must be a perfect square")
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
    source_points = (
        0.2 + 0.25 * (ii.ravel() + 0.23) / side
        + 1j * (0.2 + 0.6 * (jj.ravel() + 0.37) / side)
    )
    target_points = (
        0.55 + 0.25 * (ii.ravel() + 0.61) / side
        + 1j * (0.2 + 0.6 * (jj.ravel() + 0.19) / side)
    )
    points = source_points
    values = np.exp(
        -((points.real - 0.5) ** 2 + (points.imag - 0.5) ** 2) / (2.0 * 0.12**2)
    ).astype(np.complex128)
    weights = np.full(points_count, 0.15 / points_count, dtype=np.float64)
    t0 = time.perf_counter()
    direct = direct_beurling_apply(
        points, values, weights, block_size=256, target_points=target_points
    )
    direct_seconds = time.perf_counter() - t0
    records: list[dict] = []
    for grid in (128, 256, 512):
        t1 = time.perf_counter()
        approximate = particle_mesh_beurling_apply(
            points,
            values,
            weights,
            grid_shape=(grid, grid),
            kernel="cubic",
            deconvolve=True,
            target_points=target_points,
        )
        elapsed = time.perf_counter() - t1
        difference = approximate - direct
        records.append(
            {
                "grid": grid,
                "elapsed_seconds": elapsed,
                "l2_error": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
                "relative_l2_error": float(
                    np.linalg.norm(difference) / max(np.linalg.norm(direct), 1e-15)
                ),
                "max_error": float(np.max(np.abs(difference))),
            }
        )
    result = {
        "points": points_count,
        "source_box": [0.2, 0.45],
        "target_box": [0.55, 0.8],
        "direct_seconds": direct_seconds,
        "records": records,
        "scope": "independent blocked O(N^2) whole-plane quadrature versus periodic cubic particle-mesh control",
        "limitation": "the direct reference omits the PV self term and the particle-mesh operator remains periodic; this is a cross-check, not a production NUFFT/FMM certificate",
    }
    (output_dir / "beurling_particle_mesh_direct_crosscheck.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, points_count=args.points), indent=2))


if __name__ == "__main__":
    main()
