"""Resolution audit for the nonperiodic finite-difference rectangle BVP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.rectangle_fd import rectangle_beltrami_fd


def run(output_dir: Path, sizes: tuple[int, ...] = (128, 256, 512)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n in sizes:
        x = np.linspace(0.0, 1.0, n)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        coefficient = 0.18 + 0.11j
        truth = xx + 1j * yy + coefficient * (xx - 1j * yy)
        t0 = time.perf_counter()
        result = rectangle_beltrami_fd(
            np.full((n, n), coefficient, dtype=np.complex128), truth, iter_lim=1000
        )
        records.append(
            {
                "n": n,
                "unknowns": (n - 2) ** 2,
                "elapsed_seconds": time.perf_counter() - t0,
                "iterations": result.iterations,
                "equation_residual": result.max_equation_residual,
                "map_max_error": float(np.max(np.abs(result.map - truth))),
                "converged": result.converged,
            }
        )
    variable_records: list[dict] = []
    for n in sizes[:2]:
        x = np.linspace(0.0, 1.0, n)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        truth = (
            xx + 0.12 * np.sin(np.pi * xx) ** 2 * np.sin(np.pi * yy)
            + 1j * (yy + 0.08 * np.sin(np.pi * xx) * np.sin(np.pi * yy) ** 2)
        )
        spacing = 1.0 / (n - 1)
        fx = (truth[:, 2:] - truth[:, :-2]) / (2.0 * spacing)
        fy = (truth[2:, :] - truth[:-2, :]) / (2.0 * spacing)
        fz = 0.5 * (fx[1:-1, :] - 1j * fy[:, 1:-1])
        fbar = 0.5 * (fx[1:-1, :] + 1j * fy[:, 1:-1])
        coefficient = np.zeros((n, n), dtype=np.complex128)
        coefficient[1:-1, 1:-1] = fbar / fz
        t0 = time.perf_counter()
        result = rectangle_beltrami_fd(coefficient, truth, method="direct")
        variable_records.append(
            {
                "n": n,
                "elapsed_seconds": time.perf_counter() - t0,
                "max_abs_mu": float(np.max(np.abs(coefficient))),
                "equation_residual": result.max_equation_residual,
                "map_rms_error": float(np.sqrt(np.mean(np.abs(result.map - truth) ** 2))),
                "map_max_error": float(np.max(np.abs(result.map - truth))),
            }
        )
    result = {
        "records": records,
        "variable_manufactured_records": variable_records,
        "scope": "Dirichlet finite-difference rectangle BVP, constant affine truth",
    }
    (output_dir / "rectangle_fd_resolution_audit.json").write_text(
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
