"""Tolerance/iteration audit for the matrix-free periodic Beurling solve."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling import periodic_lift_derivatives
from qcopt.forward.beurling_gmres import periodic_beltrami_gmres


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mu = 0.45 * np.exp(2j * np.pi * (3.0 * xx - 2.0 * yy))
    records: list[dict] = []
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10):
        t0 = time.perf_counter()
        result = periodic_beltrami_gmres(mu, rtol=rtol, maxiter=200)
        elapsed = time.perf_counter() - t0
        fz, fbar = periodic_lift_derivatives(result.h)
        records.append(
            {
                "rtol": rtol,
                "iterations": result.iterations,
                "operator_residual": result.operator_residual,
                "map_residual": float(np.max(np.abs(fbar - mu * fz))),
                "converged": result.converged,
                "elapsed_seconds": elapsed,
            }
        )
    result = {"n": n, "records": records}
    (output_dir / "beurling_gmres_tolerance_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
