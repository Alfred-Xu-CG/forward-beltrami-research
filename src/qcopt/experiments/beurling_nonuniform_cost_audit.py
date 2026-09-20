"""Cost audit showing the gap between uniform FFT and O(N^2) nonuniform quadrature."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply


def run(output_dir: Path, sizes: tuple[int, ...] = (1024, 2048, 4096)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    rng = np.random.default_rng(19)
    for n in sizes:
        points = rng.random(n) + 1j * rng.random(n)
        values = np.exp(-((points.real - 0.5) ** 2 + (points.imag - 0.5) ** 2) / (2.0 * 0.12**2))
        weights = np.full(n, 1.0 / n)
        t0 = time.perf_counter()
        output = direct_beurling_apply(points, values, weights, block_size=256)
        elapsed = time.perf_counter() - t0
        records.append(
            {
                "points": n,
                "elapsed_seconds": elapsed,
                "peak_output_norm": float(np.max(np.abs(output))),
            }
        )
    result = {
        "records": records,
        "method": "blocked O(N^2) midpoint quadrature, self term omitted",
        "warning": "diagnostic reference only; not a converged PV singular-integral solver",
    }
    (output_dir / "beurling_nonuniform_cost_audit.json").write_text(
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
