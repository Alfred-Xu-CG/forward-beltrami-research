"""Resolution/cost audit for the normalized first-variation quadrature."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import normalized_bhf_variation


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n, m in ((4096, 2048), (16384, 4096)):
        rng = np.random.default_rng(5 + n)
        source = (rng.random(n) + 1j * rng.random(n)) * 2.0 - 1.0 + 0.37 + 0.21j
        source[np.abs(source) < 0.05] += 0.2
        evaluation = (rng.random(m) + 1j * rng.random(m)) * 2.0 - 1.0
        variation = 0.2 * np.exp(-np.abs(source - 0.2 - 0.3j) ** 2 / 0.4)
        weights = np.full(n, 4.0 / n)
        t0 = time.perf_counter()
        velocity = normalized_bhf_variation(
            evaluation, source, variation, weights, block_size=256
        )
        records.append(
            {
                "source_points": n,
                "evaluation_points": m,
                "elapsed_seconds": time.perf_counter() - t0,
                "max_velocity": float(np.max(np.abs(velocity))),
            }
        )
    result = {
        "records": records,
        "scope": "normalized first variation at conformal base; direct blocked quadrature",
        "limitation": "not the nonlinear arbitrary-mu BHF flow and not a singular-PV production discretization",
    }
    (output_dir / "bhf_variation_cost_audit.json").write_text(
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
