"""Scaling receipt for the fixed-tree Beurling forward and VJP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_treecode import treecode_beurling_apply, treecode_beurling_values_vjp


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    records = []
    for count in (1024, 2048, 4096):
        source = 0.05 + 0.42 * rng.random(count) + 1j * (0.05 + 0.9 * rng.random(count))
        target = 0.55 + 0.4 * rng.random(count) + 1j * (0.05 + 0.9 * rng.random(count))
        values = rng.normal(size=count) + 1j * rng.normal(size=count)
        weights = (0.4 + 0.8 * rng.random(count)) / count
        cotangent = rng.normal(size=count) + 1j * rng.normal(size=count)
        kwargs = dict(target_points=target, theta=0.22, order=12)
        t0 = time.perf_counter()
        forward = treecode_beurling_apply(source, values, weights, **kwargs)
        forward_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        gradient = treecode_beurling_values_vjp(source, values, weights, cotangent, **kwargs)
        vjp_seconds = time.perf_counter() - t1
        records.append({
            "points": count,
            "forward_seconds": forward_seconds,
            "vjp_seconds": vjp_seconds,
            "forward_l2": float(np.linalg.norm(forward)),
            "vjp_l2": float(np.linalg.norm(gradient)),
            "finite": bool(np.all(np.isfinite(forward)) and np.all(np.isfinite(gradient))),
        })
    result = {
        "records": records,
        "theta": 0.22,
        "order": 12,
        "scope": "nonperiodic scattered fixed-tree Beurling forward and exact transpose scaling",
        "limitation": "Barnes--Hut control, not a production FMM or singular quadrature theorem",
    }
    (output_dir / "beurling_treecode_scaling_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
