"""Compare the scattered Beurling treecode with a float64 CPU direct reference.

The existing 16,384-point paired-offset receipt used an external GPU direct
array.  This audit reuses the same cloud but computes an independent blocked
complex128 reference on CPU, which separates treecode truncation error from
float32/device-reference error before any singular-quadrature redesign.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_treecode import treecode_beurling_apply


def run(output_dir: Path, cloud_path: Path, block_size: int = 512) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud = np.load(cloud_path)
    started = time.perf_counter()
    reference = direct_beurling_apply(
        cloud["points"],
        cloud["values"],
        cloud["weights"],
        target_points=cloud["targets"],
        block_size=block_size,
    )
    reference_seconds = time.perf_counter() - started
    np.save(output_dir / "cpu_direct_reference.npy", reference)
    records = []
    configs = (
        (0.22, 12, None),
        (0.12, 16, None),
        (0.08, 24, None),
        (0.22, 12, 0.01),
    )
    for theta, order, near_radius in configs:
        started = time.perf_counter()
        approximation = treecode_beurling_apply(
            cloud["points"],
            cloud["values"],
            cloud["weights"],
            target_points=cloud["targets"],
            theta=theta,
            order=order,
            near_radius=near_radius,
        )
        elapsed = time.perf_counter() - started
        error = approximation - reference
        records.append(
            {
                "theta": theta,
                "order": order,
                "near_radius": near_radius,
                "seconds": float(elapsed),
                "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)),
                "max_abs_error": float(np.max(np.abs(error))),
                "finite": bool(np.all(np.isfinite(approximation))),
            }
        )
    result = {
        "points": int(cloud["points"].size),
        "block_size": int(block_size),
        "reference_seconds": float(reference_seconds),
        "reference_dtype": "complex128 blocked direct quadrature",
        "records": records,
        "scope": "paired near-source whole-plane Beurling treecode versus independent CPU reference",
        "interpretation": "if the CPU error is much smaller than the external GPU receipt, float32/device reference error dominates; otherwise the treecode or PV quadrature remains the bottleneck",
        "limitation": "direct CPU reference is O(N^2) and omits only the self term when source and target sets coincide; paired targets are disjoint",
    }
    (output_dir / "beurling_treecode_cpu_reference_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.cloud, args.block_size), indent=2))


if __name__ == "__main__":
    main()
