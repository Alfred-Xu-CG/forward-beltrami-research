"""Near-aware Barnes--Hut correction audit against an external direct receipt."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_treecode import treecode_beurling_apply


def run(
    output_dir: Path,
    cloud_path: Path,
    reference_path: Path,
    theta: float = 0.22,
    order: int = 12,
    near_radius: float = 0.01,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud = np.load(cloud_path)
    reference = np.load(reference_path)
    records = []
    for label, radius in (("baseline", None), ("near_corrected", near_radius)):
        started = time.perf_counter()
        approximation = treecode_beurling_apply(
            cloud["points"],
            cloud["values"],
            cloud["weights"],
            target_points=cloud["targets"],
            theta=theta,
            order=order,
            near_radius=radius,
        )
        elapsed = time.perf_counter() - started
        error = approximation - reference
        np.save(output_dir / f"{label}_output.npy", approximation)
        records.append(
            {
                "label": label,
                "near_radius": radius,
                "seconds": float(elapsed),
                "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)),
                "max_abs_error": float(np.max(np.abs(error))),
                "finite": bool(np.all(np.isfinite(approximation))),
            }
        )
    result = {
        "points": int(cloud["points"].size),
        "theta": theta,
        "order": order,
        "near_radius": near_radius,
        "records": records,
        "scope": "nonperiodic scattered Beurling treecode with exact near-aware leaf traversal",
        "limitation": "near-aware traversal reduces multipole error but remains a Barnes-Hut control, not a production FMM or singular quadrature theorem",
    }
    (output_dir / "beurling_treecode_near_correction_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.22)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--near-radius", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.cloud, args.reference, args.theta, args.order, args.near_radius), indent=2))


if __name__ == "__main__":
    main()
