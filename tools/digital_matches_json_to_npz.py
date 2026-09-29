"""Convert already extracted image-only aligned match JSON to compact arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--affine-map", type=Path,
                        help="explicit affine that produced the aligned moving image")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report["status"] != "ok" or report["ransac_inliers"] < 16:
        raise ValueError("at least 16 accepted image matches required")
    source = np.asarray(report["source_points_unit"], dtype=np.float32)
    target = np.asarray(report["target_points_unit"], dtype=np.float32)
    if source.shape != target.shape or source.shape != (report["ransac_inliers"], 2):
        raise ValueError("match array inconsistency")
    affine_path = args.affine_map or (
        Path(report["affine_map"]) if "affine_map" in report else None)
    payload = {"source_fixed_unit": source, "target_aligned_unit": target}
    if affine_path is not None:
        if not affine_path.is_file():
            raise FileNotFoundError(f"affine metadata source not found: {affine_path}")
        with np.load(affine_path) as affine:
            payload["post_affine_matrix"] = affine["post_affine_matrix"].copy()
            payload["post_affine_offset"] = affine["post_affine_offset"].copy()
    np.savez_compressed(args.output, **payload)
    print(json.dumps({"count": len(source), "output": str(args.output)}))


if __name__ == "__main__":
    main()
