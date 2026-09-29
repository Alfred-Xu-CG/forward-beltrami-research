"""Image-only SIFT coverage diagnostic/fallback for failed direct-SG cases.

Attempts raw contrast first, then CLAHE; choose the accepted fit with more
RANSAC inliers. No annotation or DHR transform is read. A failed pair gets
no affine file, never a silent identity output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_q1_sift_similarity import (
    estimate_sift_similarity, pixel_affine_to_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--failed-from", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    ids = set(json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"])
    successes = set()
    for report_path in args.failed_from:
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        successes.update(int(row["name"]) for row in prior["rows"]
                         if row["status"] == "ok")
    pending = sorted(ids - successes)
    rows = []
    for case in pending:
        started = time.perf_counter()
        fixed_path, moving_path = _case_images(args.root, case)
        fixed = cv2.imread(str(fixed_path), cv2.IMREAD_GRAYSCALE)
        moving = cv2.imread(str(moving_path), cv2.IMREAD_GRAYSCALE)
        if fixed is None or moving is None or fixed.shape != (512, 512) or moving.shape != (512, 512):
            raise ValueError(f"missing 512-square image: {case}")
        candidates = []
        for clahe in (False, True):
            matrix, stats = estimate_sift_similarity(fixed, moving, clahe=clahe)
            candidates.append((matrix, stats))
        accepted = [v for v in candidates if not v[1]["identity_fallback"]]
        row = {"case": case, "attempts": [v[1] for v in candidates],
               "status": "insufficient_matches"}
        if accepted:
            matrix, stats = max(accepted, key=lambda v: v[1]["ransac_inliers"])
            linear, offset = pixel_affine_to_unit(matrix, 512)
            determinant = float(linear[0, 0] * linear[1, 1] - linear[0, 1] * linear[1, 0])
            if determinant > 0:
                output = args.root / f"{case}_directSG_affine.npz"
                if output.exists():
                    raise FileExistsError(output)
                np.savez_compressed(output,
                                    post_affine_matrix=linear.astype(np.float32),
                                    post_affine_offset=offset.astype(np.float32),
                                    affine_source="image_only_sift_fallback")
                row.update({"status": "ok_sift", "chosen_clahe": stats["clahe"],
                            "ransac_inliers": stats["ransac_inliers"],
                            "determinant": determinant,
                            "output": str(output)})
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
    report = {"method": "image-only SIFT fallback for directSG failures",
              "root": str(args.root), "pending_count": len(pending), "rows": rows,
              "successful_count": sum(r["status"] == "ok_sift" for r in rows)}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pending": len(pending), "successes": report["successful_count"]}))


if __name__ == "__main__":
    main()
