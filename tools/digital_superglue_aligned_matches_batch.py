"""Extract image-only residual SuperGlue correspondences with one reused model.

The moving image is warped by each already saved image-only positive affine.
Landmarks, full DHR field, and DHR affine are not read. Successful RANSAC
matches are saved as fixed/aligned unit-square point pairs for later training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_q1_superglue_residual import select_ransac_matches


def run(root: Path, selection: Path, output_dir: Path, report_out: Path,
        device_name: str, id_key: str = "combined_train_ids") -> dict:
    if report_out.exists():
        raise FileExistsError(report_out)
    if id_key not in ("combined_train_ids", "new_confirmation_ids"):
        raise ValueError("unknown case group")
    ids = json.loads(selection.read_text(encoding="utf-8"))[id_key]
    available = [int(case) for case in ids
                 if (root / f"{case}_directSG_affine.npz").exists()]
    import deeperhistreg  # noqa: F401 - DHR package aliases only
    import superpoint_superglue as module
    device = torch.device(device_name)
    config = {
        "superpoint": {"nms_radius": 4, "keypoint_threshold": .005,
                       "max_keypoints": 3000},
        "superglue": {"weights": "outdoor", "sinkhorn_iterations": 30,
                      "match_threshold": .3},
    }
    started = time.perf_counter()
    matcher = module.sg.Matching(config).eval().to(device)
    matcher.superpoint.load_state_dict(torch.load(
        module.p.superpoint_model_path, map_location=device, weights_only=False))
    matcher.superglue.load_state_dict(torch.load(
        module.p.superglue_model_path, map_location=device, weights_only=False))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in available:
        started = time.perf_counter()
        item = load_case_inputs(root, case, device, affine_source="image_only")
        with torch.no_grad():
            pred = matcher({"image0": item["fixed"],
                            "image1": item["prewarped"]})
            pred = {key: value[0].detach().cpu().numpy()
                    for key, value in pred.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        match = select_ransac_matches(pred["keypoints0"], pred["keypoints1"],
                                      pred["matches0"], width=512, height=512)
        row = {"case": case, "status": match["status"],
               "raw_matches": match["raw_matches"],
               "ransac_inliers": match["ransac_inliers"],
               "occupied_quadrants_4x4": match["occupied_quadrants_4x4"]}
        if match["status"] == "ok":
            output = output_dir / f"{case}_alignedSG_matches.npz"
            if output.exists():
                raise FileExistsError(output)
            with np.load(root / f"{case}_directSG_affine.npz") as affine:
                matrix = affine["post_affine_matrix"].copy()
                offset = affine["post_affine_offset"].copy()
            np.savez_compressed(
                output,
                source_fixed_unit=np.asarray(match["source_points_unit"],
                                             dtype=np.float32),
                target_aligned_unit=np.asarray(match["target_points_unit"],
                                               dtype=np.float32),
                post_affine_matrix=matrix, post_affine_offset=offset,
            )
            row["output"] = str(output)
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
    result = {"method": "one reused SG model on fixed and image-only-affine-aligned moving",
              "model_seconds": model_seconds, "case_count": len(rows),
              "success_count": sum(r["status"] == "ok" for r in rows),
              "device": device_name, "rows": rows,
              "landmarks_DHR_initial_full_field_used": False}
    report_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "successes": result["success_count"],
                      "model_seconds": model_seconds,
                      "mean_inliers": sum(r["ransac_inliers"] for r in rows) / len(rows)}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--id-key", choices=("combined_train_ids",
                                              "new_confirmation_ids"),
                        default="combined_train_ids")
    args = parser.parse_args()
    run(args.root, args.selection, args.output_dir, args.report, args.device,
        id_key=args.id_key)


if __name__ == "__main__":
    main()
