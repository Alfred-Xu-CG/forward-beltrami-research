"""Extract image-only post-affine SuperGlue matches from ACROBAT canvases.

One eval-mode feature model is reused across cases. A saved DHR-derived
*initial affine* is loaded for each prewarp; full DHR displacement teachers
and anatomical/evaluation landmarks are never loaded here.
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


def extract(root: Path, case_ids: list[int], output: Path, *, device: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("distinct nonempty case IDs required")
    import deeperhistreg  # noqa: F401 - package installs legacy aliases
    import superpoint_superglue as module

    target_device = torch.device(device)
    config = {
        "superpoint": {"nms_radius": 4, "keypoint_threshold": .005,
                       "max_keypoints": 3000},
        "superglue": {"weights": "outdoor", "sinkhorn_iterations": 30,
                      "match_threshold": .3},
    }
    began = time.perf_counter()
    matcher = module.sg.Matching(config).eval().to(target_device)
    matcher.superpoint.load_state_dict(torch.load(
        module.p.superpoint_model_path, map_location=target_device,
        weights_only=False,
    ))
    matcher.superglue.load_state_dict(torch.load(
        module.p.superglue_model_path, map_location=target_device,
        weights_only=False,
    ))
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    load_seconds = time.perf_counter() - began
    rows = []
    for case in case_ids:
        began = time.perf_counter()
        images = load_case_inputs(root, case, target_device)
        fixed, prewarped = images["fixed"], images["prewarped"]
        height, width = fixed.shape[-2:]
        with torch.no_grad():
            prediction = matcher({"image0": fixed, "image1": prewarped})
            prediction = {key: value[0].detach().cpu().numpy()
                          for key, value in prediction.items()}
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        matches = select_ransac_matches(
            prediction["keypoints0"], prediction["keypoints1"],
            prediction["matches0"], width=width, height=height,
        )
        residual = np.asarray(matches["target_points_unit"], dtype=np.float64) - np.asarray(
            matches["source_points_unit"], dtype=np.float64)
        if matches["status"] == "ok":
            matches["residual_vector_rms_unit"] = float(np.sqrt(np.mean(np.sum(
                residual ** 2, axis=-1))))
        else:
            matches["residual_vector_rms_unit"] = None
        matches.update({
            "case": case,
            "fixed_keypoints": int(len(prediction["keypoints0"])),
            "aligned_moving_keypoints": int(len(prediction["keypoints1"])),
            "decode_prewarp_match_filter_seconds": time.perf_counter() - began,
        })
        rows.append(matches)
    report = {
        "question": "Are image-only post-affine matches numerous and widespread on the training cohort?",
        "method": "one reused DHR 1.0.1 SuperPoint/SuperGlue outdoor model, then partial-affine RANSAC 6 px",
        "DHR_derived_initial_affine_loaded": True,
        "full_DHR_displacement_teacher_or_anatomical_landmarks_loaded": False,
        "case_ids": case_ids,
        "device": device,
        "model_build_and_load_seconds": load_seconds,
        "total_case_seconds": float(sum(row["decode_prewarp_match_filter_seconds"]
                                        for row in rows)),
        "peak_torch_cuda_allocated_bytes": (None if target_device.type != "cuda" else
                                              int(torch.cuda.max_memory_allocated(target_device))),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("max-cases must be positive")
        ids = ids[:args.max_cases]
    result = extract(args.root, ids, args.output, device=args.device)
    print(json.dumps({
        "case_count": len(result["rows"]),
        "ok_count": sum(row["status"] == "ok" for row in result["rows"]),
        "total_case_seconds": result["total_case_seconds"],
        "peak_torch_cuda_allocated_bytes": result["peak_torch_cuda_allocated_bytes"],
    }))


if __name__ == "__main__":
    main()
