"""Image-only aligned matches and safe P1 predictions for all 20 stain directions.

Reuses previously frozen direct-image affine archives and frozen P1 starting
maps. No anatomical landmark file is opened. The separate all-20 scorer is
called only after this prediction program has finished writing its maps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_frozen_residual_predict import predict
from tools.digital_lung_all20_predict import STAINS, image_path
from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.digital_q1_superglue_residual import select_ransac_matches


def run(canvas: Path, previous: Path, output: Path, base: Path,
        one_head: Path, residual: Path, *, device_name: str,
        matches_from: Path | None = None,
        match_feedback_gain: float = 0.) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("new all-20 output directory required")
    old = json.loads((previous / "predictions.json").read_text(encoding="utf-8"))
    names = [f"{a}_to_{b}" for a, b in itertools.permutations(STAINS, 2)]
    if len(old["rows"]) != 20 or [r["name"] for r in old["rows"]] != names:
        raise ValueError("previous run must contain exactly ordered 20 pairs")
    device = torch.device(device_name)
    matcher = None
    if matches_from is not None:
        reused = json.loads((matches_from / "predictions.json").read_text())
        if [row["name"] for row in reused["rows"]] != names or any(
            row["status"] != "ok" for row in reused["rows"]
        ):
            raise ValueError("reused match cohort differs or has failures")
    else:
        matcher_load_started = time.perf_counter()
        import deeperhistreg  # noqa: F401 - package compatibility aliases
        import superpoint_superglue as module

        config = {
            "superpoint": {"nms_radius": 4, "keypoint_threshold": .005,
                           "max_keypoints": 3000},
            "superglue": {"weights": "outdoor", "sinkhorn_iterations": 30,
                          "match_threshold": .3},
        }
        matcher = module.sg.Matching(config).eval().to(device)
        matcher.superpoint.load_state_dict(torch.load(
            module.p.superpoint_model_path, map_location=device, weights_only=False))
        matcher.superglue.load_state_dict(torch.load(
            module.p.superglue_model_path, map_location=device, weights_only=False))
        matcher_load_seconds = time.perf_counter() - matcher_load_started
    if matches_from is not None:
        matcher_load_seconds = None
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous / "affines.json", output / "affines.json")
    rows = []
    for fixed_name, moving_name in itertools.permutations(STAINS, 2):
        pair_started = time.perf_counter()
        name = f"{fixed_name}_to_{moving_name}"
        fixed_path = image_path(canvas, fixed_name)
        moving_path = image_path(canvas, moving_name)
        affine = output / f"{name}_affine.npz"
        frozen = output / f"{name}_frozen_safe257.npz"
        shutil.copy2(previous / affine.name, affine)
        shutil.copy2(previous / frozen.name, frozen)
        with np.load(affine) as archive:
            matrix = archive["post_affine_matrix"].copy()
            offset = archive["post_affine_offset"].copy()
        matches = output / f"{name}_aligned_matches.npz"
        if matches_from is not None:
            match_started = time.perf_counter()
            shutil.copy2(matches_from / matches.name, matches)
            with np.load(matches) as archive:
                if not (np.array_equal(archive["post_affine_matrix"], matrix) and
                        np.array_equal(archive["post_affine_offset"], offset)):
                    raise ValueError(f"reused match affine differs: {name}")
            old_row = next(row for row in reused["rows"] if row["name"] == name)
            row = {key: old_row[key] for key in (
                "name", "status", "raw_matches", "ransac_inliers",
                "occupied_quadrants_4x4")}
            row["residual_match_wall_seconds"] = time.perf_counter() - match_started
            row["matches_reused"] = True
        else:
            match_started = time.perf_counter()
            fixed_image, _ = _read_gray_thumbnail(fixed_path, 512)
            moving_image, _ = _read_gray_thumbnail(moving_path, 512)
            with torch.no_grad():
                aligned = warp_moving_to_fixed(
                    moving_image.to(device), torch.from_numpy(matrix).to(device),
                    torch.from_numpy(offset).to(device), height=512, width=512)
                prediction = matcher({"image0": fixed_image.to(device),
                                      "image1": aligned})
                prediction = {key: value[0].detach().cpu().numpy()
                              for key, value in prediction.items()}
            selected = select_ransac_matches(
                prediction["keypoints0"], prediction["keypoints1"],
                prediction["matches0"], width=512, height=512)
            row = {"name": name, "status": selected["status"],
                   "raw_matches": selected["raw_matches"],
                   "ransac_inliers": selected["ransac_inliers"],
                   "occupied_quadrants_4x4": selected["occupied_quadrants_4x4"]}
            if selected["status"] == "ok" and selected["ransac_inliers"] >= 16:
                np.savez_compressed(
                    matches,
                    source_fixed_unit=np.asarray(
                        selected["source_points_unit"], dtype=np.float32),
                    target_aligned_unit=np.asarray(
                        selected["target_points_unit"], dtype=np.float32),
                    post_affine_matrix=matrix, post_affine_offset=offset,
                )
            elif selected["status"] == "ok":
                row["status"] = "too_few_inliers"
            row["residual_match_wall_seconds"] = time.perf_counter() - match_started
            row["matches_reused"] = False
        if row["status"] == "ok":
            predict_started = time.perf_counter()
            result = predict(
                base, one_head, residual, fixed_path, moving_path, affine,
                matches, output / f"{name}_dynamic_safe257.npz",
                device_name=device_name, repeats=1,
                match_feedback_gain=match_feedback_gain)
            row["predict_call_wall_seconds"] = time.perf_counter() - predict_started
            row.update({
                "status": "ok", "dynamic_image": result["residual_image"],
                "dynamic_q1": result["saved_binary_certificate"]["valid"],
                "dynamic_finite_vjp": result["finite_full_vjp"],
                "match_affine_frame_machine_checked": result[
                    "match_affine_frame_machine_checked"],
                "neural_forward_seconds_median": result["full_forward_seconds_median"],
                "neural_forward_and_vjp_seconds_median": result[
                    "full_forward_and_parameter_match_vjp_seconds_median"],
            })
        row["pair_wall_seconds"] = time.perf_counter() - pair_started
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "protocol": "all 20 ordered directions in one previously viewed specimen",
        "model": str(residual), "affines_reused_from": str(previous),
        "matches_reused_from": None if matches_from is None else str(matches_from),
        "matcher_checkpoint_load_wall_seconds": matcher_load_seconds,
        "direct_affine_matcher_reused_not_timed": True,
        "match_feedback_gain": match_feedback_gain,
        "annotations_read": False, "match_raster_supplied_to_student": True,
        "failed_directions_retained": True, "rows": rows,
    }
    (output / "predictions.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "previous", "output", "base", "one_head", "residual"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matches-from", type=Path)
    parser.add_argument("--match-feedback-gain", type=float, default=0.)
    args = parser.parse_args()
    result = run(args.canvas, args.previous, args.output, args.base,
                 args.one_head, args.residual, device_name=args.device,
                 matches_from=args.matches_from,
                 match_feedback_gain=args.match_feedback_gain)
    print(json.dumps({"pair_count": len(result["rows"]),
                      "success_count": sum(row["status"] == "ok"
                                           for row in result["rows"])}))


if __name__ == "__main__":
    main()
