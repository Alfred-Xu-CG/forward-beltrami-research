"""Extract DHR SuperPoint/SuperGlue matches in an affine-aligned Q1 frame.

External pretrained image features are an exploratory proposal source, not an
end-to-end learned image-to-Q1 layer. No anatomical landmarks are read here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.digital_q1_sift_residual import warp_moving_to_fixed


def select_ransac_matches(
    points0: np.ndarray, points1: np.ndarray, matches0: np.ndarray,
    *, width: int, height: int,
) -> dict:
    """Filter image0->image1 matches and convert pixel centers to unit coords."""
    points0 = np.asarray(points0)
    points1 = np.asarray(points1)
    matches0 = np.asarray(matches0)
    if points0.ndim != 2 or points0.shape[1] != 2 or (
        points1.ndim != 2 or points1.shape[1] != 2 or
        matches0.shape != (len(points0),) or width < 2 or height < 2
    ):
        raise ValueError("invalid keypoints, matches or image dimensions")
    valid = (matches0 >= 0) & (matches0 < len(points1))
    source = np.asarray(points0[valid], dtype=np.float32)
    target = np.asarray(points1[matches0[valid]], dtype=np.float32)
    report = {
        "raw_matches": int(valid.sum()),
        "ransac_inliers": 0,
        "occupied_quadrants_4x4": 0,
        "status": "insufficient_matches",
        "source_points_unit": [],
        "target_points_unit": [],
    }
    if len(source) < 8:
        return report
    cv2.setRNGSeed(290929)
    _, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=6.,
        maxIters=2000, confidence=.99, refineIters=10,
    )
    if mask is None:
        return report
    good = mask.ravel().astype(bool)
    report["ransac_inliers"] = int(good.sum())
    if report["ransac_inliers"] < 8:
        return report
    source = source[good].astype(np.float64)
    target = target[good].astype(np.float64)
    sizes = np.array([width, height], dtype=np.float64)
    source = (source + .5) / sizes
    target = (target + .5) / sizes
    bins = np.minimum(3, np.floor(source * 4).astype(int)).clip(0, 3)
    report["occupied_quadrants_4x4"] = int(len(set(map(tuple, bins))))
    report["median_match_distance_px"] = float(np.median(np.linalg.norm(
        (target - source) * sizes, axis=1)))
    report["source_bbox_unit_xyxy"] = [float(source[:, 0].min()),
                                       float(source[:, 1].min()),
                                       float(source[:, 0].max()),
                                       float(source[:, 1].max())]
    report["status"] = "ok"
    report["source_points_unit"] = source.tolist()
    report["target_points_unit"] = target.tolist()
    return report


def extract(
    fixed: torch.Tensor, moving: torch.Tensor,
    matrix: torch.Tensor, offset: torch.Tensor, *, device: str,
) -> dict:
    """Run the unmodified DHR model on fixed and affine-aligned moving views."""
    import deeperhistreg  # noqa: F401 - DHR installs legacy import aliases
    import superpoint_superglue as module

    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[:2] != (1, 1):
        raise ValueError("matching B=1,C=1 thumbnails required")
    target_device = torch.device(device)
    fixed = fixed.to(target_device)
    moving = moving.to(target_device)
    height, width = fixed.shape[-2:]
    aligned = warp_moving_to_fixed(moving, matrix.to(target_device),
                                   offset.to(target_device),
                                   height=height, width=width)
    config = {
        "superpoint": {"nms_radius": 4, "keypoint_threshold": .005,
                       "max_keypoints": 3000},
        "superglue": {"weights": "outdoor", "sinkhorn_iterations": 30,
                      "match_threshold": .3},
    }
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    started = time.perf_counter()
    model = module.sg.Matching(config).eval().to(target_device)
    model.superpoint.load_state_dict(torch.load(module.p.superpoint_model_path,
                                                map_location=target_device))
    model.superglue.load_state_dict(torch.load(module.p.superglue_model_path,
                                              map_location=target_device))
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    model_seconds = time.perf_counter() - started
    with torch.no_grad():
        started = time.perf_counter()
        prediction = model({"image0": fixed, "image1": aligned})
        prediction = {key: value[0].detach().cpu().numpy()
                      for key, value in prediction.items()}
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        inference_seconds = time.perf_counter() - started
    report = select_ransac_matches(
        prediction["keypoints0"], prediction["keypoints1"],
        prediction["matches0"], width=width, height=height,
    )
    report.update({
        "fixed_keypoints": int(len(prediction["keypoints0"])),
        "aligned_moving_keypoints": int(len(prediction["keypoints1"])),
        "model_build_and_weight_load_seconds": model_seconds,
        "inference_and_to_cpu_seconds": inference_seconds,
        "cuda_peak_allocated_bytes": (None if target_device.type != "cuda" else
                                      int(torch.cuda.max_memory_allocated(target_device))),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--affine-map", type=Path, required=True)
    parser.add_argument("--output-matches", type=Path, required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    fixed, _ = _read_gray_thumbnail(args.fixed_image, args.image_side)
    moving, _ = _read_gray_thumbnail(args.moving_image, args.image_side)
    with np.load(args.affine_map) as archive:
        matrix = torch.from_numpy(archive["post_affine_matrix"].copy())
        offset = torch.from_numpy(archive["post_affine_offset"].copy())
    report = extract(fixed, moving, matrix, offset, device=args.device)
    report.update({
        "method": "DHR 1.0.1 SuperPoint/SuperGlue post-affine, RANSAC 6 px",
        "image_side": args.image_side,
        "fixed_image": str(args.fixed_image),
        "moving_image": str(args.moving_image),
        "affine_map": str(args.affine_map),
        "total_decode_align_model_match_seconds": time.perf_counter() - started,
    })
    args.output_matches.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"source_points_unit", "target_points_unit"}}, indent=2))


if __name__ == "__main__":
    main()
