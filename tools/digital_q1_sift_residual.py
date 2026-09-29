"""Image-only SIFT matches after a fixed affine, in the safe Q1 residual frame.

This first stage extracts correspondences only. It neither reads anatomical
landmarks nor assumes its machine-generated matches are ground truth.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def extract_affine_aligned_sift_matches(
    fixed: torch.Tensor, moving: torch.Tensor,
    matrix: torch.Tensor, offset: torch.Tensor,
) -> dict:
    """Return conservative matched q->p points, both in fixed unit coordinates."""
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[:2] != (1, 1):
        raise ValueError("matching B=1,C=1 thumbnails required")
    height, width = fixed.shape[-2:]
    cv2.setRNGSeed(290929)
    cv2.setNumThreads(1)
    aligned = warp_moving_to_fixed(moving, matrix, offset,
                                   height=height, width=width)
    fixed_u8 = np.asarray(np.rint(fixed[0, 0].detach().cpu().numpy() * 255),
                          dtype=np.uint8)
    aligned_u8 = np.asarray(np.rint(aligned[0, 0].detach().cpu().numpy() * 255),
                            dtype=np.uint8)
    detector = cv2.SIFT_create(nfeatures=4000)
    fixed_points, fixed_descriptors = detector.detectAndCompute(fixed_u8, None)
    aligned_points, aligned_descriptors = detector.detectAndCompute(aligned_u8, None)
    report = {
        "fixed_keypoints": len(fixed_points),
        "aligned_moving_keypoints": len(aligned_points),
        "mutual_ratio_matches": 0,
        "ransac_inliers": 0,
        "occupied_quadrants_4x4": 0,
        "status": "insufficient_matches",
        "source_points_unit": [],
        "target_points_unit": [],
    }
    if fixed_descriptors is None or aligned_descriptors is None or (
        len(fixed_points) < 8 or len(aligned_points) < 8
    ):
        return report
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(fixed_descriptors, aligned_descriptors, k=2)
    reverse = matcher.knnMatch(aligned_descriptors, fixed_descriptors, k=2)
    reverse_good = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse
                    if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance}
    selected = [pair[0] for pair in forward if len(pair) == 2 and (
        pair[0].distance < .75 * pair[1].distance and
        reverse_good.get(pair[0].trainIdx) == pair[0].queryIdx
    )]
    report["mutual_ratio_matches"] = len(selected)
    if len(selected) < 8:
        return report
    source = np.float32([fixed_points[match.queryIdx].pt for match in selected])
    target = np.float32([aligned_points[match.trainIdx].pt for match in selected])
    _, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=6.,
        maxIters=2000, confidence=.99, refineIters=10,
    )
    if mask is None:
        return report
    inlier = mask.ravel().astype(bool)
    report["ransac_inliers"] = int(inlier.sum())
    if report["ransac_inliers"] < 8:
        return report
    source, target = source[inlier].astype(np.float64), target[inlier].astype(np.float64)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-image", required=True, type=Path)
    parser.add_argument("--moving-image", required=True, type=Path)
    parser.add_argument("--affine-map", required=True, type=Path)
    parser.add_argument("--output-matches", required=True, type=Path)
    parser.add_argument("--image-side", type=int, default=512)
    args = parser.parse_args()
    started = time.perf_counter()
    fixed, _ = _read_gray_thumbnail(args.fixed_image, args.image_side)
    moving, _ = _read_gray_thumbnail(args.moving_image, args.image_side)
    with np.load(args.affine_map) as archive:
        matrix = torch.from_numpy(archive["post_affine_matrix"].copy())
        offset = torch.from_numpy(archive["post_affine_offset"].copy())
    report = extract_affine_aligned_sift_matches(fixed, moving, matrix, offset)
    report.update({
        "method": "SIFT mutual .75 ratio then partial-affine RANSAC 6 px",
        "image_side": args.image_side,
        "fixed_image": str(args.fixed_image),
        "moving_image": str(args.moving_image),
        "affine_map": str(args.affine_map),
        "decode_align_match_seconds": time.perf_counter() - started,
    })
    args.output_matches.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"source_points_unit", "target_points_unit"}}, indent=2))


if __name__ == "__main__":
    main()
