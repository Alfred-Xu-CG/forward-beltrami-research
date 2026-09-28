"""Image-only SIFT/RANSAC positive-similarity initializer for Q1 comparison.

This is a no-training diagnostic, not a replacement for an F1/F2 residual
predictor. The saved map is the certified Q1 identity residual followed by a
positive-orientation global similarity estimated from fixed/moving images.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def pixel_affine_to_unit(pixel_matrix: np.ndarray, image_side: int) -> tuple[np.ndarray, np.ndarray]:
    """For pixel centers p=side*u-(.5,.5), convert q=A*p+t to v=(q+.5)/side."""
    pixel_matrix = np.asarray(pixel_matrix, dtype=np.float64)
    if pixel_matrix.shape != (2, 3) or image_side < 2 or not np.isfinite(pixel_matrix).all():
        raise ValueError("finite 2x3 pixel affine and image_side>=2 required")
    linear = pixel_matrix[:, :2]
    offset = (pixel_matrix[:, 2] + .5 - linear @ np.full(2, .5)) / image_side
    return linear, offset


def estimate_sift_similarity(
    fixed_gray: np.ndarray, moving_gray: np.ndarray,
    *, clahe: bool = False,
) -> tuple[np.ndarray, dict]:
    """Return fixed-pixel-center to moving-pixel-center positive similarity."""
    if fixed_gray.dtype != np.uint8 or moving_gray.dtype != np.uint8 or (
        fixed_gray.ndim != 2 or moving_gray.ndim != 2
    ):
        raise ValueError("two uint8 grayscale images required")
    cv2.setRNGSeed(290929)
    cv2.setNumThreads(1)
    if clahe:
        contrast = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        fixed_gray = contrast.apply(fixed_gray)
        moving_gray = contrast.apply(moving_gray)
    extractor = cv2.SIFT_create(nfeatures=4000)
    fixed_points, fixed_descriptors = extractor.detectAndCompute(fixed_gray, None)
    moving_points, moving_descriptors = extractor.detectAndCompute(moving_gray, None)
    report = {
        "fixed_keypoints": len(fixed_points), "moving_keypoints": len(moving_points),
        "ratio_matches": 0, "ransac_inliers": 0, "identity_fallback": True,
        "clahe": clahe,
    }
    identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
    if fixed_descriptors is None or moving_descriptors is None or len(moving_points) < 2:
        return identity, report
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fixed_descriptors, moving_descriptors, k=2)
    selected = [first for first, second in pairs if first.distance < .75 * second.distance]
    report["ratio_matches"] = len(selected)
    if len(selected) < 8:
        return identity, report
    src = np.float32([fixed_points[match.queryIdx].pt for match in selected])
    dst = np.float32([moving_points[match.trainIdx].pt for match in selected])
    candidate, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.,
        maxIters=2000, confidence=.99, refineIters=10,
    )
    report["ransac_inliers"] = 0 if mask is None else int(mask.sum())
    if candidate is None or report["ransac_inliers"] < 8 or not np.isfinite(candidate).all():
        return identity, report
    if float(np.linalg.det(candidate[:, :2])) <= 0:
        return identity, report
    report["identity_fallback"] = False
    return candidate.astype(np.float64), report


def save_similarity_q1(
    output_path: Path, pixel_matrix: np.ndarray, *, image_side: int = 512,
    control_side: int = 257,
) -> dict:
    if control_side < 2:
        raise ValueError("control_side>=2 required")
    matrix, offset = pixel_affine_to_unit(pixel_matrix, image_side)
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all() or np.linalg.det(matrix) <= 0:
        raise ValueError("positive finite affine required")
    axis = np.linspace(0, 1, control_side, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    residual = np.stack((xx, yy), axis=-1)[None]
    np.savez_compressed(
        output_path, vertices=residual, boundary_reference=residual,
        post_affine_matrix=matrix.astype(np.float32),
        post_affine_offset=offset.astype(np.float32),
    )
    certificate = certify_q1_binary_map(output_path)
    if not certificate["valid"]:
        raise ArithmeticError("stored Q1 residual failed sign or boundary audit")
    mapped = residual[0] @ matrix.T + offset
    outside = np.any((mapped < 0) | (mapped > 1), axis=-1)
    return {
        "saved_residual_valid": True,
        "saved_affine_det_float32": float(np.linalg.det(matrix.astype(np.float32).astype(np.float64))),
        "control_vertices_outside_moving_unit_square": int(outside.sum()),
        "control_side": control_side,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--control-side", type=int, default=257)
    parser.add_argument("--clahe", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    fixed, fixed_size = _read_gray_thumbnail(args.fixed_image, args.image_side)
    moving, moving_size = _read_gray_thumbnail(args.moving_image, args.image_side)
    fixed_u8 = np.asarray(np.rint(fixed[0, 0].numpy() * 255), dtype=np.uint8)
    moving_u8 = np.asarray(np.rint(moving[0, 0].numpy() * 255), dtype=np.uint8)
    pixel_matrix, report = estimate_sift_similarity(fixed_u8, moving_u8,
                                                    clahe=args.clahe)
    report.update(save_similarity_q1(
        args.output_map, pixel_matrix,
        image_side=args.image_side, control_side=args.control_side,
    ))
    report.update({
        "image_decode_match_and_save_seconds": time.perf_counter() - started,
        "fixed_original_size_xy": list(fixed_size),
        "moving_original_size_xy": list(moving_size),
        "pixel_similarity_fixed_to_moving": pixel_matrix.tolist(),
        "output_map": str(args.output_map),
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
