"""Direct image-only positive similarity initializer from one reused matcher.

No DHR registration stage, precomputed affine, anatomical landmarks, or full
deformation field is read. Pair outputs must be new files.
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from tools.digital_q1_real_optimize import _read_gray_thumbnail


def similarity_from_inlier_units(source: np.ndarray,
                                 target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares orientation-preserving complex similarity q=a*p+b."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or (
        len(source) < 2 or not np.isfinite(source).all() or not np.isfinite(target).all()
    ):
        raise ValueError("matching finite point arrays required")
    p = source - source.mean(axis=0)
    q = target - target.mean(axis=0)
    denom = float((p * p).sum())
    if denom <= 1e-10:
        raise ValueError("degenerate source point set")
    a = float((p * q).sum() / denom)
    b = float((p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum() / denom)
    matrix = np.array([[a, -b], [b, a]], dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    offset = target_center - np.array([
        a * source_center[0] - b * source_center[1],
        b * source_center[0] + a * source_center[1],
    ])
    if a * a + b * b <= 1e-8:
        raise ValueError("degenerate similarity scale")
    return matrix, offset


def extract(pairs: list[tuple[str, Path, Path, Path]], report_out: Path,
            *, device_name: str, match_threshold: float = .3) -> dict:
    if not pairs or report_out.exists() or any(out.exists() for _, _, _, out in pairs):
        raise ValueError("nonempty pairs with new outputs required")
    if not (0 < match_threshold < 1):
        raise ValueError("match threshold must be in (0,1)")
    import deeperhistreg  # noqa: F401 - installs DHR legacy module aliases
    import superpoint_superglue as module
    from tools.digital_q1_superglue_residual import select_ransac_matches

    device = torch.device(device_name)
    config = {
        "superpoint": {"nms_radius": 4, "keypoint_threshold": .005,
                       "max_keypoints": 3000},
        "superglue": {"weights": "outdoor", "sinkhorn_iterations": 30,
                      "match_threshold": match_threshold},
    }
    started = time.perf_counter()
    matcher = module.sg.Matching(config).eval().to(device)
    matcher.superpoint.load_state_dict(torch.load(
        module.p.superpoint_model_path, map_location=device, weights_only=False))
    matcher.superglue.load_state_dict(torch.load(
        module.p.superglue_model_path, map_location=device, weights_only=False))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model_seconds = time.perf_counter() - started
    rows = []
    for name, fixed_path, moving_path, output in pairs:
        started = time.perf_counter()
        fixed, _ = _read_gray_thumbnail(fixed_path, 512)
        moving, _ = _read_gray_thumbnail(moving_path, 512)
        with torch.no_grad():
            pred = matcher({"image0": fixed.to(device), "image1": moving.to(device)})
            pred = {key: value[0].detach().cpu().numpy()
                    for key, value in pred.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        match = select_ransac_matches(pred["keypoints0"], pred["keypoints1"],
                                      pred["matches0"], width=512, height=512)
        row = {"name": name, "fixed": str(fixed_path),
               "moving": str(moving_path), "output": str(output),
               "fixed_keypoints": int(len(pred["keypoints0"])),
               "moving_keypoints": int(len(pred["keypoints1"])),
               "raw_matches": match["raw_matches"],
               "ransac_inliers": match["ransac_inliers"],
               "occupied_quadrants_4x4": match["occupied_quadrants_4x4"],
               "status": match["status"]}
        if match["status"] == "ok":
            source = np.asarray(match["source_points_unit"], dtype=np.float64)
            target = np.asarray(match["target_points_unit"], dtype=np.float64)
            try:
                matrix64, offset64 = similarity_from_inlier_units(source, target)
                matrix = matrix64.astype(np.float32)
                offset = offset64.astype(np.float32)
                a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
                if a * d - b * c <= 0:
                    raise ValueError("rounded similarity loses positive determinant")
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, post_affine_matrix=matrix,
                                    post_affine_offset=offset,
                                    affine_source="direct_image_superglue")
                residual = np.column_stack((
                    matrix64[0, 0] * source[:, 0] + matrix64[0, 1] * source[:, 1],
                    matrix64[1, 0] * source[:, 0] + matrix64[1, 1] * source[:, 1],
                )) + offset64 - target
                row.update({
                    "matrix": matrix.tolist(), "offset": offset.tolist(),
                    "determinant": float(matrix64[0, 0] * matrix64[1, 1]
                                         - matrix64[0, 1] * matrix64[1, 0]),
                    "inlier_fit_rmse_512px": float(np.sqrt(np.mean(np.sum(
                        (residual * 512) ** 2, axis=1)))),
                })
            except ValueError as error:
                row["status"] = "degenerate_similarity"
                row["error"] = str(error)
        row["decode_match_ransac_fit_seconds"] = time.perf_counter() - started
        rows.append(row)
    result = {
        "method": "one reused DHR-weight SuperPoint/SuperGlue directly on raw padded canvases; RANSAC 6px; positive Procrustes similarity",
        "model_build_and_load_seconds": model_seconds,
        "match_threshold": match_threshold,
        "device": device_name, "rows": rows,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "landmarks_initial_DHR_affine_full_DHR_field_loaded": False,
        "orientation_preserving_only_when_status_ok": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", nargs=4, metavar=("NAME", "FIXED", "MOVING", "OUTPUT"),
                        required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--match-threshold", type=float, default=.3)
    args = parser.parse_args()
    pairs = [(name, Path(fixed), Path(moving), Path(out))
             for name, fixed, moving, out in args.pair]
    result = extract(pairs, args.report, device_name=args.device,
                     match_threshold=args.match_threshold)
    print(json.dumps({"model_build_and_load_seconds": result["model_build_and_load_seconds"],
                      "rows": [{key: row[key] for key in (
                          "name", "status", "raw_matches", "ransac_inliers",
                          "occupied_quadrants_4x4", "decode_match_ransac_fit_seconds",
                      )} for row in result["rows"]]}))


if __name__ == "__main__":
    main()
