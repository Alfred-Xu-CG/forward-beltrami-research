"""End-to-end frozen-onehead plus four-pass residual student P1 prediction.

The image-only affine and discrete SuperGlue/RANSAC matches are external
preprocessing. All neural parameters and continuous selected-match coordinates
are differentiated through this decoder; no saved per-pair map is loaded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_mind_amortized_network import image_features, structural_loss
from tools.digital_mind_dense257_head import (
    DenseSafeHead, dense_features, load_global_base,
)
from tools.digital_mind_match_conditioned257 import gaussian_match_features
from tools.digital_image_force_feature257 import image_force_features
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def predict(base_checkpoint: Path, one_head_checkpoint: Path,
            residual_checkpoint: Path, fixed_path: Path, moving_path: Path,
            affine_path: Path, matches_path: Path | None, output: Path, *,
            device_name: str = "cuda:0", repeats: int = 3) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if repeats < 1:
        raise ValueError("positive repeats required")
    device = torch.device(device_name)
    base = load_global_base(base_checkpoint, device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(True)
    one_saved = torch.load(one_head_checkpoint, map_location=device,
                           weights_only=False)
    one = DenseSafeHead().to(device).eval()
    one.load_state_dict(one_saved["head_state_dict"])
    residual_saved = torch.load(residual_checkpoint, map_location=device,
                                weights_only=False)
    match_channels = int(residual_saved.get("match_channels", 0))
    evidence_mode = residual_saved.get("evidence_mode", "gaussian")
    if (residual_saved.get("frozen_maps") is None or
            match_channels not in (0, 6, 12) or
            (evidence_mode in ("gaussian_force", "gaussian_zeros")
             and match_channels != 12) or
            (evidence_mode in ("force_only", "force_recurrent")
             and match_channels != 6) or
            evidence_mode not in ("gaussian", "gaussian_force", "gaussian_zeros",
                                  "force_only", "force_recurrent") or
            (evidence_mode == "force_recurrent" and
             residual_saved.get("context", "local") == "multilevel_unet")):
        raise ValueError("checkpoint is not a frozen-onehead residual student")
    if (match_channels and evidence_mode not in ("force_only", "force_recurrent")
            and matches_path is None):
        raise ValueError("match-conditioned checkpoint requires matches")
    if evidence_mode in ("force_only", "force_recurrent") and matches_path is not None:
        raise ValueError("force-only inference must not receive matches")
    residual = RecurrentDenseSafeHead(
        passes=4, match_channels=match_channels,
        context=residual_saved.get("context", "local")).to(device).eval()
    residual.load_state_dict(residual_saved["model_state_dict"])
    with np.load(affine_path) as data:
        matrix = np.asarray(data["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(data["post_affine_offset"], dtype=np.float32)
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    source = target = None
    match_matrix = match_offset = None
    if matches_path is not None and match_channels:
        with np.load(matches_path) as data:
            source_data = np.asarray(data["source_fixed_unit"], dtype=np.float32)
            target_data = np.asarray(data["target_aligned_unit"], dtype=np.float32)
            match_matrix = (data["post_affine_matrix"].copy()
                            if "post_affine_matrix" in data else None)
            match_offset = (data["post_affine_offset"].copy()
                            if "post_affine_offset" in data else None)
        if (source_data.shape != target_data.shape or source_data.ndim != 2 or
                source_data.shape[1] != 2 or len(source_data) < 16 or
                not np.isfinite(source_data).all() or
                not np.isfinite(target_data).all()):
            raise ValueError("at least 16 finite paired matches required")
        if (match_matrix is None) != (match_offset is None):
            raise ValueError("partial match affine metadata")
        if match_matrix is not None and not (
            np.array_equal(match_matrix, matrix) and np.array_equal(match_offset, offset)
        ):
            raise ValueError("match archive affine frame differs")
        source = torch.from_numpy(source_data.copy()).to(device).requires_grad_(True)
        target = torch.from_numpy(target_data.copy()).to(device).requires_grad_(True)
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    if evidence_mode in ("force_only", "force_recurrent"):
        fixed.requires_grad_(True)
        moving.requires_grad_(True)
    m = torch.tensor(matrix, device=device)
    shift = torch.tensor(offset, device=device)
    reference = identity_vertices(257, device=device)

    def prepare():
        aligned = warp_moving_to_fixed(moving, m, shift, height=512, width=512)
        feature, _, _, _ = image_features(
            F.interpolate(fixed, size=(128, 128), mode="area"),
            F.interpolate(aligned, size=(128, 128), mode="area"))
        raster = (gaussian_match_features(source, target)
                  if match_channels and source is not None else None)
        return aligned, feature, raster

    def forward(aligned, feature, raster):
        coarse = base(feature, final_side=257)
        fine_old, _, _, _ = dense_features(
            feature, fixed, aligned, coarse,
            feature_geometry=one_saved.get("feature_geometry", "legacy_align_corners"))
        frozen = one(coarse, fine_old)
        fine_new, fdesc, mdesc, mask = dense_features(
            feature, fixed, aligned, frozen,
            feature_geometry=residual_saved.get(
                "residual_feature_geometry", "legacy_align_corners"))
        evidence = raster
        if evidence_mode == "gaussian_force":
            force = image_force_features(fdesc, mdesc, mask, frozen)
            evidence = torch.cat((raster, force), dim=1)
        elif evidence_mode == "gaussian_zeros":
            evidence = torch.cat((raster, torch.zeros_like(raster)), dim=1)
        elif evidence_mode == "force_only":
            evidence = image_force_features(fdesc, mdesc, mask, frozen)
        if evidence_mode == "force_recurrent":
            mapped = residual(frozen, fine_new, evidence_fn=lambda current:
                              image_force_features(fdesc, mdesc, mask, current))
        else:
            mapped = residual(frozen, fine_new, match_feature=evidence)
        return mapped, frozen, fdesc, mdesc, mask

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        aligned, feature, raster = prepare()
        for _ in range(2):
            forward(aligned, feature, raster)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            aligned, feature, raster = prepare()
            mapped, frozen, fdesc, mdesc, mask = forward(aligned, feature, raster)
            sync()
            times.append(time.perf_counter() - start)
        image = float(structural_loss(fdesc, mdesc, mask, mapped))
        frozen_image = float(structural_loss(fdesc, mdesc, mask, frozen))
        if not validate_q1_map(mapped, reference)["valid"]:
            raise RuntimeError("invalid map in memory")
    forward_peak = (None if device.type != "cuda" else
                    int(torch.cuda.max_memory_allocated(device)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=mapped.cpu().numpy(),
                        boundary_reference=reference.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"]:
        raise RuntimeError("saved map invalid")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times_vjp = []
    finite = True
    for _ in range(repeats):
        base.zero_grad(set_to_none=True)
        one.zero_grad(set_to_none=True)
        residual.zero_grad(set_to_none=True)
        if source is not None:
            source.grad = None
            target.grad = None
        if evidence_mode in ("force_only", "force_recurrent"):
            fixed.grad = None
            moving.grad = None
        sync()
        start = time.perf_counter()
        aligned, feature, raster = prepare()
        mapped, _, fdesc, mdesc, mask = forward(aligned, feature, raster)
        structural_loss(fdesc, mdesc, mask, mapped).backward()
        sync()
        times_vjp.append(time.perf_counter() - start)
        finite &= all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                      for p in (*base.parameters(), *one.parameters(),
                                *residual.parameters()))
        if source is not None:
            finite &= (source.grad is not None and target.grad is not None and
                       bool(torch.isfinite(source.grad).all()) and
                       bool(torch.isfinite(target.grad).all()))
        if evidence_mode in ("force_only", "force_recurrent"):
            finite &= (fixed.grad is not None and moving.grad is not None and
                       bool(torch.isfinite(fixed.grad).all()) and
                       bool(torch.isfinite(moving.grad).all()))
    vjp_peak = (None if device.type != "cuda" else
                int(torch.cuda.max_memory_allocated(device)))
    report = {
        "method": "full frozen image-base plus 257 onehead plus four safe residual passes",
        "evidence_mode": evidence_mode,
        "base_checkpoint": str(base_checkpoint),
        "one_head_checkpoint": str(one_head_checkpoint),
        "residual_checkpoint": str(residual_checkpoint),
        "residual_context": residual.context,
        "residual_feature_geometry": residual_saved.get(
            "residual_feature_geometry", "legacy_align_corners"),
        "fixed": str(fixed_path), "moving": str(moving_path),
        "affine": str(affine_path),
        "matches": None if matches_path is None else str(matches_path),
        "match_count": 0 if source is None else len(source), "output": str(output),
        "frozen_image": frozen_image, "residual_image": image,
        "full_forward_seconds_median": statistics.median(times),
        "full_forward_and_parameter_match_vjp_seconds_median":
            statistics.median(times_vjp),
        "forward_peak_torch_cuda_allocated_bytes": forward_peak,
        "vjp_peak_torch_cuda_allocated_bytes": vjp_peak,
        "finite_full_vjp": finite,
        "image_pixels_in_vjp_check": evidence_mode in (
            "force_only", "force_recurrent"),
        "saved_binary_certificate": binary,
        "anatomical_landmarks_or_DHR_full_field_loaded": False,
        "external_matcher_affine_excluded_from_time_and_gradient": True,
        "match_affine_frame_machine_checked": (
            None if source is None else match_matrix is not None),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base_checkpoint", "one_head_checkpoint",
                 "residual_checkpoint", "fixed", "moving", "affine", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = predict(args.base_checkpoint, args.one_head_checkpoint,
                     args.residual_checkpoint, args.fixed, args.moving,
                     args.affine, args.matches, args.output,
                     device_name=args.device, repeats=args.repeats)
    print(json.dumps({key: result[key] for key in (
        "frozen_image", "residual_image", "full_forward_seconds_median",
        "full_forward_and_parameter_match_vjp_seconds_median",
        "forward_peak_torch_cuda_allocated_bytes",
        "vjp_peak_torch_cuda_allocated_bytes", "finite_full_vjp",
    )}))


if __name__ == "__main__":
    main()
