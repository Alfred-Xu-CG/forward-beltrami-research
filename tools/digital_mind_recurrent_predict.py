"""Frozen image-only recurrent safe-257 prediction and full parameter VJP."""

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
from tools.digital_mind_dense257_head import dense_features, load_global_base
from tools.digital_mind_match_conditioned257 import gaussian_match_features
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def predict(base_checkpoint: Path, recurrent_checkpoint: Path,
            fixed_path: Path, moving_path: Path, affine_path: Path,
            output: Path, *, device_name: str, repeats: int = 5,
            matches_path: Path | None = None) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if repeats < 1:
        raise ValueError("positive repeats required")
    device = torch.device(device_name)
    base = load_global_base(base_checkpoint, device)
    for parameter in base.parameters():
        parameter.requires_grad_(True)
    saved = torch.load(recurrent_checkpoint, map_location=device, weights_only=False)
    match_channels = int(saved.get("match_channels", 0))
    model = RecurrentDenseSafeHead(passes=int(saved["passes"]),
                                   match_channels=match_channels).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    if match_channels:
        if matches_path is None:
            raise ValueError("correspondence-conditioned checkpoint requires matches")
        with np.load(matches_path) as matches:
            source = torch.from_numpy(matches["source_fixed_unit"].copy()).to(device)
            target = torch.from_numpy(matches["target_aligned_unit"].copy()).to(device)
        if len(source) < 16 or source.shape != target.shape:
            raise ValueError("at least 16 paired aligned matches required")
    elif matches_path is not None:
        raise ValueError("unconditioned checkpoint does not accept matches")
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with np.load(affine_path) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
        affine_source = (str(archive["affine_source"])
                         if "affine_source" in archive.files else
                         "direct_image_superglue" if any(
                             token in affine_path.stem for token in
                             ("directSG", "direct_superglue"))
                         else "DHR_derived")
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not np.isfinite(matrix).all() or not np.isfinite(offset).all()
    ):
        raise ValueError("finite affine required")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    m = torch.tensor(matrix, device=device)
    shift = torch.tensor(offset, device=device)

    def prepare():
        aligned = warp_moving_to_fixed(moving, m, shift, height=512, width=512)
        feature, _, _, _ = image_features(
            F.interpolate(fixed, size=(128, 128), mode="area"),
            F.interpolate(aligned, size=(128, 128), mode="area"))
        raster = (gaussian_match_features(source, target)
                  if match_channels else None)
        return aligned, feature, raster

    def map_forward(aligned, feature, raster, *, with_passes=False):
        coarse = base(feature, final_side=257)
        fine, fdesc, mdesc, mask = dense_features(
            feature, fixed, aligned, coarse,
            feature_geometry="legacy_align_corners")
        mapped = model(coarse, fine, match_feature=raster,
                       return_passes=with_passes)
        return mapped, coarse, fdesc, mdesc, mask

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        aligned, feature, raster = prepare()
        for _ in range(2):
            map_forward(aligned, feature, raster)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        whole_times, map_times = [], []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            aligned, feature, raster = prepare()
            sync()
            middle = time.perf_counter()
            mapped, coarse, fdesc, mdesc, mask = map_forward(aligned, feature, raster)
            sync()
            end = time.perf_counter()
            whole_times.append(end - start)
            map_times.append(end - middle)
        passes, _, _, _, _ = map_forward(aligned, feature, raster,
                                        with_passes=True)
        image = float(structural_loss(fdesc, mdesc, mask, mapped))
        coarse_image = float(structural_loss(fdesc, mdesc, mask, coarse))
        valid = validate_q1_map(mapped, identity_vertices(257, device=device))
        pass_motion = [float((current - previous).square().sum(-1).mean().sqrt())
                       for current, previous in zip(passes, [coarse] + passes[:-1])]
    forward_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=mapped.cpu().numpy(),
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not valid["valid"] or not binary["valid"]:
        raise RuntimeError("saved recurrent map topology failure")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vjp_times = []
    finite_vjp = True
    finite_match_feature_vjp = True
    for _ in range(repeats):
        base.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        sync()
        started = time.perf_counter()
        differentiable_raster = (raster.detach().requires_grad_(True)
                                 if raster is not None else None)
        mapped, _, fdesc, mdesc, mask = map_forward(
            aligned, feature, differentiable_raster)
        structural_loss(fdesc, mdesc, mask, mapped).backward()
        sync()
        vjp_times.append(time.perf_counter() - started)
        finite_vjp &= all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                          for p in (*base.parameters(), *model.parameters()))
        if differentiable_raster is not None:
            finite_match_feature_vjp &= (
                differentiable_raster.grad is not None and
                bool(torch.isfinite(differentiable_raster.grad).all()))
    vjp_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    report = {
        "method": "global base plus four image-conditioned sequential safe F1 257-square passes",
        "base_checkpoint": str(base_checkpoint),
        "recurrent_checkpoint": str(recurrent_checkpoint),
        "fixed": str(fixed_path), "moving": str(moving_path),
        "affine": str(affine_path), "output": str(output),
        "affine_source": affine_source,
        "matches": None if matches_path is None else str(matches_path),
        "match_count": None if matches_path is None else len(source),
        "passes": model.passes, "per_pass_rms_residual_unit": pass_motion,
        "coarse_256_P1_descriptor_loss": coarse_image,
        "recurrent_256_P1_descriptor_loss": image,
        "prewarp_feature_base_recurrent_seconds_median": statistics.median(whole_times),
        "base_recurrent_only_after_128_feature_seconds_median": statistics.median(map_times),
        "base_recurrent_forward_and_full_parameter_vjp_seconds_median": statistics.median(vjp_times),
        "forward_peak_torch_cuda_allocated_bytes": forward_peak,
        "vjp_peak_torch_cuda_allocated_bytes": vjp_peak,
        "finite_full_parameter_vjp": finite_vjp,
        "finite_match_feature_vjp": finite_match_feature_vjp,
        "saved_binary_certificate": binary,
        "in_memory_validity": valid,
        "landmarks_or_DHR_full_field_loaded": False,
        "timing_excludes_affine_generation_image_decode_save_certificate": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base_checkpoint", "recurrent_checkpoint", "fixed_image",
                 "moving_image", "affine", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--matches", type=Path)
    args = parser.parse_args()
    report = predict(args.base_checkpoint, args.recurrent_checkpoint,
                     args.fixed_image, args.moving_image, args.affine,
                     args.output, device_name=args.device, repeats=args.repeats,
                     matches_path=args.matches)
    print(json.dumps({key: report[key] for key in (
        "passes", "coarse_256_P1_descriptor_loss",
        "recurrent_256_P1_descriptor_loss", "per_pass_rms_residual_unit",
        "prewarp_feature_base_recurrent_seconds_median",
        "base_recurrent_forward_and_full_parameter_vjp_seconds_median",
        "forward_peak_torch_cuda_allocated_bytes",
        "vjp_peak_torch_cuda_allocated_bytes", "finite_full_parameter_vjp",
    )}))


if __name__ == "__main__":
    main()
