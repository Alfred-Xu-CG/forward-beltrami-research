"""Image-only inference and full encoder/head VJP for genuine 257² safe maps."""

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
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def predict(base_checkpoint: Path, dense_checkpoint: Path, fixed_path: Path,
            moving_path: Path, affine_path: Path, output: Path, *,
            device_name: str, repeats: int = 5) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if repeats < 1:
        raise ValueError("positive repeats required")
    device = torch.device(device_name)
    base = load_global_base(base_checkpoint, device)
    for parameter in base.parameters():
        parameter.requires_grad_(True)
    saved = torch.load(dense_checkpoint, map_location=device, weights_only=False)
    feature_geometry = saved.get("feature_geometry", "legacy_align_corners")
    head = DenseSafeHead().to(device)
    head.load_state_dict(saved["head_state_dict"])
    base.eval(), head.eval()
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
    b0 = torch.tensor(offset, device=device)

    def prepare():
        aligned = warp_moving_to_fixed(moving, m, b0, height=512, width=512)
        feature, _, _, _ = image_features(
            F.interpolate(fixed, size=(128, 128), mode="area"),
            F.interpolate(aligned, size=(128, 128), mode="area"),
        )
        return aligned, feature

    def map_forward(aligned, feature):
        coarse = base(feature, final_side=257)
        fine_feature, fixed_desc, moving_desc, mask = dense_features(
            feature, fixed, aligned, coarse,
            feature_geometry=feature_geometry)
        return head(coarse, fine_feature), coarse, fixed_desc, moving_desc, mask

    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    with torch.no_grad():
        aligned, feature = prepare()
        for _ in range(2):
            map_forward(aligned, feature)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        whole_times = []
        map_times = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            aligned, feature = prepare()
            sync()
            middle = time.perf_counter()
            mapped, coarse, fdesc, mdesc, mask = map_forward(aligned, feature)
            sync()
            end = time.perf_counter()
            whole_times.append(end - start)
            map_times.append(end - middle)
        image = float(structural_loss(fdesc, mdesc, mask, mapped))
        coarse_image = float(structural_loss(fdesc, mdesc, mask, coarse))
        valid = validate_q1_map(mapped, identity_vertices(257, device=device))
    forward_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=mapped.cpu().numpy(),
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not valid["valid"] or not binary["valid"]:
        raise RuntimeError("saved map failed topology audit")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vjp_times = []
    finite_vjp = True
    for _ in range(repeats):
        base.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        sync()
        start = time.perf_counter()
        mapped, _, fdesc, mdesc, mask = map_forward(aligned, feature)
        loss = structural_loss(fdesc, mdesc, mask, mapped)
        loss.backward()
        sync()
        vjp_times.append(time.perf_counter() - start)
        finite_vjp &= all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                          for p in (*base.parameters(), *head.parameters()))
    vjp_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    result = {
        "method": "global 65-square base plus learned independently proposed 257-square F1 head",
        "base_checkpoint": str(base_checkpoint),
        "dense_checkpoint": str(dense_checkpoint),
        "feature_geometry": feature_geometry,
        "fixed": str(fixed_path), "moving": str(moving_path),
        "affine": str(affine_path), "map": str(output),
        "base_256_P1_descriptor_loss": coarse_image,
        "dense_256_P1_descriptor_loss": image,
        "prewarp_feature_base_dense_seconds_median": statistics.median(whole_times),
        "base_dense_only_after_128_feature_seconds_median": statistics.median(map_times),
        "base_dense_257_P1_forward_and_full_parameter_vjp_seconds_median": statistics.median(vjp_times),
        "forward_peak_torch_cuda_allocated_bytes": forward_peak,
        "vjp_peak_torch_cuda_allocated_bytes": vjp_peak,
        "finite_full_parameter_vjp": finite_vjp,
        "saved_binary_certificate": binary,
        "in_memory_validity": valid,
        "landmarks_full_DHR_field_machine_matches_loaded": False,
        "DHR_derived_initial_affine_used": affine_source == "DHR_derived",
        "affine_provenance": (
            "direct image-only SuperPoint/SuperGlue similarity"
            if affine_source == "direct_image_superglue" else "DHR-derived saved affine"),
        "timing_excludes_affine_generation_image_decode_save_certificate": True,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base_checkpoint", "dense_checkpoint", "fixed_image",
                 "moving_image", "affine", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = predict(args.base_checkpoint, args.dense_checkpoint,
                     args.fixed_image, args.moving_image, args.affine,
                     args.output, device_name=args.device, repeats=args.repeats)
    print(json.dumps({key: result[key] for key in (
        "base_256_P1_descriptor_loss", "dense_256_P1_descriptor_loss",
        "prewarp_feature_base_dense_seconds_median",
        "base_dense_257_P1_forward_and_full_parameter_vjp_seconds_median",
        "forward_peak_torch_cuda_allocated_bytes", "vjp_peak_torch_cuda_allocated_bytes",
        "finite_full_parameter_vjp",
    )}))


if __name__ == "__main__":
    main()
