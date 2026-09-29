"""Run a trained structural-image safe decoder on one unlabeled image pair."""

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
from tools.digital_mind_amortized_network import (
    MindSafeImageNetwork, image_features, structural_loss,
)
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.digital_affine_prewarp import warp_moving_to_fixed


def predict(checkpoint: Path, fixed_path: Path, moving_path: Path,
            affine_path: Path, output: Path, *, device_name: str,
            repeats: int = 10) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if repeats < 1:
        raise ValueError("positive repeats required")
    device = torch.device(device_name)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MindSafeImageNetwork(architecture=saved.get("architecture", "local")).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with np.load(affine_path) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("invalid initial affine")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("initial affine must preserve orientation")
    m = torch.tensor(matrix, device=device)
    b0 = torch.tensor(offset, device=device)

    def prepare():
        aligned = warp_moving_to_fixed(moving, m, b0, height=512, width=512)
        small_fixed = F.interpolate(fixed, size=(128, 128), mode="area")
        small_moving = F.interpolate(aligned, size=(128, 128), mode="area")
        return image_features(small_fixed, small_moving)

    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    with torch.no_grad():
        features = prepare()
        for _ in range(3):
            model(features[0], final_side=257)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        all_times, net_times = [], []
        for _ in range(repeats):
            sync()
            started = time.perf_counter()
            feature, fixed_desc, moving_desc, mask = prepare()
            sync()
            after_feature = time.perf_counter()
            mapped = model(feature, final_side=257)
            sync()
            after_map = time.perf_counter()
            all_times.append(after_map - started)
            net_times.append(after_map - after_feature)
        fine_loss = float(structural_loss(fixed_desc, moving_desc, mask, mapped))
        affine_loss = float(structural_loss(
            fixed_desc, moving_desc, mask, identity_vertices(65, device=device)))
        validation = validate_q1_map(mapped, identity_vertices(257, device=device))
        vertices = mapped.cpu().numpy()
    forward_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=vertices,
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"] or not validation["valid"]:
        raise RuntimeError("saved P1/Q1 map failed topology audit")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vjp_times = []
    finite_vjp = True
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        sync()
        started = time.perf_counter()
        coarse = model(feature, final_side=65)
        loss = structural_loss(fixed_desc, moving_desc, mask, coarse)
        loss.backward()
        sync()
        vjp_times.append(time.perf_counter() - started)
        finite_vjp &= all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                          for p in model.parameters())
    vjp_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    report = {"checkpoint": str(checkpoint), "fixed": str(fixed_path),
              "moving": str(moving_path), "affine": str(affine_path),
              "map": str(output), "train_case_count": len(saved["train_ids"]),
              "validation_case_count": len(saved["validation_ids"]),
              "DHR_derived_initial_affine_used": True,
              "full_DHR_displacement_matches_landmarks_used": False,
              "affine_only_descriptor_loss_128": affine_loss,
              "predicted_exported_P1_descriptor_loss_128": fine_loss,
              "prewarp_feature_and_network_seconds_median": statistics.median(all_times),
              "network_only_forward_seconds_median": statistics.median(net_times),
              "network_65_P1_forward_plus_parameter_vjp_seconds_median": statistics.median(vjp_times),
              "forward_peak_allocated_bytes": forward_peak,
              "vjp_peak_allocated_bytes": vjp_peak,
              "finite_network_vjp": finite_vjp,
              "saved_binary_certificate": binary,
              "in_memory_validity": validation,
              "timing_excludes_initial_affine_generation_image_decode_save_certificate": True}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.fixed_image, args.moving_image,
                     args.affine, args.output, device_name=args.device,
                     repeats=args.repeats)
    print(json.dumps({key: result[key] for key in (
        "affine_only_descriptor_loss_128", "predicted_exported_P1_descriptor_loss_128",
        "prewarp_feature_and_network_seconds_median", "network_only_forward_seconds_median",
        "finite_network_vjp",
    )}))


if __name__ == "__main__":
    main()
