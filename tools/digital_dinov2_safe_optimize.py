"""Frozen DINOv2 patch-descriptor image-only safe P1 optimization probe.

Official DINOv2 ViT-S/14 weights supply the image descriptors. Anatomical
landmarks and full DHR displacement are not read by this program.
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

from qcopt.neural_bijection.dense.digital_q1 import q1_dyadic_refine, validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_mind_objective_probe import sampled_p1_descriptor
from tools.digital_mind_safe_optimize import SafeThreeLevelF1, strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def frozen_patch_features(backbone: torch.nn.Module,
                          image_inverted: torch.Tensor) -> torch.Tensor:
    """DINO RGB-normalized 448² input -> unit 32² patch vectors."""
    ordinary_gray = 1. - F.interpolate(image_inverted, size=(448, 448), mode="area")
    rgb = ordinary_gray.expand(-1, 3, -1, -1)
    mean = torch.tensor((.485, .456, .406), device=rgb.device, dtype=rgb.dtype)[None, :, None, None]
    std = torch.tensor((.229, .224, .225), device=rgb.device, dtype=rgb.dtype)[None, :, None, None]
    normalized = (rgb - mean) / std
    tokens = backbone.forward_features(normalized)["x_norm_patchtokens"]
    features = tokens.transpose(1, 2).reshape(rgb.shape[0], -1, 32, 32)
    return F.normalize(features, dim=1)


def patch_loss(fixed: torch.Tensor, moving: torch.Tensor, mask: torch.Tensor,
               mapped: torch.Tensor) -> torch.Tensor:
    warped = F.normalize(sampled_p1_descriptor(moving, mapped), dim=1)
    cosine_gap = 1. - (fixed * warped).sum(dim=1, keepdim=True)
    return (cosine_gap * mask).sum() / (mask.sum() + 1e-8)


def optimize(fixed_path: Path, moving_path: Path, affine_path: Path,
             output: Path, *, steps: int, device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1:
        raise ValueError("steps must be positive")
    device = torch.device(device_name)
    with np.load(affine_path) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not np.isfinite(matrix).all() or not np.isfinite(offset).all()
    ):
        raise ValueError("invalid affine")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    affine_matrix = torch.tensor(matrix, device=device)
    affine_offset = torch.tensor(offset, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_features = time.perf_counter()
    backbone = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
    backbone = backbone.to(device).eval()
    with torch.no_grad():
        aligned = warp_moving_to_fixed(moving, affine_matrix, affine_offset,
                                       height=512, width=512)
        fixed_feature = frozen_patch_features(backbone, fixed)
        moving_feature = frozen_patch_features(backbone, aligned)
        mask = (F.interpolate(fixed, size=(32, 32), mode="area") > .04).to(fixed.dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    feature_seconds_including_model_load = time.perf_counter() - start_features
    del backbone

    model = SafeThreeLevelF1().to(device)
    optimizer = torch.optim.Adam(model.logits, lr=.02)
    with torch.no_grad():
        initial_image = float(patch_loss(fixed_feature, moving_feature, mask,
                                          identity_vertices(65, device=device)))
    best_objective = float("inf")
    best_map = None
    best_step = -1
    trace = []
    durations = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        mapped = model()
        image = patch_loss(fixed_feature, moving_feature, mask, mapped)
        strain = strain_penalty(mapped)
        loss = image + strain
        value = float(loss.detach())
        if value < best_objective:
            best_objective = value
            best_step = step
            best_map = mapped.detach().clone()
            best_image = float(image.detach())
            best_strain = float(strain.detach())
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.logits):
            raise FloatingPointError(f"invalid latent VJP at step {step}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        if step in (0, 4, 9, 24, 49, 99, steps - 1):
            trace.append({"pre_update_step": step, "image": float(image.detach()),
                          "strain": float(strain.detach()), "total": value})
    assert best_map is not None
    exported = q1_dyadic_refine(q1_dyadic_refine(best_map))
    validity = validate_q1_map(exported, identity_vertices(257, device=device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=exported.cpu().numpy(),
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not validity["valid"] or not binary["valid"]:
        raise RuntimeError("saved map failed topology audit")
    result = {"method": "frozen official DINOv2-S/14 32-square patch cosine + safe F1",
              "fixed": str(fixed_path), "moving": str(moving_path),
              "affine": str(affine_path), "map": str(output), "steps": steps,
              "initial_patch_cosine_gap": initial_image,
              "best_pre_update_step": best_step,
              "best_regularized_loss": best_objective,
              "best_image_loss": best_image, "best_strain": best_strain,
              "feature_seconds_including_cached_model_load": feature_seconds_including_model_load,
              "median_optimizer_step_seconds": statistics.median(durations),
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "trace": trace, "saved_binary_certificate": binary,
              "in_memory_validity": validity,
              "landmarks_machine_matches_full_DHR_field_loaded": False,
              "DHR_derived_initial_affine_used": True,
              "representation_pretrained_external": True}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed_image", "moving_image", "affine", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = optimize(args.fixed_image, args.moving_image, args.affine,
                      args.output, steps=args.steps, device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "initial_patch_cosine_gap", "best_image_loss", "best_pre_update_step",
        "feature_seconds_including_cached_model_load", "median_optimizer_step_seconds",
        "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
