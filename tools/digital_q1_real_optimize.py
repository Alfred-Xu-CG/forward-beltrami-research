"""Development O-mode: image-only per-pair optimisation of safe Q1 latents.

This uses one real pair as a development case, not a training/test split.
The two supplied JPEGs are independently resampled to the same unit-square
pixel-center frame. No landmarks are read by this module. Its output map is
fixed-normalized -> moving-normalized, so moving -> fixed landmark scoring
requires a separately checked inverse and original-size conversion.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qcopt.neural_bijection.dense import HybridPatchSeedVertexQ1Pyramid
from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_sampling import warp_moving_at_q1_map


def _image_correlation_loss(fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
    first = fixed - fixed.mean(dim=(-2, -1), keepdim=True)
    second = warped - warped.mean(dim=(-2, -1), keepdim=True)
    numerator = (first * second).mean(dim=(-2, -1))
    variance = first.square().mean(dim=(-2, -1)) * second.square().mean(dim=(-2, -1))
    return (1 - numerator / (variance + 1e-12).sqrt()).mean()


def optimize_tensors(
    fixed: torch.Tensor, moving: torch.Tensor, *,
    final_side: int, steps: int, learning_rate: float, device: str,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str | None]]:
    """Fit only latent fields; no image encoder or landmark supervision."""
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[0] != 1 or fixed.shape[1] != 1:
        raise ValueError("fixed and moving must be matching B=1,C=1 image tensors")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    target_device = torch.device(device)
    fixed = fixed.to(device=target_device, dtype=torch.float32)
    moving = moving.to(device=target_device, dtype=torch.float32)
    decoder = HybridPatchSeedVertexQ1Pyramid(17, final_side, patch_cells=4).to(target_device)
    seed = torch.nn.ParameterList(
        torch.nn.Parameter(torch.zeros((1, 15, 15, 2), device=target_device))
        for _ in range(decoder.seed_passes)
    )
    levels = torch.nn.ParameterList(
        torch.nn.Parameter(torch.zeros((1, side - 2, side - 2, 2), device=target_device))
        for side in decoder.level_sides
    )
    optimizer = torch.optim.Adam((*seed, *levels), lr=learning_rate)
    axis = torch.arange(final_side, device=target_device, dtype=torch.float32) / (final_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    best_image_loss = float("inf")
    best_map = None
    history: list[float] = []
    durations: list[float] = []
    finite_gradient_steps = 0
    for _ in range(steps):
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = decoder(tuple(seed), tuple(levels))
        warped = warp_moving_at_q1_map(
            moving, mapped, height=fixed.shape[-2], width=fixed.shape[-1],
        )
        image_loss = _image_correlation_loss(fixed, warped)
        loss = image_loss + .05 * (mapped - identity).square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in (*seed, *levels)]
        finite = all(gradient is not None and bool(torch.isfinite(gradient).all())
                     for gradient in gradients)
        finite_gradient_steps += int(finite)
        if not finite:
            raise FloatingPointError("nonfinite or missing latent gradient")
        value = float(image_loss.detach())
        if value < best_image_loss:
            best_image_loss = value
            best_map = mapped.detach().clone()
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        history.append(value)
        durations.append(time.perf_counter() - started)
    assert best_map is not None
    report = validate_q1_map(best_map, identity)
    return best_map, {
        "mode": "O_per_pair_latent_optimization",
        "device": str(target_device),
        "final_side": final_side,
        "image_height": fixed.shape[-2],
        "image_width": fixed.shape[-1],
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_image_loss": history[0],
        "last_image_loss": history[-1],
        "best_image_loss": best_image_loss,
        "step_seconds_median": statistics.median(durations),
        "finite_gradient_steps": finite_gradient_steps,
        "nonpositive_corners": report["nonpositive_corners"],
        "boundary_ordered_rectangle": report["boundary_ordered_rectangle"],
        "boundary_max_error": report["boundary_max_error"],
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
    }


def _read_gray_thumbnail(path: Path, image_side: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        original_size = image.size
        image.draft("RGB", (image_side, image_side))
        array = np.asarray(
            image.resize((image_side, image_side), Image.Resampling.BILINEAR).convert("L"),
            dtype=np.float32,
        )
    return torch.from_numpy((1 - array / 255)[None, None]), original_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--final-side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=.04)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    fixed, fixed_size = _read_gray_thumbnail(args.fixed_image, args.image_side)
    moving, moving_size = _read_gray_thumbnail(args.moving_image, args.image_side)
    mapped, result = optimize_tensors(
        fixed, moving, final_side=args.final_side, steps=args.steps,
        learning_rate=args.learning_rate, device=args.device,
    )
    axis = np.arange(args.final_side, dtype=np.float32) / np.float32(args.final_side - 1)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    reference = np.stack((x, y), axis=-1)[None]
    np.savez_compressed(
        args.output_map, vertices=mapped.cpu().numpy(), boundary_reference=reference,
    )
    certificate = certify_q1_binary_map(args.output_map)
    result.update({
        "moving_image": str(args.moving_image),
        "fixed_image": str(args.fixed_image),
        "moving_original_size_xy": list(moving_size),
        "fixed_original_size_xy": list(fixed_size),
        "normalization": "each whole JPEG independently resampled to unit square",
        "saved_map": str(args.output_map),
        "saved_binary_valid": certificate["valid"],
        "saved_binary_positive_corners": certificate["positive_corners"],
        "saved_binary_nonpositive_corners": certificate["nonpositive_corners"],
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
