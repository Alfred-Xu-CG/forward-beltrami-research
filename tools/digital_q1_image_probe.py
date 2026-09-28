"""Small deterministic end-to-end Q1 image-to-latent training probe.

This is an integration/performance test on analytic synthetic image pairs,
not real pathology registration or held-out generalization evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense import HybridPatchSeedVertexQ1Pyramid
from qcopt.neural_bijection.dense.digital_q1 import (
    q1_corner_determinants,
    validate_q1_map,
)
from qcopt.neural_bijection.dense.forward_p1_encoder import ForwardP1ImageEncoder
from qcopt.neural_bijection.dense.q1_image_sampling import (
    fixed_pixel_centers,
    warp_moving_at_q1_map,
)


def _level_sides(final_side: int) -> tuple[int, ...]:
    if final_side < 17:
        raise ValueError("final_side must be at least 17")
    sides = []
    side = 17
    while side < final_side:
        side = 2 * side - 1
        sides.append(side)
    if side != final_side:
        raise ValueError("final_side must be reachable from 17 by dyadic refinement")
    return tuple(sides)


def run_probe(
    *, final_side: int, image_side: int, width: int, steps: int,
    device: str,
) -> dict[str, int | float | str | bool | None]:
    """Train all encoder heads against a generated same-modality image pair."""
    levels = _level_sides(final_side)
    if image_side < 4 or width < 2 or steps < 1:
        raise ValueError("image_side>=4, width>=2 and steps>=1 are required")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.manual_seed(290929)
    dtype = torch.float32
    q = fixed_pixel_centers(
        image_side, image_side, dtype=dtype, device=target_device,
    )
    x, y = q[..., 0], q[..., 1]
    texture = (
        .35 * torch.sin(8 * math.pi * x + 3 * math.pi * y)
        + .30 * torch.cos(2 * math.pi * x - 9 * math.pi * y)
        + .45 * torch.exp(-((x - .43).square() + (y - .62).square()) / .02)
    )
    moving = texture[:, None]
    interior = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    true_map = torch.stack((x + .015 * interior, y - .010 * interior), dim=-1)
    fixed = F.grid_sample(
        moving, 2 * true_map - 1, mode="bilinear",
        padding_mode="border", align_corners=False,
    ).detach()
    encoder = ForwardP1ImageEncoder(
        17, levels, seed_passes=4, feature_side=min(final_side, 257),
        width=width,
    ).to(target_device)
    decoder = HybridPatchSeedVertexQ1Pyramid(17, final_side, patch_cells=4).to(target_device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    losses: list[float] = []
    durations: list[float] = []
    finite_gradient_steps = 0
    nonzero_gradient_steps = 0
    for _ in range(steps):
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        seed, fine = encoder(fixed, moving)
        map_vertices = decoder(seed, fine)
        warped = warp_moving_at_q1_map(
            moving, map_vertices, height=image_side, width=image_side,
        )
        loss = F.mse_loss(warped, fixed)
        loss.backward()
        gradients = [parameter.grad for parameter in encoder.parameters()
                     if parameter.grad is not None]
        finite_gradient_steps += int(bool(gradients) and all(torch.isfinite(g).all() for g in gradients))
        nonzero_gradient_steps += int(any(bool((g != 0).any()) for g in gradients))
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
    with torch.no_grad():
        seed, fine = encoder(fixed, moving)
        map_vertices = decoder(seed, fine)
        axis = torch.arange(final_side, device=target_device, dtype=dtype) / (final_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1)[None]
        report = validate_q1_map(map_vertices, identity)
        corner_min = float(q1_corner_determinants(map_vertices).amin())
    return {
        "device": str(target_device),
        "final_side": final_side,
        "control_vertices": final_side * final_side,
        "image_pixels": image_side * image_side,
        "encoder_width": width,
        "steps": steps,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "step_seconds_median": statistics.median(durations),
        "finite_gradient_steps": finite_gradient_steps,
        "nonzero_gradient_steps": nonzero_gradient_steps,
        "minimum_raw_corner": corner_min,
        "minimum_normalized_jacobian": corner_min * (final_side - 1) ** 2,
        "nonpositive_corners": report["nonpositive_corners"],
        "boundary_ordered_rectangle": report["boundary_ordered_rectangle"],
        "boundary_max_error": report["boundary_max_error"],
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(target_device)
            if target_device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run_probe(
        final_side=args.final_side, image_side=args.image_side,
        width=args.width, steps=args.steps, device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
