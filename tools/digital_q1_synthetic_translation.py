"""Narrow image-encoder sanity test on known translated pathology textures.

The moving image is generated from the *same* source texture. This is not a
cross-stain registration benchmark and uses no anatomical landmarks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_q1_network_teacher import _identity, _save_checkpoint, _save_map
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def make_translated_moving(fixed: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Return M_t(u)=F(u-t), where t is in unit-square x/y coordinates."""
    if fixed.ndim != 4 or fixed.shape[1] != 1 or shifts.shape != (fixed.shape[0], 2):
        raise ValueError("expected BCHW grayscale and Bx2 shifts")
    if not bool(torch.isfinite(fixed).all()) or not bool(torch.isfinite(shifts).all()):
        raise ValueError("nonfinite images or shifts")
    height, width = fixed.shape[-2:]
    y = torch.linspace(-1, 1, height, dtype=fixed.dtype, device=fixed.device)
    x = torch.linspace(-1, 1, width, dtype=fixed.dtype, device=fixed.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    grid = identity - 2 * shifts.to(device=fixed.device, dtype=fixed.dtype)[:, None, None, :]
    return F.grid_sample(fixed, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def translation_target(side: int, shifts: torch.Tensor) -> torch.Tensor:
    if side < 2 or shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("side>=2 and Bx2 shifts required")
    return _identity(side, shifts.device).to(shifts.dtype) + shifts[:, None, None, :]


def _draw_shifts(batch: int, generator: torch.Generator, bound: float) -> torch.Tensor:
    return (2 * torch.rand((batch, 2), generator=generator) - 1) * bound


@torch.no_grad()
def evaluate(
    model: Q1ImageRegistrationNetwork, texture: torch.Tensor, *,
    samples: int, batch_size: int, seed: int, shift_bound: float,
    output_map: Path,
) -> dict:
    device = next(model.parameters()).device
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    errors, identity_errors, shuffled_errors, blank_errors, offsets = [], [], [], [], []
    nonpositive = 0
    worst_boundary_error = 0.0
    minimum_affine_det = float("inf")
    saved_example = False
    for begin in range(0, samples, batch_size):
        count = min(batch_size, samples - begin)
        shifts = _draw_shifts(count, rng, shift_bound).to(device)
        fixed = texture.to(device).expand(count, -1, -1, -1)
        moving = make_translated_moving(fixed, shifts)
        target = translation_target(model.decoder.final_side, shifts)
        residual, matrix, offset = model(fixed, moving)
        mapped = model.apply_affine(residual, matrix, offset)
        permuted = model.apply_affine(*model(fixed, moving.roll(1, 0)))
        blank = torch.zeros_like(fixed)
        blank_map = model.apply_affine(*model(blank, blank))
        errors.append((mapped - target).square().sum(-1).mean((-2, -1)).cpu())
        identity_errors.append((translation_target(model.decoder.final_side,
                            torch.zeros_like(shifts)) - target).square().sum(-1).mean((-2, -1)).cpu())
        shuffled_errors.append((permuted - target).square().sum(-1).mean((-2, -1)).cpu())
        blank_errors.append((blank_map - target).square().sum(-1).mean((-2, -1)).cpu())
        offsets.append((offset - shifts).square().sum(-1).cpu())
        boundary = _identity(model.decoder.final_side, device).expand(count, -1, -1, -1)
        audit = validate_q1_map(residual, boundary)
        nonpositive += audit["nonpositive_corners"]
        worst_boundary_error = max(worst_boundary_error, audit["boundary_max_error"])
        minimum_affine_det = min(minimum_affine_det, float(torch.linalg.det(matrix).min()))
        if not saved_example:
            output_map.parent.mkdir(parents=True, exist_ok=True)
            certificate = _save_map(output_map, (residual[:1], matrix[:1], offset[:1]))
            saved_example = True
    metric = lambda parts: float(torch.cat(parts).mean().sqrt())
    return {
        "heldout_examples": samples,
        "heldout_map_vector_rmse": metric(errors),
        "identity_vector_rmse": metric(identity_errors),
        "moving_image_shuffled_rmse": metric(shuffled_errors),
        "blank_pair_rmse": metric(blank_errors),
        "affine_offset_vector_rmse": metric(offsets),
        "nonpositive_residual_corners": nonpositive,
        "residual_boundary_max_error": worst_boundary_error,
        "minimum_affine_determinant": minimum_affine_det,
        "saved_example_certificate": certificate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture", type=Path, action="append", required=True,
                        help="exactly three JPEGs from three different development sources")
    parser.add_argument("--held-out", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--image-side", type=int, default=128)
    parser.add_argument("--control-side", type=int, default=257)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--shift-bound", type=float, default=.06)
    parser.add_argument("--seed", type=int, default=290929)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.texture) != 3 or args.steps < 1 or args.batch < 2 or not 0 < args.shift_bound < .2:
        raise ValueError("three textures, positive steps, batch>=2, and shift_bound in (0,.2) required")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    images = [_read_gray_thumbnail(path, args.image_side)[0] for path in args.texture]
    textures = torch.cat(images).to(device)
    train_ids = [index for index in range(3) if index != args.held_out]
    model = Q1ImageRegistrationNetwork(
        seed_side=17, final_side=args.control_side, feature_side=args.image_side,
        width=args.width, flow_hint=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = torch.Generator().manual_seed(args.seed + 7)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    finite_steps = 0
    first_loss = None
    last_loss = None
    best_loss = float("inf")
    for step in range(args.steps):
        chosen = torch.randint(0, 2, (args.batch,), generator=rng)
        shifts = _draw_shifts(args.batch, rng, args.shift_bound).to(device)
        fixed = torch.stack([textures[train_ids[int(index)]] for index in chosen])
        moving = make_translated_moving(fixed, shifts)
        target = translation_target(args.control_side, shifts)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = model.apply_affine(*model(fixed, moving))
        loss = (mapped - target).square().sum(-1).mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        finite = all(gradient is not None and bool(torch.isfinite(gradient).all())
                     for gradient in gradients)
        if not finite:
            raise FloatingPointError(f"nonfinite network gradient at step {step}")
        finite_steps += 1
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        best_loss = min(best_loss, value)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    assessment = evaluate(model, textures[args.held_out:args.held_out + 1],
                          samples=64, batch_size=8, seed=args.seed + 100000,
                          shift_bound=args.shift_bound, output_map=args.output_map)
    report = {
        "mode": "same_source_synthetic_translation_not_real_cross_stain_G2",
        "texture_files": [str(path) for path in args.texture],
        "train_source_indices": train_ids,
        "heldout_source_index": args.held_out,
        "image_side": args.image_side,
        "control_side": args.control_side,
        "steps": args.steps,
        "batch": args.batch,
        "width": args.width,
        "learning_rate": args.learning_rate,
        "shift_bound_unit_coordinates": args.shift_bound,
        "seed": args.seed,
        "device": str(device),
        "first_train_vector_rmse": first_loss ** .5,
        "last_train_vector_rmse": last_loss ** .5,
        "best_train_vector_rmse": best_loss ** .5,
        "finite_gradient_steps": finite_steps,
        "median_complete_training_step_seconds": statistics.median(durations),
        "training_cuda_peak_allocated_bytes": training_peak,
        **assessment,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(args.output_weights, model)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
