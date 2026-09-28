"""Frozen border-cue ablation and classical phase correlation for synthetic shifts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.digital_q1_network_teacher import load_checkpoint
from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.digital_q1_synthetic_translation import (
    _draw_shifts, make_translated_moving, translation_target,
)


def zero_boundary(image: torch.Tensor, margin: int) -> torch.Tensor:
    if image.ndim != 4 or margin < 0 or 2 * margin >= min(image.shape[-2:]):
        raise ValueError("invalid BCHW image or margin")
    masked = image.clone()
    if margin:
        masked[:, :, :margin, :] = 0
        masked[:, :, -margin:, :] = 0
        masked[:, :, :, :margin] = 0
        masked[:, :, :, -margin:] = 0
    return masked


def phase_shift_normalized(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """OpenCV phase shift in normalized x/y pixel-center coordinates."""
    if fixed.ndim != 2 or fixed.shape != moving.shape or min(fixed.shape) < 8:
        raise ValueError("matching 2D images of at least 8 pixels required")
    # OpenCV applies the supplied Hanning window in place; never let that
    # mutate an input tensor view or contaminate the next evaluation batch.
    fixed = np.array(fixed, dtype=np.float32, order="C", copy=True)
    moving = np.array(moving, dtype=np.float32, order="C", copy=True)
    height, width = fixed.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (shift_x, shift_y), _ = cv2.phaseCorrelate(fixed, moving, window)
    return np.asarray([shift_x / (width - 1), shift_y / (height - 1)], dtype=np.float64)


@torch.no_grad()
def evaluate_controls(model, texture: torch.Tensor, *, margin: int,
                      seed: int, samples: int = 64, batch_size: int = 8,
                      shift_bound: float = .06) -> dict:
    device = next(model.parameters()).device
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    original_errors = []
    masked_errors = []
    phase_errors = []
    phase_durations = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        shifts = _draw_shifts(count, rng, shift_bound).to(device)
        fixed = texture.to(device).expand(count, -1, -1, -1)
        moving = make_translated_moving(fixed, shifts)
        target = translation_target(model.decoder.final_side, shifts)
        original = model.apply_affine(*model(fixed, moving))
        masked = model.apply_affine(*model(zero_boundary(fixed, margin),
                                           zero_boundary(moving, margin)))
        original_errors.append((original - target).square().sum(-1).mean((-2, -1)).cpu())
        masked_errors.append((masked - target).square().sum(-1).mean((-2, -1)).cpu())
        fixed_cpu = fixed[:, 0].cpu().numpy()
        moving_cpu = moving[:, 0].cpu().numpy()
        shifts_cpu = shifts.cpu().numpy()
        for index in range(count):
            begin = time.perf_counter()
            estimate = phase_shift_normalized(fixed_cpu[index], moving_cpu[index])
            phase_durations.append(time.perf_counter() - begin)
            phase_errors.append(float(np.square(estimate - shifts_cpu[index]).sum()))
    return {
        "samples": samples,
        "margin_image_pixels": margin,
        "original_model_vector_rmse": float(torch.cat(original_errors).mean().sqrt()),
        "border_zeroed_model_vector_rmse": float(torch.cat(masked_errors).mean().sqrt()),
        "phase_correlation_vector_rmse": float(np.mean(phase_errors) ** .5),
        "phase_correlation_median_seconds_per_pair": statistics.median(phase_durations),
        "phase_correlation_mean_squared_vector_errors": phase_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--texture-array", type=Path,
                        help="optional exact 128-square float32 tensor exported by the training image environment")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--image-side", type=int, default=128)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--seed", type=int, default=290929)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.texture_array is None:
        image, _ = _read_gray_thumbnail(args.texture, args.image_side)
    else:
        array = np.load(args.texture_array)
        if array.shape != (1, 1, args.image_side, args.image_side) or array.dtype != np.float32:
            raise ValueError("texture-array must contain one float32 BCHW image at image-side")
        image = torch.from_numpy(array.copy())
    model = load_checkpoint(args.weights, device=args.device)
    result = evaluate_controls(model, image, margin=args.margin,
                               seed=args.seed + 100000)
    result.update({
        "mode": "post_pilot_synthetic_translation_shortcut_controls_not_real_registration",
        "texture": str(args.texture),
        "texture_array": str(args.texture_array) if args.texture_array else None,
        "weights": str(args.weights),
        "image_side": args.image_side,
        "control_side": model.decoder.final_side,
        "seed": args.seed,
        "device": args.device,
    })
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "phase_correlation_mean_squared_vector_errors"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
