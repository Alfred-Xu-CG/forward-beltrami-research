"""Measure frozen Q1 image-network inference, excluding JPEG I/O and certificates.

The second timing includes encoder, safe 257-grid decoder, affine application,
Q1 map query at 512-square pixel centers, and the final moving-image warp.
It times a materialized float tensor for sampling; topology claims refer to
the separately saved/certified residual-plus-affine representation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from qcopt.neural_bijection.dense.q1_image_sampling import warp_moving_at_q1_map
from tools.digital_q1_network_teacher import load_checkpoint
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def benchmark_tensors(
    model: Q1ImageRegistrationNetwork, fixed: torch.Tensor,
    moving: torch.Tensor, *, warmups: int = 3, repeats: int = 20,
) -> dict:
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
        raise ValueError("matching BCHW grayscale inputs required")
    if warmups < 0 or repeats < 1:
        raise ValueError("invalid warmups or repeats")
    device = next(model.parameters()).device
    fixed = fixed.to(device=device, dtype=torch.float32)
    moving = moving.to(device=device, dtype=torch.float32)
    model.eval()
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None

    def forward_only() -> torch.Tensor:
        residual, matrix, offset = model(fixed, moving)
        return model.apply_affine(residual, matrix, offset)

    def forward_and_warp() -> torch.Tensor:
        mapped = forward_only()
        return warp_moving_at_q1_map(
            moving, mapped, height=fixed.shape[-2], width=fixed.shape[-1],
        )

    with torch.no_grad():
        for _ in range(warmups):
            forward_and_warp()
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times: dict[str, list[float]] = {"forward": [], "forward_plus_warp": []}
        for name, operation in (("forward", forward_only),
                                ("forward_plus_warp", forward_and_warp)):
            for _ in range(repeats):
                synchronize()
                started = time.perf_counter()
                output = operation()
                synchronize()
                assert bool(torch.isfinite(output).all())
                times[name].append(time.perf_counter() - started)
    return {
        "device": str(device),
        "batch": int(fixed.shape[0]),
        "control_vertices": int(model.decoder.final_side ** 2),
        "image_pixels": int(fixed.shape[-2] * fixed.shape[-1]),
        "warmups": warmups,
        "repeats": repeats,
        "forward_median_seconds": statistics.median(times["forward"]),
        "forward_plus_image_warp_median_seconds": statistics.median(times["forward_plus_warp"]),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }


def benchmark_vjp_tensors(
    model: Q1ImageRegistrationNetwork, fixed: torch.Tensor,
    moving: torch.Tensor, *, warmups: int = 3, repeats: int = 10,
) -> dict:
    """Time a full image-loss forward and VJP to images and model weights."""
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
        raise ValueError("matching BCHW grayscale inputs required")
    if warmups < 0 or repeats < 1:
        raise ValueError("invalid warmups or repeats")
    device = next(model.parameters()).device
    fixed = fixed.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
    moving = moving.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
    model.eval()
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None

    def one_step() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        fixed.grad = None
        moving.grad = None
        synchronize()
        started = time.perf_counter()
        residual, matrix, offset = model(fixed, moving)
        mapped = model.apply_affine(residual, matrix, offset)
        warped = warp_moving_at_q1_map(
            moving, mapped, height=fixed.shape[-2], width=fixed.shape[-1],
        )
        loss = (warped - fixed).square().mean()
        synchronize()
        forward = time.perf_counter() - started
        loss.backward()
        synchronize()
        return forward, time.perf_counter() - started

    for _ in range(warmups):
        one_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = [one_step() for _ in range(repeats)]
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    return {
        "device": str(device),
        "batch": int(fixed.shape[0]),
        "control_vertices": int(model.decoder.final_side ** 2),
        "image_pixels": int(fixed.shape[-2] * fixed.shape[-1]),
        "warmups": warmups,
        "repeats": repeats,
        "forward_with_image_loss_median_seconds": statistics.median(t[0] for t in times),
        "forward_vjp_median_seconds": statistics.median(t[1] for t in times),
        "input_gradients_finite": bool(
            fixed.grad is not None and moving.grad is not None
            and torch.isfinite(fixed.grad).all() and torch.isfinite(moving.grad).all()
        ),
        "model_gradients_finite": bool(
            gradients and all(g is not None and bool(torch.isfinite(g).all())
                              for g in gradients)
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--measure-vjp", action="store_true")
    parser.add_argument("--output", type=Path,
                        help="optional JSON report path for reproducible timing records")
    args = parser.parse_args()
    fixed, _ = _read_gray_thumbnail(args.fixed_image, args.image_side)
    moving, _ = _read_gray_thumbnail(args.moving_image, args.image_side)
    model = load_checkpoint(args.weights, device=args.device)
    report = benchmark_tensors(
        model, fixed, moving, warmups=args.warmups, repeats=args.repeats,
    )
    if args.measure_vjp:
        report["vjp"] = benchmark_vjp_tensors(
            model, fixed, moving, warmups=args.warmups, repeats=args.repeats,
        )
    report["excluded"] = "JPEG decoding, saved-map certificate, CPU host transfer, landmark inversion"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
