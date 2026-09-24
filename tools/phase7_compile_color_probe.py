"""Test compilation of the safe F1 color kernel, including its backward."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense.colored_vertex_relaxation import (
    SafeColoredVertexRelaxation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.side < 3 or args.batch < 1 or args.repeats < 1:
        raise ValueError("side >= 3, batch >= 1 and repeats >= 1 are required")
    axis = torch.arange(args.side, device=device, dtype=torch.float32) / (args.side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None].expand(args.batch, -1, -1, -1)
    generator = torch.Generator(device=device).manual_seed(713)
    logits = .6 * torch.randn(
        args.batch, args.side - 2, args.side - 2, 2,
        device=device, generator=generator,
    )
    cotangent = torch.randn(
        args.batch, args.side, args.side, 2,
        device=device, generator=generator,
    )
    floor = base.new_full((args.batch,), .05 / (args.side - 1) ** 2)

    def trial(compiled: bool):
        layer = SafeColoredVertexRelaxation(
            args.side, motion_mode="radial", raw_span=2.0,
            index_mode="generated", checkpoint_colors=True,
        ).to(device)
        if compiled:
            layer._update_color = torch.compile(layer._update_color)
        state = base.detach().clone().requires_grad_()
        latent = logits.detach().clone().requires_grad_()
        began = time.perf_counter()
        output = layer(state, latent, area_floor=floor)
        (output * cotangent).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        startup_seconds = time.perf_counter() - began
        times = []
        peaks = []
        reserved = []
        for _ in range(args.repeats):
            state.grad = None
            latent.grad = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            output = layer(state, latent, area_floor=floor)
            (output * cotangent).sum().backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peaks.append(torch.cuda.max_memory_allocated(device))
                reserved.append(torch.cuda.max_memory_reserved(device))
            times.append(time.perf_counter() - began)
        square_a = output[:, :-1, :-1]
        square_b = output[:, :-1, 1:]
        square_c = output[:, 1:, 1:]
        square_d = output[:, 1:, :-1]
        def cross(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        min_jacobian = float(torch.minimum(
            cross(square_b - square_a, square_c - square_a).amin(),
            cross(square_c - square_a, square_d - square_a).amin(),
        ) * (args.side - 1) ** 2)
        return {
            "compiled": compiled,
            "startup_seconds": startup_seconds,
            "median_hot_seconds": statistics.median(times),
            "hot_seconds": times,
            "peak_cuda_allocated_bytes": max(peaks) if peaks else None,
            "peak_cuda_reserved_bytes": max(reserved) if reserved else None,
            "minimum_jacobian": min_jacobian,
            "output": output.detach().cpu(),
            "base_gradient": state.grad.detach().cpu(),
            "logit_gradient": latent.grad.detach().cpu(),
        }

    eager = trial(False)
    compiled = trial(True)
    comparison = {
        "output_max_abs": float((eager["output"] - compiled["output"]).abs().amax()),
        "output_rmse": float((eager["output"] - compiled["output"]).square().mean().sqrt()),
        "base_gradient_max_abs": float(
            (eager["base_gradient"] - compiled["base_gradient"]).abs().amax()
        ),
        "base_gradient_rmse": float(
            (eager["base_gradient"] - compiled["base_gradient"]).square().mean().sqrt()
        ),
        "base_gradient_max_reference": float(eager["base_gradient"].abs().amax()),
        "logit_gradient_max_abs": float(
            (eager["logit_gradient"] - compiled["logit_gradient"]).abs().amax()
        ),
        "logit_gradient_rmse": float(
            (eager["logit_gradient"] - compiled["logit_gradient"]).square().mean().sqrt()
        ),
        "logit_gradient_max_reference": float(eager["logit_gradient"].abs().amax()),
    }
    for result in (eager, compiled):
        for key in ("output", "base_gradient", "logit_gradient"):
            del result[key]
    print(json.dumps({
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "batch": args.batch,
        "device": str(device),
        "torch_version": torch.__version__,
        "eager": eager,
        "compiled": compiled,
        "comparison": comparison,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
