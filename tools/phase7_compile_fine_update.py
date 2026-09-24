"""Compare eager and torch.compile for a topology-safe fine P1 update.

Compile time is reported separately; no speed claim includes that startup cost.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import SafeColoredVertexRelaxation
from phase7_multiscale_fiber_reachability import minimum_jacobian


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--mode", choices=["default", "reduce-overhead",
                                           "max-autotune"], default="default")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    axis = torch.linspace(0, 1, args.side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None].expand(
        args.batch, -1, -1, -1).contiguous()
    generator = torch.Generator(device=device).manual_seed(38171)
    latent = .2 * torch.randn(
        args.batch, args.side - 2, args.side - 2, 2,
        device=device, dtype=dtype, generator=generator)
    cotangent = torch.randn(
        args.batch, args.side, args.side, 2,
        device=device, dtype=dtype, generator=generator)
    area_floor = identity.new_full(
        (args.batch,), .05 / (args.side - 1) ** 2)
    eager = SafeColoredVertexRelaxation(
        args.side, safety_fraction=.85, motion_mode="radial",
        raw_span=2., floor_fraction=0.,
    ).to(device)
    compiled = torch.compile(eager, mode=args.mode, fullgraph=False)

    def run(layer, warmup: int) -> tuple[torch.Tensor, torch.Tensor, list[float], int]:
        z = latent.detach().clone().requires_grad_()
        for _ in range(warmup):
            y = layer(identity, z, area_floor=area_floor)
            grad, = torch.autograd.grad((y * cotangent).sum(), z)
        times = []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeat):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            y = layer(identity, z, area_floor=area_floor)
            grad, = torch.autograd.grad((y * cotangent).sum(), z)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
        return y.detach(), grad.detach(), times, torch.cuda.max_memory_allocated(device)

    eager_y, eager_g, eager_times, eager_peak = run(eager, 2)
    torch.cuda.synchronize(device)
    tick = time.perf_counter()
    compiled_y, compiled_g, compiled_times, compiled_peak = run(compiled, 2)
    torch.cuda.synchronize(device)
    compile_plus_warm_seconds = time.perf_counter() - tick - sum(compiled_times)
    print(json.dumps({
        "experiment": "phase7_compile_fine_update",
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "batch": args.batch,
        "dtype": args.dtype,
        "mode": args.mode,
        "repeat": args.repeat,
        "torch_version": torch.__version__,
        "compile_plus_warm_seconds": compile_plus_warm_seconds,
        "eager_median_forward_vjp_seconds": statistics.median(eager_times),
        "compiled_median_forward_vjp_seconds": statistics.median(compiled_times),
        "eager_peak_allocated_bytes": eager_peak,
        "compiled_peak_allocated_bytes": compiled_peak,
        "max_output_difference": float((eager_y - compiled_y).abs().max()),
        "max_vjp_difference": float((eager_g - compiled_g).abs().max()),
        "eager_minimum_jacobian": minimum_jacobian(eager_y),
        "compiled_minimum_jacobian": minimum_jacobian(compiled_y),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
