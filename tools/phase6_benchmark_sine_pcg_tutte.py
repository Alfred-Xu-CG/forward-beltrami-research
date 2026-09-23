"""Measure bounded all-edge sine-PCG equilibrium and its implicit VJP."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time

import torch

from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def make_logits(side: int, batch: int, mode: str, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(60531)
    shapes = ((side, side - 1), (side - 1, side), (side - 1, side - 1))
    if mode == "random":
        return tuple((0.7 * torch.randn(batch, *shape, dtype=torch.float64, device=device)).requires_grad_()
                     for shape in shapes)
    if mode != "smooth":
        raise ValueError("mode must be random or smooth")
    row = torch.arange(side, dtype=torch.float64, device=device) / (side - 1)
    column = row
    logit_fields = []
    for kind, shape in enumerate(shapes):
        yy, xx = torch.meshgrid(
            (row if kind == 0 else 0.5 * (row[:-1] + row[1:])),
            (column if kind == 1 else 0.5 * (column[:-1] + column[1:])), indexing="ij",
        )
        field = 2.5 * torch.sin(2 * math.pi * xx + kind) * torch.cos(2 * math.pi * yy - 0.4 * kind)
        if field.shape != shape:
            raise AssertionError("conductance field shape does not match edge family")
        logit_fields.append(field.expand(batch, -1, -1).contiguous().requires_grad_())
    return tuple(logit_fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mode", choices=("random", "smooth"), default="smooth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = SinePreconditionedTutteLayer(
        args.side, tolerance=args.tolerance, max_iterations=args.max_iterations,
    ).to(device)
    logits = make_logits(args.side, args.batch, args.mode, device)
    torch.manual_seed(55317)
    cotangent = torch.randn(args.batch, args.side, args.side, 2, device=device, dtype=torch.float64)
    cotangent[:, 0] = 0; cotangent[:, -1] = 0; cotangent[:, :, 0] = 0; cotangent[:, :, -1] = 0
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    backward_times = []
    samples = []
    peak_rss = _rss_bytes()
    for repeat in range(args.repeats + 1):
        for value in logits:
            value.grad = None
        synchronize()
        began = time.perf_counter()
        mapped = layer(*logits)
        synchronize()
        middle = time.perf_counter()
        (mapped * cotangent).sum().backward()
        synchronize()
        finished = time.perf_counter()
        if repeat:
            forward_times.append(middle - began)
            backward_times.append(finished - middle)
            samples.append({
                "forward_seconds": middle - began,
                "backward_seconds": finished - middle,
                "forward_stats": dict(layer.last_forward_stats),
                "backward_stats": dict(layer.last_backward_stats),
                "maximum_abs_logit_gradient": max(float(value.grad.abs().max()) for value in logits),
            })
            resident = _rss_bytes()
            if resident is not None:
                peak_rss = max(peak_rss or 0, resident)
    print(json.dumps({
        "method": "all_edge_bounded_sine_pcg_tutte",
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "edge_logit_count_including_inert_boundary_edges": sum(value.numel() for value in logits) // args.batch,
        "batch": args.batch,
        "mode": args.mode,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "dtype": "float64",
        "torch_version": torch.__version__,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "median_forward_seconds": statistics.median(forward_times),
        "median_backward_seconds": statistics.median(backward_times),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "sampled_peak_process_rss_bytes": peak_rss,
        "samples": samples,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
