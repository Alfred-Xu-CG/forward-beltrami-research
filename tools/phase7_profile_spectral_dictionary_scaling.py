"""Profile streaming K-mode fine P1 synthesis without a K-by-grid basis cache."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import SineModeP1Refiner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=("colored", "patch"), default="colored")
    parser.add_argument("--dictionary-sizes", type=int, nargs="+", default=(1, 8, 32, 64))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--checkpoint-updates", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    axis = torch.arange(257, device=device, dtype=torch.float32) / 256
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack((xx, yy), dim=-1)[None].expand(args.batch, -1, -1, -1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for k in args.dictionary_sizes:
        if k < 1 or k > 128:
            raise ValueError("dictionary size must be in [1,128]")
        cycles = tuple((128, j + 1) for j in range(k))
        refiner = SineModeP1Refiner(
            257, 1025, cycles=cycles, mechanism=args.mechanism,
            checkpoint_updates=args.checkpoint_updates,
        ).to(device)
        amplitudes = torch.zeros((args.batch, k), device=device,
                                 dtype=torch.float32, requires_grad=True)
        generator = torch.Generator(device=device).manual_seed(34017)
        cotangent = torch.randn((args.batch, 1025, 1025, 2),
                                 generator=generator, device=device)
        with torch.no_grad():
            refiner(coarse, amplitudes)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        forward_times = []
        with torch.no_grad():
            for _ in range(3):
                sync()
                start = time.perf_counter()
                mapped = refiner(coarse, amplitudes)
                sync()
                forward_times.append(time.perf_counter() - start)
        forward_peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        warm = refiner(coarse, amplitudes)
        (warm * cotangent).mean().backward()
        amplitudes.grad = None
        del warm
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        joint_times = []
        for _ in range(3):
            amplitudes.grad = None
            sync()
            start = time.perf_counter()
            output = refiner(coarse, amplitudes)
            (output * cotangent).mean().backward()
            sync()
            joint_times.append(time.perf_counter() - start)
        vjp_peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        rows.append({
            "mode_count": k,
            "median_forward_seconds": statistics.median(forward_times),
            "median_forward_and_vjp_seconds": statistics.median(joint_times),
            "forward_peak_allocated_bytes": forward_peak,
            "forward_and_vjp_peak_allocated_bytes": vjp_peak,
            "amplitude_gradient_finite": bool(torch.isfinite(amplitudes.grad).all()),
            "amplitude_gradient_max_abs": float(amplitudes.grad.abs().amax()),
            "nonfinite_output_count": int((~torch.isfinite(mapped)).sum()),
        })
    print(json.dumps({
        "method": "phase7_streamed_spectral_p1_dictionary_scaling",
        "mechanism": args.mechanism,
        "checkpoint_updates": args.checkpoint_updates,
        "batch": args.batch,
        "control_side": 1025,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "dtype": "float32",
        "device": str(device),
        "torch_version": torch.__version__,
        "rows": rows,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
