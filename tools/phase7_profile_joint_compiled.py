"""Measure compiling only the F1 fine update inside the full certified layer."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import CoarsePatchFineVertexP1Layer
from phase7_multiscale_fiber_reachability import minimum_jacobian


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--latent-dtype", choices=["float32", "float64"],
                        default="float32")
    parser.add_argument("--amplitude", type=float, default=.1)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    latent_dtype = getattr(torch, args.latent_dtype)
    generator = torch.Generator(device=device).manual_seed(91737)
    zc = (args.amplitude * torch.randn(args.batch, 255, 255, 2, device=device,
                           dtype=latent_dtype, generator=generator)).requires_grad_()
    zf = (args.amplitude * torch.randn(args.batch, 1023, 1023, 2, device=device,
                           dtype=latent_dtype, generator=generator)).requires_grad_()
    cotangent = torch.randn(args.batch, 1025, 1025, 2, device=device,
                             dtype=torch.float64, generator=generator)
    eager = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True,
    ).to(device)
    compiled = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True,
    ).to(device)
    compiled.fine = torch.compile(compiled.fine, mode="default")

    def measure(model, warmup: int):
        torch.cuda.synchronize(device)
        startup = time.perf_counter()
        for _ in range(warmup):
            output = model(zc, zf)
            gc, gf = torch.autograd.grad((output * cotangent).mean(), (zc, zf))
        torch.cuda.synchronize(device)
        startup_seconds = time.perf_counter() - startup
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        times = []
        for _ in range(args.repeat):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            output = model(zc, zf)
            gc, gf = torch.autograd.grad((output * cotangent).mean(), (zc, zf))
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return output.detach(), gc.detach(), gf.detach(), {
            "warmup_seconds": startup_seconds,
            "median_forward_vjp_seconds": statistics.median(times),
            "peak_cuda_allocated_bytes": peak_bytes,
            "peak_above_baseline_bytes": peak_bytes - baseline_bytes,
            "minimum_jacobian": minimum_jacobian(output),
            "fallback_identity": bool(torch.all(output == model.fine_identity)),
        }

    eager_y, eager_gc, eager_gf, eager_stats = measure(eager, 2)
    compiled_y, compiled_gc, compiled_gf, compiled_stats = measure(compiled, 2)
    print(json.dumps({
        "experiment": "phase7_full_layer_compile_fine",
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "batch": args.batch,
        "amplitude": args.amplitude,
        "latent_dtype": args.latent_dtype,
        "compute_dtype": "float64",
        "repeat": args.repeat,
        "torch_version": torch.__version__,
        "eager": eager_stats,
        "compiled_fine": compiled_stats,
        "max_output_difference": float((eager_y - compiled_y).abs().max()),
        "max_coarse_vjp_difference": float((eager_gc - compiled_gc).abs().max()),
        "max_fine_vjp_difference": float((eager_gf - compiled_gf).abs().max()),
        "eager_vjp_finite": bool(torch.isfinite(eager_gc).all() and torch.isfinite(eager_gf).all()),
        "compiled_vjp_finite": bool(torch.isfinite(compiled_gc).all() and torch.isfinite(compiled_gf).all()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
