"""Measured F1-D/F2-D forward and full-latent VJP on a declared Q1 grid."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    SafeColoredQ1Relaxation,
    StaggeredPatchQ1Layer,
    q1_corner_determinants,
    validate_q1_map,
)


def run_benchmark(
    *, side: int, mode: str, device: str, dtype: str = "float32",
    repeats: int = 3, batch: int = 1, seed: int = 290929,
    save_path: str | Path | None = None,
) -> dict[str, object]:
    if side < 9 or mode not in ("f1", "f2") or repeats < 1 or batch < 1:
        raise ValueError("side>=9, mode=f1/f2, repeats>=1 and batch>=1 required")
    precision = {"float32": torch.float32, "float64": torch.float64}[dtype]
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(seed)
    axis = torch.arange(side, dtype=precision, device=device_object) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)
    if mode == "f1":
        layer: torch.nn.Module = SafeColoredQ1Relaxation(side).to(device_object)
        fields = (0.15 * torch.randn(
            batch, side - 2, side - 2, 2, dtype=precision, device=device_object,
        )).requires_grad_()
        def forward() -> torch.Tensor:
            return layer(identity, fields)
        variables = (fields,)
        passes = 4  # Four sequential vertex colors.
    else:
        patch_cells = 4 if side == 9 else 8
        if (side - 1) % patch_cells:
            raise ValueError("F2 patch_cells must divide side-1")
        layer = StaggeredPatchQ1Layer(side, patch_cells).to(device_object)
        fields = tuple(
            (0.15 * torch.randn(
                batch, side - 2, side - 2, 2,
                dtype=precision, device=device_object,
            )).requires_grad_()
            for _ in range(4)
        )
        def forward() -> torch.Tensor:
            return layer(identity, fields)
        variables = fields
        passes = 4  # Four sequential staggered patch offsets.

    cotangent = torch.randn_like(identity)

    def synchronize() -> None:
        if device_object.type == "cuda":
            torch.cuda.synchronize(device_object)

    def trial() -> tuple[float, float, torch.Tensor, tuple[torch.Tensor, ...]]:
        synchronize()
        start = time.perf_counter()
        mapped = forward()
        synchronize()
        after_forward = time.perf_counter()
        scalar = (mapped * cotangent).sum() / mapped.numel()
        gradients = torch.autograd.grad(scalar, variables)
        synchronize()
        after_vjp = time.perf_counter()
        return after_forward - start, after_vjp - after_forward, mapped, gradients

    trial()  # Warm up allocations and CUDA kernels; not included in medians.
    synchronize()
    if device_object.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_object)
        resident = torch.cuda.memory_allocated(device_object)
    else:
        resident = None
    forward_times: list[float] = []
    vjp_times: list[float] = []
    mapped = identity
    gradients: tuple[torch.Tensor, ...] = ()
    for _ in range(repeats):
        forward_seconds, vjp_seconds, mapped, gradients = trial()
        forward_times.append(forward_seconds)
        vjp_times.append(vjp_seconds)

    corners = q1_corner_determinants(mapped)
    edges = (
        (mapped[:, 0] - identity[:, 0]).abs().amax(),
        (mapped[:, -1] - identity[:, -1]).abs().amax(),
        (mapped[:, :, 0] - identity[:, :, 0]).abs().amax(),
        (mapped[:, :, -1] - identity[:, :, -1]).abs().amax(),
    )
    result: dict[str, object] = {
        "side": side,
        "control_vertices": side * side,
        "cells": (side - 1) ** 2,
        "corner_constraints": 4 * (side - 1) ** 2,
        "batch": batch,
        "mode": mode,
        "passes": passes,
        "latent_scalars": sum(variable.numel() for variable in variables),
        "dtype": dtype,
        "device": str(device_object),
        "seed": seed,
        "repeats": repeats,
        "forward_seconds_median": statistics.median(forward_times),
        "vjp_seconds_median": statistics.median(vjp_times),
        "corner_min": float(corners.amin()),
        "nonpositive_corners": int((corners <= 0).sum()),
        "nonfinite_corners": int((~torch.isfinite(corners)).sum()),
        "boundary_max_error": float(torch.stack(edges).amax()),
        "grad_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
        "grad_max_abs": max(float(value.abs().amax()) for value in gradients),
        "cuda_resident_bytes": resident,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device_object)
            if device_object.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device_object)
            if device_object.type == "cuda" else None
        ),
    }
    if save_path is not None:
        output = Path(save_path)
        if output.suffix != ".npz":
            raise ValueError("saved map path must end in .npz")
        np.savez(
            output,
            vertices=mapped.detach().cpu().numpy(),
            boundary_reference=identity.detach().cpu().numpy(),
        )
        with np.load(output, allow_pickle=False) as arrays:
            reloaded = {
                "vertices": torch.from_numpy(arrays["vertices"].copy()),
                "boundary_reference": torch.from_numpy(arrays["boundary_reference"].copy()),
            }
        saved_report = validate_q1_map(
            reloaded["vertices"], reloaded["boundary_reference"],
        )
        result["saved_map_valid"] = saved_report["valid"]
        result["saved_map_corner_min"] = saved_report["corner_min"]
        result["saved_map_nonpositive_corners"] = saved_report["nonpositive_corners"]
        result["saved_map_nonfinite_corners"] = saved_report["nonfinite_corners"]
        result["saved_map_boundary_max_error"] = saved_report["boundary_max_error"]
        result["save_path"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--mode", choices=("f1", "f2"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--save-map", dest="save_path", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_benchmark(**vars(arguments)), sort_keys=True))


if __name__ == "__main__":
    main()
