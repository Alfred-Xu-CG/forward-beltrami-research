"""Nonperiodic scattered Beurling direct quadrature on CPU/GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_torch import direct_beurling_apply_torch


def _cloud(n: int, *, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(20260919 + n)
    source = 0.05 + 0.42 * torch.rand(n, device=device, dtype=torch.float32)
    source = source.to(dtype=torch.complex64 if dtype == torch.complex64 else torch.complex128)
    source = source + 1j * (0.05 + 0.9 * torch.rand(n, device=device, dtype=torch.float32)).to(dtype)
    target = 0.55 + 0.4 * torch.rand(n, device=device, dtype=torch.float32)
    target = target.to(dtype=dtype) + 1j * (0.05 + 0.9 * torch.rand(n, device=device, dtype=torch.float32)).to(dtype)
    values = (torch.randn(n, device=device, dtype=torch.float32) + 1j * torch.randn(n, device=device, dtype=torch.float32)).to(dtype)
    weights = (0.4 + 0.8 * torch.rand(n, device=device, dtype=torch.float32)) / n
    return source, target, values, weights


def run(output_dir: Path, sizes: tuple[int, ...] = (16384, 32768), device: str | None = None, block_size: int = 1024) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    small_source, small_target, small_values, small_weights = _cloud(192, dtype=torch.complex128, device=torch.device("cpu"))
    reference = direct_beurling_apply(
        small_source.numpy(), small_values.numpy(), small_weights.numpy(), target_points=small_target.numpy(), block_size=64
    )
    small_output = direct_beurling_apply_torch(
        small_source, small_values, small_weights, target_points=small_target, block_size=64, dtype=torch.complex128
    )
    parity = float(torch.max(torch.abs(small_output - torch.from_numpy(reference))))
    records = []
    for n in sizes:
        source, target, values, weights = _cloud(n, dtype=torch.complex64, device=selected)
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        started = time.perf_counter()
        output = direct_beurling_apply_torch(
            source, values, weights, target_points=target, block_size=block_size, dtype=torch.complex64
        )
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        elapsed = time.perf_counter() - started
        records.append(
            {
                "points": n,
                "interactions": n * n,
                "seconds": elapsed,
                "interactions_per_second": float(n * n / elapsed),
                "output_l2": float(torch.sqrt(torch.sum(torch.abs(output) ** 2))),
                "finite": bool(torch.all(torch.isfinite(output))),
            }
        )
    result = {
        "device": str(selected),
        "torch_version": torch.__version__,
        "block_size": block_size,
        "small_grid_numpy_parity_max": parity,
        "records": records,
        "scope": "blocked nonperiodic whole-plane Beurling direct quadrature on arbitrary scattered source/target points",
        "limitation": "O(N^2) interactions; this is a differentiable GPU baseline/control, not an FMM or NUFFT",
    }
    (output_dir / "beurling_torch_gpu_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes), args.device, args.block_size), indent=2))


if __name__ == "__main__":
    main()
