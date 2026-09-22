"""Time dense monotone-map forward/VJP on CPU or an actually idle GPU.

Example: python tools/phase6_benchmark_monotone.py --side 257 --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import torch

from qcopt.neural_bijection.dense import DenseMonotoneGridLayer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _target(side: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    shape = torch.sin(2.0 * math.pi * xx) * torch.sin(2.0 * math.pi * yy)
    return torch.stack((xx + 0.03 * shape, yy + 0.05 * shape), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--axis", choices=("vertical", "horizontal"), default="vertical")
    args = parser.parse_args()
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.side < 2 or args.batch < 1 or args.repeat < 1:
        raise ValueError("side, batch, and repeat must be positive")
    layer = DenseMonotoneGridLayer(args.side, axis=args.axis)
    first = (0.2 * torch.randn(args.batch, args.side - 1, device=device, dtype=dtype)).requires_grad_()
    second = (0.2 * torch.randn(args.batch, args.side, args.side - 1, device=device, dtype=dtype)).requires_grad_()
    target = _target(args.side, dtype=dtype, device=device).unsqueeze(0)
    for _ in range(2):
        result = layer(first, second)
        ((result - target).square().mean()).backward()
        first.grad = None
        second.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    backward_times = []
    for _ in range(args.repeat):
        begin = time.perf_counter()
        result = layer(first, second)
        loss = (result - target).square().mean()
        _sync(device)
        middle = time.perf_counter()
        loss.backward()
        _sync(device)
        end = time.perf_counter()
        forward_times.append(middle - begin)
        backward_times.append(end - middle)
        first.grad = None
        second.grad = None

    one = result[0]
    a = one[:-1, :-1]
    b = one[:-1, 1:]
    c = one[1:, 1:]
    d = one[1:, :-1]
    area_lower = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])
    area_upper = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (
        c[..., 1] - a[..., 1]
    ) * (d[..., 0] - a[..., 0])
    minimum_ratio = min(area_lower.min().item(), area_upper.min().item()) * (args.side - 1) ** 2
    payload = {
        "route": "A",
        "side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "batch": args.batch,
        "axis": args.axis,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "forward_seconds": forward_times,
        "backward_seconds": backward_times,
        "minimum_signed_area_ratio": minimum_ratio,
        "target_map_rmse": (result - target).square().mean().sqrt().item(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "loss": loss.item(),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
