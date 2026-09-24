"""Oracle reachability and VJP of a million-vertex local patch-field P1 map."""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import certify_p1_or_identity
from qcopt.neural_bijection.dense.patch_field import ResidualStaggeredPatchP1Layer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_unknown_carrier_bank import dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=99023)
    parser.add_argument("--patch-cells", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    side = 1025
    image_table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(side - 1, side - 1), height=512, width=512)
    image_table.prepare(device=device, dtype=torch.float64)
    _, _, targets, _, _, _ = dataset(
        args.count, args.seed, device, args.batch, image_table,
        tuple(range(80, 129)))
    axis = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    layer = ResidualStaggeredPatchP1Layer(
        side, patch_cells=args.patch_cells, cycles=args.cycles,
        minimum_jacobian=.05, raw_span=.5).to(device)
    span = .5 * args.patch_cells / (side - 1)
    square_sum = 0.
    count = 0
    minimum = math.inf
    passed = 0
    forward_times = []
    vjp_times = []
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, args.count, args.batch):
        target = targets[start:start + args.batch]
        batch = len(target)
        proposal = (target - identity)[:, 1:-1, 1:-1] / span
        latent = torch.atanh(proposal.clamp(-.95, .95)).detach().requires_grad_()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        candidate = layer(identity.expand(batch, -1, -1, -1), latent)
        output, valid = certify_p1_or_identity(candidate, identity)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_times.append(time.perf_counter() - tick)
        passed += int(valid.sum())
        minimum = min(minimum, minimum_jacobian(output))
        square_sum += float((output - target).square().sum())
        count += batch * side * side
        probe = (output[:, 1:-1, 1:-1] * target[:, 1:-1, 1:-1]).mean()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        gradient, = torch.autograd.grad(probe, latent)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        vjp_times.append(time.perf_counter() - tick)
        if not bool(torch.isfinite(gradient).all()) or not bool((gradient != 0).any()):
            raise AssertionError("patch-field oracle VJP was zero or non-finite")
    print(json.dumps({
        "experiment": "phase7_patch_field_unknown_carrier_oracle",
        "control_vertices": side * side,
        "control_faces": 2 * (side - 1) ** 2,
        "count": args.count,
        "batch": args.batch,
        "seed": args.seed,
        "patch_cells": args.patch_cells,
        "cycles": args.cycles,
        "map_vector_rmse": math.sqrt(square_sum / count),
        "minimum_jacobian": minimum,
        "passed_certificate": passed,
        "mean_forward_batch_seconds": sum(forward_times) / len(forward_times),
        "mean_vjp_batch_seconds": sum(vjp_times) / len(vjp_times),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
