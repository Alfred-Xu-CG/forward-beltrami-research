"""Measure a genuine joint coarse-patch/fine-vertex P1 decoder at large grids."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import CoarsePatchFineVertexP1Layer
from phase7_multiscale_fiber_reachability import minimum_jacobian


def coordinate_latents(
    coarse_side: int, fine_side: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nonzero two-coordinate test fields, not inverse-fitted teacher fields."""
    coarse_axis = torch.arange(
        1, coarse_side - 1, device=device, dtype=torch.float32
    ) / (coarse_side - 1)
    cy, cx = torch.meshgrid(coarse_axis, coarse_axis, indexing="ij")
    coarse = torch.stack((
        .12 * torch.sin(2 * math.pi * cx) * torch.sin(math.pi * cy),
        .09 * torch.sin(math.pi * cx) * torch.sin(2 * math.pi * cy),
    ), dim=-1)[None].contiguous()
    fine_axis = torch.arange(
        1, fine_side - 1, device=device, dtype=torch.float32
    ) / (fine_side - 1)
    fy, fx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    fine = torch.stack((
        .09 * torch.sin(22 * math.pi * fx) * torch.sin(18 * math.pi * fy),
        .07 * torch.sin(17 * math.pi * fx) * torch.sin(21 * math.pi * fy),
    ), dim=-1)[None].contiguous()
    return coarse, fine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-side", type=int, default=513)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-fine", action="store_true")
    args = parser.parse_args()
    coarse_side = args.coarse_side
    fine_side = 4 * (coarse_side - 1) + 1
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("benchmark needs CUDA timings and memory metrics")
    layer = CoarsePatchFineVertexP1Layer(
        coarse_side, fine_side,
        coarse_patch_cells=args.patch_cells,
        coarse_cycles=2,
        minimum_jacobian=.05,
        compute_dtype=torch.float64,
        certify_output=True,
        checkpoint_fine=args.checkpoint_fine,
    ).to(device)
    coarse, fine = coordinate_latents(coarse_side, fine_side, device)
    coarse.requires_grad_()
    fine.requires_grad_()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    forward_times, vjp_times = [], []
    output = None
    grad_coarse = grad_fine = None
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        output = layer(coarse, fine)
        torch.cuda.synchronize(device)
        forward_times.append(time.perf_counter() - tick)
        loss = 1e8 * (
            output - layer.fine_identity
        ).square().sum(dim=-1).mean()
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        grad_coarse, grad_fine = torch.autograd.grad(
            loss, (coarse, fine))
        torch.cuda.synchronize(device)
        vjp_times.append(time.perf_counter() - tick)
    assert output is not None
    assert grad_coarse is not None and grad_fine is not None
    peak_forward_vjp = torch.cuda.max_memory_allocated(device)
    with torch.no_grad():
        min_j = minimum_jacobian(output)
        displacement = float((output - layer.fine_identity).abs().amax())
        boundary = max(
            float((output[:, 0] - layer.fine_identity[:, 0]).abs().amax()),
            float((output[:, -1] - layer.fine_identity[:, -1]).abs().amax()),
            float((output[:, :, 0] - layer.fine_identity[:, :, 0]).abs().amax()),
            float((output[:, :, -1] - layer.fine_identity[:, :, -1]).abs().amax()),
        )
        coarse_max = float(grad_coarse.abs().amax())
        fine_max = float(grad_fine.abs().amax())
        coarse_finite = bool(torch.isfinite(grad_coarse).all())
        fine_finite = bool(torch.isfinite(grad_fine).all())
    if min_j <= 0 or displacement <= 0 or boundary != 0:
        raise AssertionError("invalid or trivial final P1 map")
    if not (coarse_max > 0 and fine_max > 0 and coarse_finite and fine_finite):
        raise AssertionError("joint latent VJP failed")
    print(json.dumps(dict(
        experiment="phase7_large_joint_multiscale_decoder",
        coarse_side=coarse_side,
        control_side=fine_side,
        control_vertices=fine_side ** 2,
        control_faces=2 * (fine_side - 1) ** 2,
        batch=1,
        dtype="float64_geometry_float32_latent",
        device=torch.cuda.get_device_name(device),
        patch_cells=args.patch_cells,
        repeats=args.repeats,
        checkpoint_fine=args.checkpoint_fine,
        minimum_jacobian=min_j,
        maximum_displacement=displacement,
        maximum_boundary_error=boundary,
        coarse_gradient_max=coarse_max,
        fine_gradient_max=fine_max,
        coarse_gradient_finite=coarse_finite,
        fine_gradient_finite=fine_finite,
        forward_seconds=statistics.median(forward_times),
        vjp_seconds=statistics.median(vjp_times),
        peak_forward_vjp_cuda_allocated_bytes=peak_forward_vjp,
        peak_including_topology_cuda_allocated_bytes=(
            torch.cuda.max_memory_allocated(device)),
        torch_version=torch.__version__,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
