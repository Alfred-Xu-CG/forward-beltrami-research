"""Forward/VJP benchmark of actual mixed-scale coarse-patch/fine-vertex P1 latent."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, exact_dyadic_p1_refine,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_matched_low_patch import (
    low_map_from_params, matched_low_patch,
    multistart_refine_low_params,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_mixed_scale_image_pipeline import (
    mixed_dataset, photo_high,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=94721)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or args.directions < 1:
        raise ValueError("repeats must be at least 2 and directions positive")
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    data = mixed_dataset(
        args.batch, args.seed, device, args.batch,
        table, (80.5, 96.5, 112.5, 127.5))
    fixed, moving, target = data[:3]
    true_k = data[-1]
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16,
        coarse_cycles=2, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    low_times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with torch.no_grad():
            origin, vector, _ = matched_low_patch(
                fixed, moving)
            origin, vector, _ = multistart_refine_low_params(
                fixed, moving, origin, vector,
                steps=4, directions=args.directions)
        torch.cuda.synchronize(device)
        low_times.append(time.perf_counter() - tick)
    low_proposal = low_map_from_params(
        origin, vector, 257, torch.float64)
    coarse_latent = torch.atanh((
        (low_proposal - decoder.identity)[:, 1:-1, 1:-1] /
        (8 / 256)).clamp(-.95, .95)).float()
    with torch.no_grad():
        coarse_map = decoder.coarse(
            decoder.identity.expand(args.batch, -1, -1, -1),
            coarse_latent.double())
        fine_base = exact_dyadic_p1_refine(
            exact_dyadic_p1_refine(coarse_map))
    offsets = torch.arange(
        -.5, .525, .05,
        device=device, dtype=torch.float32)
    high_times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with torch.no_grad():
            high, estimated_k = photo_high(
                fixed, moving, fine_base, table, offsets)
        torch.cuda.synchronize(device)
        high_times.append(time.perf_counter() - tick)
    fine_latent = torch.atanh((
        high / (2 / 1024)).clamp(-.95, .95)
        [:, :, 1:-1, 1:-1]).permute(
            0, 2, 3, 1).contiguous().float()
    forward_times = []
    vjp_times = []
    for _ in range(args.repeats):
        coarse_latent = coarse_latent.detach().requires_grad_()
        fine_latent = fine_latent.detach().requires_grad_()
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        output = decoder(coarse_latent, fine_latent)
        torch.cuda.synchronize(device)
        forward_times.append(time.perf_counter() - tick)
        loss = 1e8 * (
            output - target).square().sum(dim=-1).mean()
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        grad_coarse, grad_fine = torch.autograd.grad(
            loss, (coarse_latent, fine_latent))
        torch.cuda.synchronize(device)
        vjp_times.append(time.perf_counter() - tick)
    if not bool(torch.isfinite(grad_coarse).all()):
        raise AssertionError("coarse gradient was non-finite")
    if not bool(torch.isfinite(grad_fine).all()):
        raise AssertionError("fine gradient was non-finite")
    print(json.dumps(dict(
        experiment="phase7_mixed_scale_joint_latent_vjp",
        control_vertices=1025 ** 2,
        coarse_vertices=257 ** 2,
        control_faces=2 * 1024 ** 2,
        batch=args.batch,
        directions=args.directions,
        repeats=args.repeats,
        seed=args.seed,
        map_vector_rmse=math.sqrt(float(
            (output - target).square().sum(dim=-1).mean())),
        minimum_jacobian=minimum_jacobian(output),
        frequency_mae=float((estimated_k - true_k).abs().mean()),
        coarse_gradient_nonzero=int((grad_coarse != 0).sum()),
        fine_gradient_nonzero=int((grad_fine != 0).sum()),
        coarse_gradient_finite=bool(
            torch.isfinite(grad_coarse).all()),
        fine_gradient_finite=bool(
            torch.isfinite(grad_fine).all()),
        low_inference_median_seconds=statistics.median(low_times[1:]),
        high_inference_median_seconds=statistics.median(high_times[1:]),
        decoder_forward_median_seconds=statistics.median(forward_times[1:]),
        decoder_vjp_median_seconds=statistics.median(vjp_times[1:]),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
