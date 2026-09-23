"""Fit independent high-frequency detail on a strong shear P1 base.

Compares a fine log-edge-density residual with the smooth potential refiner.
Both mechanisms retain the same original-grid P1 topology for all latents.
The base is an oracle shear latent; this isolates fine-decoder conditioning and
is explicitly not an image-to-latent experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    HybridMonotoneFiberP1Layer,
    MultiscaleMonotoneFiberP1Layer,
)
from phase7_multiscale_fiber_reachability import (
    exact_teacher_latent,
    minimum_jacobian,
    resize,
    teacher_map,
    vector_rmse,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--base-side", type=int, default=1025)
    parser.add_argument("--method", choices=("density", "potential"), required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--potential-span", type=float, default=0.005)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    side = args.side
    if args.base_side > side or args.base_side < 3:
        raise ValueError("base-side must be between 3 and side")
    base_target = teacher_map(side, "shear", device)
    detail_target = teacher_map(side, "high128", device)
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    target = base_target.clone()
    target[..., 0] += detail_target[..., 0] - axis[None, None]
    exact_base_latent = exact_teacher_latent(base_target, 0.05, 8.0).detach()
    base_latent = resize(exact_base_latent, args.base_side - 2,
                         args.base_side - 1)
    base_dense = resize(base_latent, side - 2, side - 1)
    if args.method == "density":
        layer = MultiscaleMonotoneFiberP1Layer(side).to(device)
        shape = (1, side - 2, side - 1)
        target_latent = exact_teacher_latent(target, 0.05, 8.0)
        teacher_fine = target_latent - base_dense
        def decode(fine: torch.Tensor) -> torch.Tensor:
            return layer([base_latent, fine])
    else:
        layer = HybridMonotoneFiberP1Layer(
            side, potential_span=args.potential_span,
        ).to(device)
        shape = (1, side - 2, side - 2)
        with torch.no_grad():
            decoded_base = layer.base([base_latent])
        residual = target[:, 1:-1, 1:-1, 0] - decoded_base[:, 1:-1, 1:-1, 0]
        if float(residual.abs().amax()) >= layer.fine.potential_span:
            raise ValueError("oracle residual exceeds potential_span")
        teacher_fine = torch.atanh(residual / layer.fine.potential_span)
        def decode(fine: torch.Tensor) -> torch.Tensor:
            return layer([base_latent], fine)
    field = torch.nn.Parameter(torch.zeros(shape, device=device))
    optimizer = torch.optim.Adam([field], lr=args.lr, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.0001,
    )
    timings = []
    best_rmse, best_step = float("inf"), None
    with torch.no_grad():
        initial = decode(field)
        teacher = decode(teacher_fine)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = decode(field)
        loss = 1e6 * (output - target).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - tick)
        rmse = float((loss.detach() / 1e6).sqrt())
        if rmse < best_rmse:
            best_rmse, best_step = rmse, step
    with torch.no_grad():
        final = decode(field)
    print(json.dumps({
        "method": "phase7_fit_high128_detail_on_strong_shear_base",
        "fine_parameterization": args.method,
        "side": side,
        "vertices": side ** 2,
        "faces": 2 * (side - 1) ** 2,
        "fine_latent_elements": field.numel(),
        "base_latent_elements": base_latent.numel(),
        "base_side": args.base_side,
        "steps": args.steps,
        "learning_rate": args.lr,
        "potential_span": args.potential_span if args.method == "potential" else None,
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "initial_vertex_vector_rmse": vector_rmse(initial, target),
        "teacher_vertex_vector_rmse": vector_rmse(teacher, target),
        "teacher_minimum_jacobian": minimum_jacobian(teacher),
        "teacher_fine_logit_max_abs": float(teacher_fine.abs().amax()),
        "best_training_step": best_step,
        "best_training_vertex_vector_rmse": best_rmse,
        "final_vertex_vector_rmse": vector_rmse(final, target),
        "final_minimum_jacobian": minimum_jacobian(final),
        "final_fine_logit_max_abs": float(field.detach().abs().amax()),
        "fine_gradient_finite": bool(torch.isfinite(field.grad).all()),
        "median_training_step_seconds": statistics.median(
            timings[10:] if len(timings) > 10 else timings
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
