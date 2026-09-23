"""Dense two-component F1 latent fit on strong shear plus 128-cycle detail."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    MultiscaleMonotoneFiberP1Layer,
    SafeColoredVertexRelaxation,
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
    parser.add_argument("--base-side", type=int, default=257)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--detail-direction", choices=("x", "xy"), default="x")
    parser.add_argument("--detail-amplitude", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    side = args.side
    base_target = teacher_map(side, "shear", device)
    detail_target = teacher_map(side, "high128", device)
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    target = base_target.clone()
    detail = args.detail_amplitude / 0.0001 * (
        detail_target[..., 0] - axis[None, None]
    )
    target[..., 0] += detail
    if args.detail_direction == "xy":
        target[..., 1] += detail
    if minimum_jacobian(target) <= 0:
        raise ValueError("the independent sampled target is not P1 oriented")
    base_fiber = MultiscaleMonotoneFiberP1Layer(side).to(device)
    exact_base = exact_teacher_latent(base_target, 0.05, 8.0)
    coarse_base = resize(exact_base, args.base_side - 2, args.base_side - 1)
    with torch.no_grad():
        base = base_fiber([coarse_base])
    layer = SafeColoredVertexRelaxation(
        side, safety_fraction=0.85, motion_mode="radial", raw_span=2.0,
    ).to(device)
    raw_scale = 2.0 / (side - 1)
    desired = (target[:, 1:-1, 1:-1] - base[:, 1:-1, 1:-1]) / raw_scale
    if float(desired.abs().amax()) >= 1:
        raise ValueError("target exceeds one-pass raw displacement span")
    teacher = torch.atanh(desired)
    with torch.no_grad():
        teacher_output = layer(base, teacher)
    latent = torch.nn.Parameter(torch.zeros_like(teacher))
    optimizer = torch.optim.Adam([latent], lr=args.lr, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.0001,
    )
    times, best_rmse, best_step = [], float("inf"), None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = layer(base, latent)
        loss = 1e6 * (output - target).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
        rmse = float((loss.detach() / 1e6).sqrt())
        if rmse < best_rmse:
            best_rmse, best_step = rmse, step
    with torch.no_grad():
        final = layer(base, latent)
    generator = torch.Generator(device=device).manual_seed(91017)
    cotangent = torch.randn(target.shape, generator=generator, device=device)
    random_vjp = torch.autograd.grad(
        (layer(base, latent) * cotangent).mean(), latent,
    )[0]
    print(json.dumps({
        "method": "phase7_dense_two_component_f1_detail_fit",
        "side": side,
        "base_side": args.base_side,
        "vertices": side ** 2,
        "faces": 2 * (side - 1) ** 2,
        "fine_latent_elements": latent.numel(),
        "base_latent_elements": coarse_base.numel(),
        "steps": args.steps,
        "learning_rate": args.lr,
        "detail_direction": args.detail_direction,
        "detail_amplitude": args.detail_amplitude,
        "target_minimum_jacobian": minimum_jacobian(target),
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "initial_vertex_vector_rmse": vector_rmse(base, target),
        "teacher_vertex_vector_rmse": vector_rmse(teacher_output, target),
        "teacher_minimum_jacobian": minimum_jacobian(teacher_output),
        "teacher_fine_logit_max_abs": float(teacher.abs().amax()),
        "best_training_step": best_step,
        "best_training_vertex_vector_rmse": best_rmse,
        "final_vertex_vector_rmse": vector_rmse(final, target),
        "final_minimum_jacobian": minimum_jacobian(final),
        "fine_gradient_finite": bool(torch.isfinite(latent.grad).all()),
        "random_cotangent_vjp_finite": bool(torch.isfinite(random_vjp).all()),
        "random_cotangent_vjp_max_abs": float(random_vjp.abs().amax()),
        "median_training_step_seconds": statistics.median(
            times[10:] if len(times) > 10 else times,
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
