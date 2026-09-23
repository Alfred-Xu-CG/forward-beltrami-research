"""Fit solve-free positive-fiber latents to independently specified 1025-P1 targets."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    MultiscaleMonotoneFiberP1Layer,
    SoftplusPotentialFiberP1Layer,
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
    parser.add_argument("--latent-sides", nargs="+", type=int,
                        default=[33, 65, 129, 257, 513, 1025])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"),
                        default="cosine")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kind", choices=("shear", "high128"), default="shear")
    parser.add_argument("--parameterization", choices=("density", "potential"),
                        default="density")
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = (MultiscaleMonotoneFiberP1Layer(args.side).to(device)
             if args.parameterization == "density" else
             SoftplusPotentialFiberP1Layer(args.side).to(device))
    target = teacher_map(args.side, args.kind, device)
    if args.parameterization == "density":
        exact_logit = exact_teacher_latent(target, layer.fiber.floor_fraction,
                                           layer.fiber.logit_span)
        dense_shape = (args.side - 2, args.side - 1)
    else:
        unit = torch.arange(args.side, device=device,
                            dtype=target.dtype) / (args.side - 1)
        displacement = target[:, 1:-1, 1:-1, 0] - unit[None, None, 1:-1]
        exact_logit = torch.atanh(displacement / layer.potential_span)
        dense_shape = (args.side - 2, args.side - 2)

    def decode(field: torch.Tensor) -> torch.Tensor:
        if args.parameterization == "density":
            return layer([field])
        return layer(resize(field, *dense_shape))
    results = []
    for latent_side in args.latent_sides:
        if latent_side > args.side or latent_side < 3:
            raise ValueError("latent side must be between 3 and final side")
        latent_shape = ((latent_side - 2, latent_side - 1)
                        if args.parameterization == "density" else
                        (latent_side - 2, latent_side - 2))
        field = torch.nn.Parameter(torch.zeros(
            (1, *latent_shape), device=device,
            dtype=torch.float32,
        ))
        optimizer = torch.optim.Adam([field], lr=args.lr, eps=1e-12)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr * 0.0001,
        ) if args.lr_schedule == "cosine" else None)
        timings = []
        first_rmse = None
        best_rmse = float("inf")
        best_step = None
        for step in range(args.steps):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = decode(field)
            loss = (output - target).square().sum(dim=-1).mean() * 1e6
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - tick)
            if step == 0:
                first_rmse = vector_rmse(output.detach(), target)
            current_rmse = float((loss.detach() / 1e6).sqrt())
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_step = step
        with torch.no_grad():
            output = decode(field)
            # Bilinear projection of a teacher logit is a representability
            # reference, not necessarily the best fit at a coarse scale.
            teacher_coarse = resize(exact_logit, *latent_shape)
            oracle_output = decode(teacher_coarse)
            results.append({
                "latent_side": latent_side,
                "latent_elements": field.numel(),
                "first_step_vertex_vector_rmse": first_rmse,
                "final_vertex_vector_rmse": vector_rmse(output, target),
                "best_training_vertex_vector_rmse": best_rmse,
                "best_training_step": best_step,
                "teacher_logit_max_abs": float(exact_logit.abs().amax()),
                "final_logit_max_abs": float(field.abs().amax()),
                "teacher_projection_vertex_vector_rmse": vector_rmse(
                    oracle_output, target,
                ),
                "minimum_jacobian": minimum_jacobian(output),
                "gradient_finite": bool(torch.isfinite(field.grad).all()),
                "median_training_step_seconds": statistics.median(timings[10:]),
            })
    print(json.dumps({
        "method": "phase7_direct_latent_fit_positive_fiber_p1",
        "target": args.kind,
        "side": args.side,
        "vertices": args.side ** 2,
        "faces": 2 * (args.side - 1) ** 2,
        "steps_per_latent_side": args.steps,
        "learning_rate": args.lr,
        "learning_rate_schedule": args.lr_schedule,
        "parameterization": args.parameterization,
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
