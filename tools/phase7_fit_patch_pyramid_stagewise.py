"""Direct target-supervised oracle fit of a compact patch P1 pyramid."""

from __future__ import annotations

import argparse
import json
import time

import torch

from phase7_teacher_reachability import target_map
from qcopt.neural_bijection.dense import StaggeredPatchP1Layer, exact_dyadic_p1_refine
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--seed-side", type=int, default=17)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--target-kind", choices=("base", "high32", "high64", "high128", "local_swirl"),
                        default="high32")
    parser.add_argument("--steps-per-level", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--scale-learning-rate", action="store_true")
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    side = args.seed_side
    axis = torch.arange(side, device=device) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    current = torch.stack((xx, yy), dim=-1)[None]
    history = []
    saved_latents = []
    start = time.perf_counter()
    while True:
        target = target_map(side, device, torch.float32, args.target_kind)
        layer = StaggeredPatchP1Layer(side, args.patch_cells).to(device)
        latents = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, patch_pass.interior_ids.numel(), 2,
                                           device=device))
            for patch_pass in layer.passes
        )
        effective_lr = (
            args.learning_rate * min(1.0, 256 / (side - 1))
            if args.scale_learning_rate else args.learning_rate
        )
        optimizer = torch.optim.Adam(latents, lr=effective_lr, eps=args.adam_eps)
        before = float((current - target).square().sum(dim=-1).mean().sqrt())
        initial_grad_max = None
        stage_start = time.perf_counter()
        for step in range(args.steps_per_level):
            optimizer.zero_grad(set_to_none=True)
            mapped = layer(current, tuple(latents))
            loss = (mapped - target).square().sum(dim=-1).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"nonfinite loss at side={side}, step={step + 1}")
            loss.backward()
            if any(x.grad is None or not torch.isfinite(x.grad).all().item()
                   for x in latents):
                raise RuntimeError(f"nonfinite VJP at side={side}, step={step + 1}")
            if step == 0:
                initial_grad_max = max(float(x.grad.abs().amax()) for x in latents)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.no_grad():
            current = layer(current, tuple(latents)).detach()
            after = float((current - target).square().sum(dim=-1).mean().sqrt())
            max_error = float((current - target).abs().amax())
            minimum = certify_convex_quad_output(current)
        history.append({
            "side": side,
            "control_vertices": side * side,
            "before_vertex_vector_rmse": before,
            "after_vertex_vector_rmse": after,
            "after_maximum_coordinate_error": max_error,
            "minimum_jacobian": minimum,
            "latent_scalars": sum(x.numel() for x in latents),
            "learning_rate": effective_lr,
            "adam_eps": args.adam_eps,
            "first_step_gradient_max": initial_grad_max,
            "stage_seconds": time.perf_counter() - stage_start,
        })
        saved_latents.append(tuple(x.detach().cpu() for x in latents))
        if side == args.side:
            break
        side = 2 * side - 1
        if side > args.side:
            raise ValueError("side must be dyadically reachable from seed side")
        current = exact_dyadic_p1_refine(current)
    if args.save_state:
        torch.save({"latents": saved_latents, "config": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "phase7_patch_pyramid_stagewise_direct_target_oracle",
        "target_kind": args.target_kind,
        "side": args.side,
        "seed_side": args.seed_side,
        "steps_per_level": args.steps_per_level,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "history": history,
        "total_seconds": time.perf_counter() - start,
        "checkpoint_path": args.save_state,
        "note": "Uses known target vertex coordinates at every stage; not image inference.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
