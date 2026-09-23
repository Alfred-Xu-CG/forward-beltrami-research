"""Direct-latent fitting of an independent target for the patch-field layer."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from phase7_teacher_reachability import target_map
from qcopt.neural_bijection.dense import StaggeredPatchP1Layer
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--target-kind", choices=("base", "high32", "high64", "local_swirl"), default="local_swirl")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--diagnose-nonfinite", action="store_true")
    args = parser.parse_args()
    if args.cycles < 1 or args.steps < 1:
        raise ValueError("cycles and steps must be positive")
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    side = args.side
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    target = target_map(side, device, torch.float32, args.target_kind)
    target_minimum = certify_convex_quad_output(target)
    layer = StaggeredPatchP1Layer(side, args.patch_cells).to(device)
    latents = torch.nn.ParameterList(
        torch.nn.Parameter(torch.zeros(1, side - 2, side - 2, 2, device=device))
        for _ in range(4 * args.cycles)
    )
    optimizer = torch.optim.Adam(latents, lr=args.learning_rate)

    def error(mapped: torch.Tensor) -> tuple[torch.Tensor, float]:
        square = (mapped - target).square().sum(dim=-1)
        return square.mean(), float(square.mean().sqrt().detach())

    def decode() -> torch.Tensor:
        current = base
        for cycle in range(args.cycles):
            current = layer(current, tuple(latents[4 * cycle:4 * (cycle + 1)]))
        return current

    initial = error(base)[1]
    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if args.diagnose_nonfinite:
            with torch.autograd.detect_anomaly():
                mapped = decode()
                objective, rmse = error(mapped)
                if not torch.isfinite(objective).item():
                    raise RuntimeError(f"nonfinite oracle loss at step {step}")
                objective.backward()
        else:
            mapped = decode()
            objective, rmse = error(mapped)
            if not torch.isfinite(objective).item():
                raise RuntimeError(f"nonfinite oracle loss at step {step}")
            objective.backward()
        if args.diagnose_nonfinite and not torch.isfinite(mapped).all().item():
            raise RuntimeError(f"nonfinite mapped coordinates before backward at step {step}")
        if args.diagnose_nonfinite:
            bad = [i for i, latent in enumerate(latents)
                   if latent.grad is None or not torch.isfinite(latent.grad).all().item()]
            if bad:
                raise RuntimeError(f"nonfinite latent gradients at step {step}: fields={bad}")
        optimizer.step()
        if args.diagnose_nonfinite:
            bad = [i for i, latent in enumerate(latents) if not torch.isfinite(latent).all().item()]
            if bad:
                raise RuntimeError(f"nonfinite latent parameters after step {step}: fields={bad}")
        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            history.append({"step": step, "vertex_vector_rmse": rmse})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    with torch.no_grad():
        final = decode()
        final_rmse = error(final)[1]
        final_max_error = float((final - target).abs().amax())
        final_minimum = certify_convex_quad_output(final)
        displacement_rms = float((final - base).square().sum(dim=-1).mean().sqrt())
    if args.save_state:
        torch.save({"latents": [x.detach().cpu() for x in latents], "config": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "phase7_patch_field_direct_latent_oracle",
        "target_kind": args.target_kind,
        "side": side,
        "control_vertices": side**2,
        "control_faces": 2 * (side - 1)**2,
        "patch_cells": args.patch_cells,
        "cycles": args.cycles,
        "latent_scalars": sum(x.numel() for x in latents),
        "device": str(device),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_identity_vertex_vector_rmse": initial,
        "final_vertex_vector_rmse": final_rmse,
        "final_max_coordinate_error": final_max_error,
        "final_displacement_rms": displacement_rms,
        "target_minimum_jacobian": target_minimum,
        "final_minimum_jacobian": final_minimum,
        "train_seconds": train_seconds,
        "history": history,
        "checkpoint_path": args.save_state,
        "note": "Known target coordinates supervise latent optimization; not image inference.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
