"""Geometry-only same-level F2->F1 recurrence on an irregular Q1/P1 mesh.

This tests safe mixed composition and its first-order VJP, not image registration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, StaggeredPatchQ1Layer,
    q1_corner_determinants,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def run(side: int, rounds: int, device: str, prefix: Path,
        repeats: int = 3, checkpoint_rounds: bool = True,
        minimum_jacobian: float | None = .05,
        patch_cells: int = 8, f2_raw_span: float = .5) -> dict:
    if (side < 17 or patch_cells < 2 or patch_cells % 2
            or (side - 1) % patch_cells or rounds < 1 or repeats < 1):
        raise ValueError("side must be compatible with an even patch width")
    target_device = torch.device(device)
    axis = torch.arange(side, device=target_device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), -1)[None]
    wave = torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy)
    base = torch.stack((xx + .05 * wave, yy), -1)[None]
    target = torch.stack((xx + .08 * wave, yy), -1)[None]
    torch.manual_seed(290930)
    f2_latents = (.5 * torch.randn(rounds, 4, 1, side - 2, side - 2, 2,
                                   device=target_device))
    f1_latents = (.5 * torch.randn(rounds, 1, side - 2, side - 2, 2,
                                   device=target_device))
    f2 = StaggeredPatchQ1Layer(side, patch_cells=patch_cells,
                                proposal_mode="current_edge",
                                raw_span=f2_raw_span,
                                minimum_jacobian=minimum_jacobian).to(target_device)
    f1 = AdaptiveSoftRadialQ1Relaxation(
        side, raw_span=8, minimum_jacobian=minimum_jacobian,
    ).to(target_device)

    def macro(state: torch.Tensor, z2: torch.Tensor, z1: torch.Tensor
              ) -> torch.Tensor:
        updated = f2(state, tuple(z2[:len(f2.passes)].unbind(0)))
        return f1(updated, z1)

    def all_rounds(state: torch.Tensor, z2: torch.Tensor, z1: torch.Tensor
                   ) -> torch.Tensor:
        for k in range(rounds):
            if checkpoint_rounds:
                state = checkpoint(macro, state, z2[k], z1[k], use_reentrant=False)
            else:
                state = macro(state, z2[k], z1[k])
        return state

    def measured() -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = base.detach().clone().requires_grad_()
        z2 = f2_latents.detach().clone().requires_grad_()
        z1 = f1_latents.detach().clone().requires_grad_()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        start = time.perf_counter()
        result = all_rounds(state, z2, z1)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        forward = time.perf_counter()
        loss = (result - target).square().sum(-1).mean()
        grads = torch.autograd.grad(loss, (state, z2, z1))
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        end = time.perf_counter()
        if not all(bool(torch.isfinite(grad).all()) for grad in grads):
            raise FloatingPointError("nonfinite mixed F2/F1 VJP")
        return forward - start, end - start, result.detach(), grads[1].detach(), grads[2].detach()

    measured()  # warmup
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    times = [measured() for _ in range(repeats)]
    final, grad_f2, grad_f1 = times[-1][2:]
    timed_peak_bytes = (torch.cuda.max_memory_allocated(target_device)
                        if target_device.type == "cuda" else None)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    gradient_archive = prefix.with_name(prefix.name + "_latents_and_vjp.npz")
    np.savez_compressed(
        gradient_archive,
        f2_latents=f2_latents.detach().cpu().numpy(),
        f1_latents=f1_latents.detach().cpu().numpy(),
        f2_latent_vjp=grad_f2.cpu().numpy(),
        f1_latent_vjp=grad_f1.cpu().numpy(),
    )
    stage_reports = []
    with torch.no_grad():
        state = base
        for k in range(rounds):
            before = state
            state = f2(state, tuple(f2_latents[k, :len(f2.passes)].unbind(0)))
            for name in ("f2", "f1"):
                if name == "f1":
                    before = state
                    state = f1(state, f1_latents[k])
                archive = prefix.with_name(f"{prefix.name}_round{k + 1}_{name}.npz")
                np.savez_compressed(archive,
                                    vertices=state.cpu().numpy().astype(np.float32),
                                    boundary_reference=identity.cpu().numpy().astype(np.float32))
                certificate = certify_q1_binary_map(archive)
                if not certificate["valid"]:
                    raise ArithmeticError(f"invalid saved state {archive}")
                stage_reports.append({
                    "round": k + 1, "substep": name,
                    "minimum_normalized_q1_corner": float(
                        q1_corner_determinants(state).amin() * (side - 1) ** 2),
                    "componentwise_rms_change_from_substep_entry": float(
                        (state - before).square().mean().sqrt()),
                    "certificate": certificate, "saved_map": archive.name,
                })
                before = state
    if not torch.equal(final, state):
        raise ArithmeticError("timed and archived trajectories differ")
    report = {
        "question": "Can F2 then F1 safely compose repeatedly on one irregular mesh with finite VJP?",
        "scope": "geometry-only; no image encoder, anatomy, or training claim",
        "side": side, "control_vertices": side * side,
        "triangles": 2 * (side - 1) ** 2,
        "rounds": rounds, "batch": 1, "dtype": "float32", "device": device,
        "source": "x+0.05*sin(2*pi*x)*sin(pi*y), y",
        "target": "x+0.08*sin(2*pi*x)*sin(pi*y), y",
        "latent_seed": 290930, "latent_standard_deviation": .5,
        "f2_patch_cells": patch_cells, "f2_raw_span": f2_raw_span,
        "f1_raw_span": 8,
        "minimum_jacobian": minimum_jacobian,
        "checkpoint_rounds": checkpoint_rounds,
        "warmup": 1, "timed_repeats": repeats,
        "median_forward_seconds": statistics.median(item[0] for item in times),
        "median_forward_plus_vjp_seconds": statistics.median(item[1] for item in times),
        "peak_torch_cuda_allocated_bytes_during_timed_runs": timed_peak_bytes,
        "latent_and_vjp_archive": gradient_archive.name,
        "finite_f2_latent_vjp": bool(torch.isfinite(grad_f2).all()),
        "finite_f1_latent_vjp": bool(torch.isfinite(grad_f1).all()),
        "nonzero_f2_latent_vjp": bool((grad_f2 != 0).any()),
        "nonzero_f1_latent_vjp": bool((grad_f1 != 0).any()),
        "f2_latent_vjp_nonzero_fraction": float((grad_f2 != 0).float().mean()),
        "f1_latent_vjp_nonzero_fraction": float((grad_f1 != 0).float().mean()),
        "f2_latent_vjp_l2_norm": float(grad_f2.norm()),
        "f1_latent_vjp_l2_norm": float(grad_f1.norm()),
        "final_componentwise_coordinate_rmse_to_target": float(
            (final - target).square().mean().sqrt()),
        "stages": stage_reports,
    }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--minimum-jacobian", type=float, default=.05,
                        help="normalized Q1 corner floor; negative disables it")
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--f2-raw-span", type=float, default=.5)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.side, args.rounds, args.device, args.prefix,
                         args.repeats, not args.no_checkpoint,
                         None if args.minimum_jacobian < 0 else args.minimum_jacobian,
                         args.patch_cells, args.f2_raw_span),
          indent=2))


if __name__ == "__main__":
    main()
