"""Explicitly label-using F1 expressivity oracle; never a deployable result."""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, validate_q1_map,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.digital_lung_lesion3_score import CASES, PREFIX, scaled_landmarks
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_mind_sparse_match_finetune import p1_at_points
from tools.digital_q1_dhr_distill import identity_vertices


def run(canvas: Path, annotations: Path, case: str, baseline_path: Path,
        output: Path, *, steps: int, passes: int, device_name: str) -> dict:
    if case not in CASES or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("valid case and new output required")
    if steps < 1 or passes < 1:
        raise ValueError("positive steps and passes required")
    layout = json.loads((canvas / f"{case}_layout.json").read_text(encoding="utf-8"))
    fixed = scaled_landmarks(annotations / f"{PREFIX}He-les3.csv")
    moving = scaled_landmarks(annotations / f"{PREFIX}{CASES[case]}-les3.csv")
    names = sorted(fixed.keys() & moving.keys(), key=int)
    if len(names) != 80:
        raise ValueError("80 paired labels expected")
    fixed_px = np.stack([fixed[name] for name in names])
    moving_px = np.stack([moving[name] for name in names])
    query = original_pixel_to_canvas_unit(fixed_px, layout["fixed"], 512)
    target = original_pixel_to_canvas_unit(moving_px, layout["moving"], 512)
    with np.load(baseline_path) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        matrix = np.asarray(data["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(data["post_affine_offset"], dtype=np.float32)
    if vertices.shape != (1, 257, 257, 2):
        raise ValueError("257-square map required")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    device = torch.device(device_name)
    base = torch.from_numpy(vertices.copy()).to(device)
    q = torch.from_numpy(query.astype(np.float32)).to(device)
    target_t = torch.from_numpy(target.astype(np.float32)).to(device)
    m = torch.from_numpy(matrix.copy()).to(device)
    shift = torch.from_numpy(offset.copy()).to(device)
    logits = nn.ParameterList(nn.Parameter(torch.zeros((1, 255, 255, 2),
                                              device=device)) for _ in range(passes))
    update = AdaptiveSoftRadialQ1Relaxation(
        257, raw_span=8., safety_fraction=.75, minimum_jacobian=.05).to(device)
    optimizer = torch.optim.Adam(logits, lr=.02)
    best_total = float("inf")
    best = base.detach().clone()
    trace = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mapped = base
        for proposal in logits:
            mapped = update(mapped, proposal)
        residual_points = p1_at_points(mapped, q)
        pred = torch.stack((
            residual_points[:, 0] * m[0, 0] + residual_points[:, 1] * m[0, 1] + shift[0],
            residual_points[:, 0] * m[1, 0] + residual_points[:, 1] * m[1, 1] + shift[1],
        ), dim=1)
        label_mse = (pred - target_t).square().sum(-1).mean()
        strain = strain_penalty(mapped)
        loss = label_mse + .001 * strain
        value = float(loss.detach())
        if value < best_total:
            best_total = value
            best = mapped.detach().clone()
            best_step = step
            best_label_rmse_canvas_unit = float(torch.sqrt(label_mse.detach()))
            best_strain = float(strain.detach())
        loss.backward()
        if not all(proposal.grad is not None and
                   bool(torch.isfinite(proposal.grad).all()) for proposal in logits):
            raise FloatingPointError(f"invalid oracle VJP at {step}")
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "label_rmse_canvas_unit": float(
                torch.sqrt(label_mse.detach())), "strain": float(strain.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    identity = identity_vertices(257, device=device)
    if not validate_q1_map(best, identity)["valid"]:
        raise RuntimeError("oracle map invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=best.cpu().numpy(),
                        boundary_reference=identity.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"]:
        raise RuntimeError("saved oracle map invalid")
    report = {"method": "LANDMARK-USING NONDEPLOYABLE ORACLE for safe-layer expressivity",
              "case": case, "landmark_count": len(names), "steps": steps,
              "passes": passes,
              "learning_rate": .02, "strain_weight": .001,
              "initial_label_rmse_canvas_unit": trace[0]["label_rmse_canvas_unit"],
              "best_label_rmse_canvas_unit": best_label_rmse_canvas_unit,
              "best_step": best_step, "best_strain": best_strain,
              "seconds": elapsed,
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "labels_used_to_optimize": True,
              "not_registration_result": True,
              "saved_certificate": binary, "trace": trace}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "annotations", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(args.canvas, args.annotations, args.case, args.baseline,
                 args.output, steps=args.steps, passes=args.passes,
                 device_name=args.device)
    print(json.dumps({key: report[key] for key in (
        "case", "initial_label_rmse_canvas_unit",
        "best_label_rmse_canvas_unit", "best_step", "seconds",
    )}))


if __name__ == "__main__":
    main()
