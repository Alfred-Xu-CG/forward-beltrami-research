"""Audit the memory-bounded multi-step real-pair BHF flow layer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.bhf_flow_recompute import _flow_forward
from qcopt.forward.bhf_flow_recompute import recompute_bhf_flow_real_pair
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def run(
    output_dir: Path,
    n: int,
    steps: int,
    device: str,
    adaptive_safe: bool = False,
    requested_step: float | None = None,
    variation_scale: float | None = None,
    smooth_safe: bool = False,
    smoothing: float = 1e-7,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    mesh = structured_rectangle(n, n)
    target_device = torch.device(device)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32, device=target_device)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long, device=target_device)
    x, y = source[:, 0], source[:, 1]
    initial = torch.stack((0.90 * x + 0.08 * y, 0.04 * x + 0.96 * y), dim=1).detach().clone().requires_grad_(True)
    centers = source[faces].mean(dim=1)
    amplitude_scale = (0.08 if adaptive_safe else 0.018) if variation_scale is None else float(variation_scale)
    amplitude = amplitude_scale * torch.exp(-((centers[:, 0] - 0.42) ** 2 + (centers[:, 1] - 0.55) ** 2) / 0.14)
    variation = torch.stack((amplitude, torch.zeros_like(amplitude)), dim=1)
    step_size = (0.5 if adaptive_safe else 0.02) if requested_step is None else float(requested_step)
    step_history: list[float] = []
    with torch.no_grad():
        _flow_forward(
            initial.detach(), source, faces, variation, step_size, steps,
            8, 128, 2048, adaptive_safe, 1e-6, 0.95, step_history,
            False, smooth_safe, smoothing,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = recompute_bhf_flow_real_pair(
        initial, source, faces, variation,
        step_size=step_size, steps=steps, near_order=8,
        target_block_size=128, pair_block_size=2048,
        adaptive_safe=adaptive_safe, min_det_margin=1e-6, safety=0.95,
        smooth_safe=smooth_safe, smoothing=smoothing,
    )
    loss = torch.sum(output * output)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    backward_started = time.perf_counter()
    loss.backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_started
    mapped = output.detach().cpu().numpy()
    report = audit_injectivity(mesh, mapped, rectangle=False)
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "steps": int(steps),
        "step_size": step_size,
        "adaptive_safe": bool(adaptive_safe),
        "smooth_safe": bool(smooth_safe),
        "smoothing": float(smoothing),
        "accepted_step_history": step_history,
        "device": str(device),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "loss": float(loss.detach().cpu()),
        "gradient_l2": float(torch.norm(initial.grad).detach().cpu()),
        "finite": bool(torch.isfinite(output).all() and torch.isfinite(initial.grad).all()),
        "injectivity_certified": bool(report.certified),
        "flipped_faces": int(len(report.flipped_faces)),
        "minimum_signed_area_ratio": float(report.minimum_signed_area_ratio),
        "scope": "multi-step BHF flow with trajectory recomputation in backward",
        "memory_model": "stores initial state and static mesh data; replays all steps during backward; no per-step dense interaction graphs are retained",
        "limitation": "recompute/checkpoint differentiation, not an implicit adjoint; global BHF PV theorem and atlas coupling remain open",
    }
    (output_dir / "bhf_flow_recompute_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adaptive-safe", action="store_true")
    parser.add_argument("--requested-step", type=float, default=None)
    parser.add_argument("--variation-scale", type=float, default=None)
    parser.add_argument("--smooth-safe", action="store_true")
    parser.add_argument("--smoothing", type=float, default=1e-7)
    args = parser.parse_args()
    print(json.dumps(run(
        args.output_dir, args.n, args.steps, args.device, args.adaptive_safe,
        args.requested_step, args.variation_scale, args.smooth_safe, args.smoothing,
    ), indent=2))


if __name__ == "__main__":
    main()
