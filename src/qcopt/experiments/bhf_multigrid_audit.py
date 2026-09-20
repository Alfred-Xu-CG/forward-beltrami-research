"""Audit a differentiable coarse-to-fine (multigrid-style) BHF layer.

The experiment compares a finest-grid BHF trajectory with a differentiable
coarse-to-fine trajectory.  Coarse vertex values are bilinearly prolonged with
PyTorch, so gradients can flow through the coarse BHF stage and the prolongation
operator.  The audit is deliberately modest: it measures speed, gradient
finiteness, map discrepancy, and independent triangle-orientation certificates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.forward.bhf_flow_recompute import recompute_bhf_flow_real_pair
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _grid_tensors(n: int, device: torch.device) -> tuple[object, torch.Tensor, torch.Tensor]:
    mesh = structured_rectangle(n, n)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long, device=device)
    return mesh, source, faces


def _initial(source: torch.Tensor) -> torch.Tensor:
    x, y = source[:, 0], source[:, 1]
    return torch.stack((0.90 * x + 0.08 * y, 0.04 * x + 0.96 * y), dim=1)


def _variation(source: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    centers = source[faces].mean(dim=1)
    amplitude = 0.018 * torch.exp(-((centers[:, 0] - 0.42) ** 2 + (centers[:, 1] - 0.55) ** 2) / 0.14)
    return torch.stack((amplitude, torch.zeros_like(amplitude)), dim=1)


def _prolongate(coarse: torch.Tensor, fine_n: int) -> torch.Tensor:
    coarse_n = int(round(coarse.shape[0] ** 0.5)) - 1
    image = coarse.reshape(1, coarse_n + 1, coarse_n + 1, 2).permute(0, 3, 1, 2)
    fine = F.interpolate(image, size=(fine_n + 1, fine_n + 1), mode="bilinear", align_corners=True)
    return fine.permute(0, 2, 3, 1).reshape((fine_n + 1) * (fine_n + 1), 2)


def _run_flow(initial: torch.Tensor, source: torch.Tensor, faces: torch.Tensor, variation: torch.Tensor, steps: int) -> torch.Tensor:
    return recompute_bhf_flow_real_pair(
        initial, source, faces, variation, step_size=0.02, steps=steps,
        near_order=8, target_block_size=128, pair_block_size=2048,
    )


def _record_map(mesh: object, mapped: torch.Tensor) -> dict:
    arr = mapped.detach().cpu().numpy()
    report = audit_injectivity(mesh, arr, rectangle=False)
    return {
        "injectivity_certified": bool(report.certified),
        "flipped_faces": int(len(report.flipped_faces)),
        "minimum_signed_area_ratio": float(report.minimum_signed_area_ratio),
    }


def run(output_dir: Path, coarse_n: int, fine_n: int, coarse_steps: int, fine_steps: int, device_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fine_mesh, fine_source, fine_faces = _grid_tensors(fine_n, device)
    coarse_mesh, coarse_source, coarse_faces = _grid_tensors(coarse_n, device)
    fine_var = _variation(fine_source, fine_faces)
    coarse_var = _variation(coarse_source, coarse_faces)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Finest-grid reference trajectory.
    reference_initial = _initial(fine_source).detach().clone().requires_grad_(True)
    sync(); t0 = time.perf_counter()
    reference = _run_flow(reference_initial, fine_source, fine_faces, fine_var, coarse_steps + fine_steps)
    sync(); ref_forward = time.perf_counter() - t0
    ref_loss = torch.sum(reference * reference)
    t0 = time.perf_counter(); ref_loss.backward(); sync(); ref_backward = time.perf_counter() - t0

    # Differentiable coarse-to-fine trajectory.
    coarse_initial = _initial(coarse_source).detach().clone().requires_grad_(True)
    t0 = time.perf_counter()
    coarse_output = _run_flow(coarse_initial, coarse_source, coarse_faces, coarse_var, coarse_steps)
    prolonged = _prolongate(coarse_output, fine_n)
    multigrid = _run_flow(prolonged, fine_source, fine_faces, fine_var, fine_steps)
    sync(); mg_forward = time.perf_counter() - t0
    mg_loss = torch.sum(multigrid * multigrid)
    t0 = time.perf_counter(); mg_loss.backward(); sync(); mg_backward = time.perf_counter() - t0

    map_error = torch.linalg.vector_norm(multigrid.detach() - reference.detach()) / torch.linalg.vector_norm(reference.detach())
    result = {
        "coarse_grid": f"{coarse_n}x{coarse_n} cells",
        "fine_grid": f"{fine_n}x{fine_n} cells",
        "coarse_steps": coarse_steps,
        "fine_steps": fine_steps,
        "total_reference_steps": coarse_steps + fine_steps,
        "device": str(device),
        "reference_forward_seconds": ref_forward,
        "reference_backward_seconds": ref_backward,
        "multigrid_forward_seconds": mg_forward,
        "multigrid_backward_seconds": mg_backward,
        "relative_map_error_to_fine_reference": float(map_error.detach().cpu()),
        "reference_gradient_l2": float(torch.linalg.vector_norm(reference_initial.grad).detach().cpu()),
        "coarse_gradient_l2": float(torch.linalg.vector_norm(coarse_initial.grad).detach().cpu()),
        "finite_gradients": bool(torch.isfinite(reference_initial.grad).all() and torch.isfinite(coarse_initial.grad).all()),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "reference": _record_map(fine_mesh, reference),
        "multigrid": _record_map(fine_mesh, multigrid),
        "scope": "differentiable coarse BHF -> bilinear prolongation -> fine BHF correction",
        "limitation": "coarse and fine stages use different meshes; this is a speed/gradient prototype, not a universal homeomorphism theorem",
    }
    (output_dir / "bhf_multigrid_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coarse-n", type=int, default=32)
    parser.add_argument("--fine-n", type=int, default=128)
    parser.add_argument("--coarse-steps", type=int, default=2)
    parser.add_argument("--fine-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.coarse_n, args.fine_n, args.coarse_steps, args.fine_steps, args.device), indent=2))


if __name__ == "__main__":
    main()
