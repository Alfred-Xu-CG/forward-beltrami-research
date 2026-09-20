"""Autograd cost and VJP audit for the tensor-native BHF assembly."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.bhf_torch import bhf_near_far_apply_torch_differentiable_real_pair
from qcopt.mesh import structured_rectangle


def _face_fz(source_pair: torch.Tensor, image_pair: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    source_tri = source_pair[faces]
    image_tri = image_pair[faces]
    dx1 = source_tri[:, 1, 0] - source_tri[:, 0, 0]
    dy1 = source_tri[:, 1, 1] - source_tri[:, 0, 1]
    dx2 = source_tri[:, 2, 0] - source_tri[:, 0, 0]
    dy2 = source_tri[:, 2, 1] - source_tri[:, 0, 1]
    determinant = dx1 * dy2 - dx2 * dy1
    du1 = image_tri[:, 1, 0] - image_tri[:, 0, 0]
    du2 = image_tri[:, 2, 0] - image_tri[:, 0, 0]
    dv1 = image_tri[:, 1, 1] - image_tri[:, 0, 1]
    dv2 = image_tri[:, 2, 1] - image_tri[:, 0, 1]
    ux = (du1 * dy2 - du2 * dy1) / determinant
    uy = (-du1 * dx2 + du2 * dx1) / determinant
    vx = (dv1 * dy2 - dv2 * dy1) / determinant
    vy = (-dv1 * dx2 + dv2 * dx1) / determinant
    return torch.stack((0.5 * (ux + vy), 0.5 * (vx - uy)), dim=1)


def run(output_dir: Path, n: int, device: str = "cuda") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    mesh = structured_rectangle(n, n)
    target_device = torch.device(device)
    source_pair = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32, device=target_device)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long, device=target_device)
    base_real = 0.82 * source_pair[:, 0] + 0.18 * source_pair[:, 0] + 0.07 * source_pair[:, 1]
    base_imag = 0.82 * source_pair[:, 1] + 0.18 * source_pair[:, 1] + 0.07 * source_pair[:, 0]
    image_pair = torch.stack((base_real, base_imag), dim=1).detach().clone().requires_grad_(True)
    source_tri = source_pair[faces]
    center_x = source_tri[:, :, 0].mean(dim=1)
    center_y = source_tri[:, :, 1].mean(dim=1)
    variation_scalar = 0.04 * torch.exp(-((center_x - 0.38) ** 2 + (center_y - 0.42) ** 2) / 0.12)
    variation = torch.stack((variation_scalar, torch.zeros_like(variation_scalar)), dim=1)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    fz = _face_fz(source_pair, image_pair, faces)
    velocity = bhf_near_far_apply_torch_differentiable_real_pair(
        source_pair, image_pair, faces, fz, variation, near_order=16,
        target_block_size=128, pair_block_size=2048,
    )
    # Spell out the real/imaginary energy.  This avoids an old Torch 1.7
    # complex-abs backward path that returns a complex gradient for a real
    # input tensor.
    loss = torch.sum(velocity * velocity)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    backward_started = time.perf_counter()
    # Legacy Torch 1.7 checkpointing supports ``backward`` but not
    # ``autograd.grad``.  The returned tensor is the same VJP used by the
    # modern explicit-grad CPU tests.
    loss.backward()
    gradient = image_pair.grad
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_started
    result = {
        "n": n,
        "faces": int(mesh.n_faces),
        "vertices": int(mesh.vertices.shape[0]),
        "device": str(device),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "loss": float(loss.detach().cpu()),
        "gradient_l2": float(torch.norm(gradient).detach().cpu()),
        "finite": bool(torch.isfinite(velocity).all() and torch.isfinite(gradient).all()),
        "scope": "fully real-valued tensor-native differentiable BHF near/far/Duffy assembly",
        "limitation": "single assembly VJP control; multi-step nonlinear flow still needs an implicit adjoint and atlas coupling",
    }
    (output_dir / "bhf_torch_autograd_gpu_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.device), indent=2))


if __name__ == "__main__":
    main()
