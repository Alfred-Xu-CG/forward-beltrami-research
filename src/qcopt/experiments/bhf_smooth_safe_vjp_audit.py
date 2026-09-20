"""Directional-derivative audit for the smooth-safe BHF replay layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.bhf_flow_recompute import recompute_bhf_flow_real_pair
from qcopt.mesh import structured_rectangle


def _problem(n: int, device: torch.device):
    mesh = structured_rectangle(n, n)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long, device=device)
    x, y = source[:, 0], source[:, 1]
    initial = torch.stack((0.92 * x + 0.05 * y, 0.03 * x + 0.94 * y), dim=1)
    centers = source[faces].mean(dim=1)
    amplitude = 0.12 * torch.exp(-((centers[:, 0] - 0.4) ** 2 + (centers[:, 1] - 0.6) ** 2) / 0.12)
    variation = torch.stack((amplitude, torch.zeros_like(amplitude)), dim=1)
    direction = torch.stack((
        torch.sin(3.0 * np.pi * x) * torch.cos(2.0 * np.pi * y),
        torch.cos(2.0 * np.pi * x) * torch.sin(3.0 * np.pi * y),
    ), dim=1)
    direction = direction / torch.norm(direction)
    return mesh, source, faces, initial, variation, direction


def _evaluate(initial, source, faces, variation, *, step_size, steps, near_order, target_block_size, pair_block_size, min_det_margin, safety, smoothing):
    output = recompute_bhf_flow_real_pair(
        initial, source, faces, variation,
        step_size=step_size, steps=steps, near_order=near_order,
        target_block_size=target_block_size, pair_block_size=pair_block_size,
        adaptive_safe=True, smooth_safe=True,
        min_det_margin=min_det_margin, safety=safety, smoothing=smoothing,
    )
    return torch.sum(output * output)


def run(output_dir: Path, n: int, device_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    mesh, source, faces, initial, variation, direction = _problem(n, device)
    kwargs = dict(step_size=10.0, steps=2, near_order=4, target_block_size=64,
                  pair_block_size=512, min_det_margin=1e-5, safety=0.95,
                  smoothing=1e-7)
    base = initial.detach().clone().requires_grad_(True)
    loss = _evaluate(base, source, faces, variation, **kwargs)
    loss.backward()
    directional = float(torch.sum(base.grad * direction).detach().cpu())
    rows = []
    for eps in (1e-3, 3e-4, 1e-4, 3e-5):
        with torch.no_grad():
            plus = _evaluate(initial + eps * direction, source, faces, variation, **kwargs)
            minus = _evaluate(initial - eps * direction, source, faces, variation, **kwargs)
        finite_difference = float(((plus - minus) / (2.0 * eps)).detach().cpu())
        error = abs(finite_difference - directional)
        rows.append({
            "epsilon": eps,
            "finite_difference": finite_difference,
            "absolute_error": error,
            "relative_error": error / max(1.0, abs(finite_difference), abs(directional)),
        })
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "device": device_name,
        "step_size": kwargs["step_size"],
        "steps": kwargs["steps"],
        "smooth_safe": True,
        "smoothing": kwargs["smoothing"],
        "base_loss": float(loss.detach().cpu()),
        "analytic_directional_derivative": directional,
        "finite_difference_rows": rows,
        "scope": "smooth determinant-safe BHF replay VJP under active clipping",
        "limitation": "directional check is local; global BHF PV theorem and implicit adjoint remain open",
    }
    (output_dir / "bhf_smooth_safe_vjp_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.device), indent=2))


if __name__ == "__main__":
    main()
