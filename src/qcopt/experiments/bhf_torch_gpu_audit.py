"""Realistic-resolution GPU BHF near/far assembly and safe-flow audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.beltrami import face_jacobians
from qcopt.forward.bhf_torch import bhf_near_far_apply_torch
from qcopt.forward.safe_step import maximum_safe_step
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _face_fz(mesh, values):
    jac = face_jacobians(mesh, values)
    return 0.5 * ((jac[:, 0, 0] + jac[:, 1, 1]) + 1j * (jac[:, 1, 0] - jac[:, 0, 1]))


def run(output_dir: Path, n: int = 128, steps: int = 1, device: str = "cuda") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the requested GPU audit")
    mesh = structured_rectangle(n, n)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    base = 0.18 + 0.07j
    image = (1.0 - base) * source + base * np.conjugate(source)
    triangles = source[mesh.faces]
    variation = 0.04 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.38 - 0.42j) ** 2 / 0.12)
    records = []
    started_total = time.perf_counter()
    for iteration in range(steps):
        current_xy = np.column_stack((image.real, image.imag))
        fz = _face_fz(mesh, current_xy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        velocity = bhf_near_far_apply_torch(
            source, image, mesh.faces, fz, variation,
            device=device, dtype=torch.complex64, near_order=16,
            target_block_size=128, pair_block_size=2048,
        )
        np.save(output_dir / f"velocity_{iteration}.npy", velocity)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        assembly_seconds = time.perf_counter() - started
        delta = np.column_stack((velocity.real, velocity.imag))
        bound = maximum_safe_step(mesh, current_xy, delta, min_det_margin=0.15)
        step = min(0.2, 0.9 * bound if np.isfinite(bound) else 0.2)
        image = image + step * velocity
        report = audit_injectivity(mesh, np.column_stack((image.real, image.imag)), rectangle=False)
        records.append({
            "iteration": iteration,
            "assembly_seconds": assembly_seconds,
            "safe_bound": float(bound),
            "accepted_step": float(step),
            "min_determinant": float(report.minimum_signed_area_ratio),
            "flipped_faces": len(report.flipped_faces),
            "certified": bool(report.certified),
            "velocity_max": float(np.max(np.abs(velocity))),
        })
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "steps": steps,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started_total,
        "records": records,
        "scope": "GPU-native blocked BHF far quadrature plus vectorized incident-face Duffy correction",
        "limitation": "same discrete PV/atlas assumptions as the NumPy reference; this is an acceleration control, not a global BHF convergence theorem",
    }
    (output_dir / "bhf_torch_gpu_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.steps, args.device), indent=2))


if __name__ == "__main__":
    main()
