"""Realistic unstructured-mesh BHF near/far/Duffy GPU audit.

The regular-grid BHF experiments do not test whether the mesh-native
quadrature survives loss of translational structure.  This standalone audit
builds a jittered rectangular Delaunay mesh, runs the real-pair differentiable
GPU assembly, compares two local Duffy orders, and applies determinant-safe
candidate steps with an independent face-orientation/boundary audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay


def _load_bhf(path: str):
    spec = importlib.util.spec_from_file_location("bhf_torch_remote", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bhf_torch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mesh(n: int, seed: int = 20260920):
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    h = 1.0 / n
    interior = (xx > 0.0) & (xx < 1.0) & (yy > 0.0) & (yy < 1.0)
    xx[interior] += rng.uniform(-0.22 * h, 0.22 * h, size=int(np.sum(interior)))
    yy[interior] += rng.uniform(-0.22 * h, 0.22 * h, size=int(np.sum(interior)))
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    faces = Delaunay(points).simplices.astype(np.int64)
    signed = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    faces[signed < 0.0, 1], faces[signed < 0.0, 2] = faces[signed < 0.0, 2], faces[signed < 0.0, 1]
    boundary = np.flatnonzero((np.isclose(points[:, 0], 0.0)) | (np.isclose(points[:, 0], 1.0)) |
                              (np.isclose(points[:, 1], 0.0)) | (np.isclose(points[:, 1], 1.0)))
    return points, faces, boundary


def _face_fz(source: torch.Tensor, image: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    st = source[faces]
    it = image[faces]
    e1 = st[:, 1] - st[:, 0]
    e2 = st[:, 2] - st[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    du1, du2 = it[:, 1, 0] - it[:, 0, 0], it[:, 2, 0] - it[:, 0, 0]
    dv1, dv2 = it[:, 1, 1] - it[:, 0, 1], it[:, 2, 1] - it[:, 0, 1]
    ux = (du1 * e2[:, 1] - du2 * e1[:, 1]) / det
    uy = (-du1 * e2[:, 0] + du2 * e1[:, 0]) / det
    vx = (dv1 * e2[:, 1] - dv2 * e1[:, 1]) / det
    vy = (-dv1 * e2[:, 0] + dv2 * e1[:, 0]) / det
    return torch.stack((0.5 * (ux + vy), 0.5 * (vx - uy)), dim=1)


def _audit_map(points: np.ndarray, faces: np.ndarray, mapped: np.ndarray, boundary: np.ndarray) -> dict[str, object]:
    signed = np.cross(mapped[faces[:, 1]] - mapped[faces[:, 0]], mapped[faces[:, 2]] - mapped[faces[:, 0]])
    # Boundary order is inherited from the square grid; sort by perimeter arc.
    bxy = mapped[boundary]
    arc = np.where(np.isclose(points[boundary, 1], 0.0), points[boundary, 0],
                   np.where(np.isclose(points[boundary, 0], 1.0), 1.0 + points[boundary, 1],
                            np.where(np.isclose(points[boundary, 1], 1.0), 3.0 - points[boundary, 0],
                                     4.0 - points[boundary, 1])))
    order = np.argsort(arc)
    loop = bxy[order]
    edges = np.roll(loop, -1, axis=0) - loop
    edge_cross = np.cross(loop, np.roll(loop, -1, axis=0))
    # A lightweight independent boundary check: signed polygon area and finite edges.
    polygon_area = 0.5 * float(np.sum(edge_cross))
    return {
        "min_signed_face_area": float(np.min(signed)),
        "minimum_signed_area_ratio": float(np.min(signed) / np.median(signed)),
        "flipped_faces": int(np.sum(signed <= 0.0)),
        "boundary_polygon_area": polygon_area,
        "boundary_orientation_ok": bool(polygon_area > 0.0),
        "boundary_edges_finite": bool(np.all(np.isfinite(edges))),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(signed))),
    }


def run(output_dir: Path, bhf_path: str, n: int = 128, device: str = "cuda") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    bhf = _load_bhf(bhf_path)
    points, faces_np, boundary = _mesh(n)
    source = torch.as_tensor(points, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
    x, y = source[:, 0], source[:, 1]
    image = torch.stack((x + 0.045 * torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y),
                         y + 0.035 * torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)), dim=1)
    image = image.detach().clone().requires_grad_(True)
    tri = source[faces]
    cx, cy = tri[:, :, 0].mean(dim=1), tri[:, :, 1].mean(dim=1)
    nu_scalar = 0.035 * torch.exp(-((cx - 0.43) ** 2 + (cy - 0.54) ** 2) / 0.10)
    variation = torch.stack((nu_scalar, torch.zeros_like(nu_scalar)), dim=1)
    records: list[dict[str, object]] = []
    velocities: dict[int, torch.Tensor] = {}
    for order in (8, 16):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        fz = _face_fz(source, image, faces)
        velocity = bhf.bhf_near_far_apply_torch_differentiable_real_pair(
            source, image, faces, fz, variation, near_order=order,
            target_block_size=128, pair_block_size=2048,
        )
        loss = torch.sum(velocity * velocity)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        image.grad = None
        loss.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        records.append({
            "near_order": order,
            "forward_seconds": forward_seconds,
            "backward_gradient_l2": float(torch.norm(image.grad).detach().cpu()),
            "velocity_l2": float(torch.norm(velocity).detach().cpu()),
            "finite": bool(torch.isfinite(velocity).all() and torch.isfinite(image.grad).all()),
        })
        velocities[order] = velocity.detach().cpu()
    velocity8, velocity16 = velocities[8], velocities[16]
    records[-1]["order8_to_order16_relative_difference"] = float(
        torch.norm(velocity16 - velocity8) / torch.clamp(torch.norm(velocity16), min=1e-20)
    )
    base = image.detach().cpu().numpy()
    step_records = []
    velocity_np = velocity16.numpy()
    for step in (0.25, 0.5, 1.0):
        mapped = base + step * velocity_np
        audit = _audit_map(points, faces_np, mapped, boundary)
        audit["step"] = step
        step_records.append(audit)
    result = {
        "schema": "forward-beltrami-bhf-unstructured-gpu-v1",
        "n": n,
        "vertices": int(len(points)),
        "faces": int(len(faces_np)),
        "boundary_vertices": int(len(boundary)),
        "device": str(device),
        "quadrature_records": records,
        "step_records": step_records,
        "scope": "jittered rectangular Delaunay mesh with differentiable real-pair BHF near/far/Duffy assembly",
        "limitation": "one-step unstructured control; a global adaptive PV theorem, multi-step implicit adjoint, and boundary self-intersection certificate remain open",
    }
    (output_dir / "bhf_unstructured_gpu_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bhf-path", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.bhf_path, args.n, args.device), indent=2))


if __name__ == "__main__":
    main()
