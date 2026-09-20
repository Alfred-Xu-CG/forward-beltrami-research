"""Directed implicit Tutte VJP audit on a nonuniform Delaunay mesh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

from qcopt.forward.tutte_directed_implicit import DirectedTutteSystem, directed_tutte_embedding_torch_implicit
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import TriMesh

from .mmatrix_global_unstructured_audit import _orient_faces, _square_points


def _target_boundary(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    linear = np.array([[1.07, 0.13], [0.03, 0.94]], dtype=np.float64)
    return np.ascontiguousarray(points[loop] @ linear.T)


def run(output_dir: Path, interior: int = 3800, side: int = 64, seed: int = 20260919) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(seed)
    points, _ = _square_points(rng, interior, side)
    faces = _orient_faces(points, Delaunay(points).simplices)
    mesh = TriMesh(points, faces)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = _target_boundary(points, mesh.boundary_loops[0])
    records = []
    for spread in (1.0, 3.0):
        logits_np = np.zeros((system.n_rows, system.max_degree), dtype=np.float64)
        logits_np[system.valid_mask] = spread * rng.normal(size=int(np.count_nonzero(system.valid_mask)))
        boundary_t = torch.as_tensor(boundary, dtype=torch.float64)
        logits_t = torch.as_tensor(logits_np, dtype=torch.float64).requires_grad_(True)
        started = time.perf_counter()
        mapped_t = directed_tutte_embedding_torch_implicit(mesh, boundary_t, logits_t, system)
        reference = torch.as_tensor(points[system.interior], dtype=torch.float64)
        loss = torch.mean((mapped_t[system.interior] - reference) ** 2)
        loss.backward()
        elapsed = time.perf_counter() - started
        mapped = mapped_t.detach().cpu().numpy()
        report = audit_injectivity(mesh, mapped, rectangle=False)
        records.append(
            {
                "logit_spread": float(spread),
                "elapsed_seconds": float(elapsed),
                "loss": float(loss.detach().cpu()),
                "finite_output": bool(torch.isfinite(mapped_t).all()),
                "finite_gradient": bool(torch.isfinite(logits_t.grad).all()),
                "gradient_l2": float(torch.linalg.vector_norm(logits_t.grad).detach().cpu()),
                "flipped_faces": int(len(report.flipped_faces)),
                "minimum_signed_area": float(report.minimum_signed_area_ratio),
                "injectivity_certified": bool(report.certified),
                "boundary_intersections": int(len(report.boundary_intersections)),
                "bad_branch_vertices": int(len(report.bad_branch_vertices)),
            }
        )
    result = {
        "vertices": int(mesh.n_vertices),
        "boundary_vertices": int(len(mesh.boundary_loops[0])),
        "faces": int(mesh.n_faces),
        "interior_rows": int(system.n_rows),
        "max_degree": int(system.max_degree),
        "records": records,
        "scope": "directed positive-row implicit Tutte layer on a nonuniform Delaunay triangulation",
        "limitation": "finite unstructured stress evidence; no general nonsymmetric theorem or arbitrary-mu QC consistency claim",
    }
    (output_dir / "tutte_directed_unstructured_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interior", type=int, default=3800)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.interior, args.side, args.seed), indent=2))


if __name__ == "__main__":
    main()
