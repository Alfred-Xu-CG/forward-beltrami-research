"""Fixed-R2-boundary realizability projection at realistic resolution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.forward.compatibility_projection import project_facewise_mu
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle

from qcopt.experiments.compatibility_boundary_theorem_audit import side_sliding_map


def target_with_same_r2_boundary(vertices: np.ndarray) -> np.ndarray:
    """Add a smooth interior-only deformation to the side-sliding boundary."""

    base = side_sliding_map(vertices)
    x, y = vertices[:, 0], vertices[:, 1]
    envelope = np.sin(np.pi * x) * np.sin(np.pi * y)
    target = base.copy()
    target[:, 0] += 0.018 * envelope * np.sin(2.0 * np.pi * x + 0.3 * np.pi * y)
    target[:, 1] += 0.015 * envelope * np.cos(2.0 * np.pi * y - 0.2 * np.pi * x)
    return target


def run(output_dir: Path, n: int = 256, max_nfev: int = 5) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh = structured_rectangle(n, n)
    boundary_map = side_sliding_map(mesh.vertices)
    target_map = target_with_same_r2_boundary(mesh.vertices)
    target_mu = face_beltrami(mesh, target_map)
    initial_report = audit_injectivity(mesh, boundary_map, rectangle=True)
    projection = project_facewise_mu(
        mesh,
        target_mu,
        boundary_map=boundary_map,
        initial_map=boundary_map,
        max_nfev=max_nfev,
        regularization=1e-8,
    )
    projected_report = audit_injectivity(mesh, projection.map, rectangle=True)
    induced_error = float(np.linalg.norm(projection.projected_mu - target_mu) / max(np.linalg.norm(target_mu), 1e-15))
    boundary = mesh.boundary_loops[0]
    boundary_error = float(np.max(np.abs(projection.map[boundary] - boundary_map[boundary])))
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "max_nfev": int(max_nfev),
        "target_max_abs_mu": float(np.max(np.abs(target_mu))),
        "projection_success": bool(projection.success),
        "projection_nfev": int(projection.nfev),
        "projection_residual_l2": float(projection.residual_norm),
        "projection_relative_mu_error": induced_error,
        "boundary_max_error": boundary_error,
        "initial_boundary_certified": bool(initial_report.certified),
        "projected_injectivity_certified": bool(projected_report.certified),
        "projected_minimum_signed_area_ratio": float(projected_report.minimum_signed_area_ratio),
        "projected_flipped_faces": int(len(projected_report.flipped_faces)),
        "projected_rectangle_side_violations": list(projected_report.rectangle_side_violations),
        "finite": bool(np.all(np.isfinite(projection.map))),
        "scope": "fixed R2 rectangle-side boundary and nonlinear facewise-mu projection",
        "interpretation": "A compatible target field with a non-identity side-sliding boundary can be recovered by the sparse projection while preserving the independently audited R2 boundary and sampled PL topology.",
        "limitation": "Boundary tangential coordinates are fixed rather than optimized; arbitrary incompatible fields and projection uniqueness remain open.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "compatibility_rectangle_projection_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--max-nfev", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.max_nfev), indent=2))


if __name__ == "__main__":
    main()
