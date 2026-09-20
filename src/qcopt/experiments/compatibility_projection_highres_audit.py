"""Realistic-resolution sparse realizability projection audit.

This is intentionally a scaling/injectivity audit rather than a claim of a
global projection theorem.  It runs the fixed-boundary nonlinear projection
on a manufactured smooth map and records the sparse Jacobian structure at the
same resolution used by the other high-resolution route audits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.forward.benchmarks import smooth_twist_map
from qcopt.forward.compatibility_projection import project_facewise_mu, projection_mu_jacobian
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def run(
    output_dir: Path,
    n: int = 512,
    amplitude: float = 0.0015,
    max_nfev: int = 5,
    regularization: float = 1e-8,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    truth = smooth_twist_map(mesh.vertices, amplitude=amplitude)
    target = face_beltrami(mesh, truth)
    started = time.perf_counter()
    projection = project_facewise_mu(
        mesh,
        target,
        boundary_map=truth,
        initial_map=mesh.vertices,
        max_nfev=max_nfev,
        regularization=regularization,
    )
    interior = np.asarray(
        sorted(set(range(mesh.n_vertices)) - set(mesh.boundary_loops[0].tolist())),
        dtype=np.int64,
    )
    jacobian = projection_mu_jacobian(
        mesh,
        projection.map,
        regularization=regularization,
        interior=interior,
    )
    injectivity = audit_injectivity(mesh, projection.map)
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "free_coordinates": int(2 * len(interior)),
        "target_max_abs_mu": float(np.max(np.abs(target))),
        "projection_residual_norm": float(projection.residual_norm),
        "projection_nfev": int(projection.nfev),
        "projection_success": bool(projection.success),
        "jacobian_shape": [int(v) for v in jacobian.shape],
        "jacobian_nnz": int(jacobian.nnz),
        "jacobian_nnz_per_row": float(jacobian.nnz / jacobian.shape[0]),
        "min_face_determinant": float(injectivity.minimum_signed_area_ratio),
        "flipped_faces": len(injectivity.flipped_faces),
        "injectivity_certified": bool(injectivity.certified),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "512^2 fixed-boundary nonlinear realizability projection with sparse analytic Jacobian",
        "limitation": "high-resolution numerical scaling does not establish projection uniqueness, a global homeomorphism theorem, or arbitrary-mu realizability",
    }
    (output_dir / "compatibility_projection_highres_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--amplitude", type=float, default=0.0015)
    parser.add_argument("--max-nfev", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.amplitude, args.max_nfev), indent=2))


if __name__ == "__main__":
    main()
