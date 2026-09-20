"""Implicit VJP audit for the sparse facewise Beltrami projection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from qcopt.beltrami import face_beltrami
from qcopt.forward.benchmarks import smooth_twist_map
from qcopt.forward.compatibility_projection import project_facewise_mu, projection_mu_jacobian
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def run(output_dir: Path, n: int = 128, amplitude: float = 0.003, regularization: float = 1e-8, eps: float = 2e-5, max_nfev: int = 20) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    truth = smooth_twist_map(mesh.vertices, amplitude=amplitude)
    target = face_beltrami(mesh, truth)
    interior = np.asarray(sorted(set(range(mesh.n_vertices)) - set(mesh.boundary_loops[0].tolist())), dtype=np.int64)
    started = time.perf_counter()
    base = project_facewise_mu(mesh, target, boundary_map=truth, initial_map=mesh.vertices, max_nfev=max_nfev, regularization=regularization)
    jacobian = projection_mu_jacobian(mesh, base.map, regularization=regularization, interior=interior)
    normal = (jacobian.T @ jacobian).tocsr()
    rng = np.random.default_rng(20260919)
    cotangent = rng.normal(size=2 * len(interior))
    adjoint = spsolve(normal, cotangent)
    target_gradient = np.asarray(jacobian @ adjoint)[: 2 * mesh.n_faces]
    directions = {
        "arbitrary_target": rng.normal(size=2 * mesh.n_faces),
        "tangent_target": np.asarray(jacobian @ rng.normal(size=2 * len(interior)))[: 2 * mesh.n_faces],
    }
    checks = {}
    for name, direction in directions.items():
        direction = direction / np.linalg.norm(direction)
        direction_mu = direction[: mesh.n_faces] + 1j * direction[mesh.n_faces :]
        plus = project_facewise_mu(mesh, target + eps * direction_mu, boundary_map=truth, initial_map=base.map, max_nfev=max_nfev, regularization=regularization)
        minus = project_facewise_mu(mesh, target - eps * direction_mu, boundary_map=truth, initial_map=base.map, max_nfev=max_nfev, regularization=regularization)
        dx = (plus.map[interior] - minus.map[interior]).reshape(-1) / (2.0 * eps)
        finite_directional = float(cotangent @ dx)
        implicit_directional = float(target_gradient @ direction)
        checks[name] = {
            "finite_directional": finite_directional,
            "implicit_directional": implicit_directional,
            "implicit_directional_abs_error": abs(finite_directional - implicit_directional),
            "plus_residual_norm": float(plus.residual_norm),
            "minus_residual_norm": float(minus.residual_norm),
        }
    report = audit_injectivity(mesh, base.map)
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "free_coordinates": int(2 * len(interior)),
        "projection_residual_norm": float(base.residual_norm),
        "projection_nfev": int(base.nfev),
        "jacobian_shape": [int(v) for v in jacobian.shape],
        "jacobian_nnz": int(jacobian.nnz),
        "normal_nnz": int(normal.nnz),
        "regularization": regularization,
        "eps": eps,
        "max_nfev": max_nfev,
        "directional_checks": checks,
        "min_face_determinant": float(report.minimum_signed_area_ratio),
        "flipped_faces": len(report.flipped_faces),
        "injectivity_certified": bool(report.certified),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "implicit sparse normal-equation VJP of fixed-boundary facewise Beltrami projection",
        "limitation": "normal-equation conditioning, projection uniqueness, active topology changes, and global homeomorphism theorem remain open",
    }
    (output_dir / "compatibility_projection_vjp_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--amplitude", type=float, default=0.003)
    parser.add_argument("--eps", type=float, default=2e-5)
    parser.add_argument("--max-nfev", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.amplitude, eps=args.eps, max_nfev=args.max_nfev), indent=2))


if __name__ == "__main__":
    main()
