"""Small-mesh nonlinear realizability projection audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.forward.benchmarks import smooth_twist_map
from qcopt.forward.compatibility_projection import project_facewise_mu
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n, amplitude in ((4, 0.03), (8, 0.02), (16, 0.01), (32, 0.006), (64, 0.003)):
        mesh = structured_rectangle(n, n)
        truth = smooth_twist_map(mesh.vertices, amplitude=amplitude)
        target = face_beltrami(mesh, truth)
        t0 = time.perf_counter()
        result = project_facewise_mu(
            mesh,
            target,
            boundary_map=truth,
            initial_map=mesh.vertices,
            max_nfev=120,
        )
        records.append({
            "grid_cells_per_axis": n,
            "faces": mesh.n_faces,
            "target_max_abs_mu": float(np.max(np.abs(target))),
            "residual_norm": result.residual_norm,
            "nfev": result.nfev,
            "elapsed_seconds": time.perf_counter() - t0,
            "success": result.success,
            "injectivity_certified": bool(audit_injectivity(mesh, result.map, rectangle=False).certified),
        })
    n = 5
    mesh = structured_rectangle(n, n)
    rng = np.random.default_rng(2026)
    random_mu = 0.45 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    random_mu *= np.minimum(1.0, 0.65 / np.maximum(np.abs(random_mu), 1e-12))
    random_result = project_facewise_mu(mesh, random_mu, max_nfev=150)
    random_audit = audit_injectivity(mesh, random_result.map, rectangle=True)
    result = {
        "manufactured_records": records,
        "random_incompatible": {
            "faces": mesh.n_faces,
            "target_max_abs_mu": float(np.max(np.abs(random_mu))),
            "residual_norm": random_result.residual_norm,
            "success": random_result.success,
            "nfev": random_result.nfev,
            "injectivity_certified": bool(random_audit.certified),
            "flipped_faces": len(random_audit.flipped_faces),
        },
        "scope": "nonlinear facewise-mu projection with fixed boundary on small meshes",
        "limitation": "analytic sparse Jacobian improves medium-mesh scaling, but there is still no global homeomorphism theorem or 256² projection",
    }
    (output_dir / "compatibility_projection_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
