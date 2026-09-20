"""Adaptive explicit BHF-flow control using near/far vertex assembly."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_jacobians
from qcopt.forward.bhf_variation import all_vertex_near_far_bhf_variation
from qcopt.forward.safe_step import maximum_safe_step
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _face_fz(mesh, values):
    jac = face_jacobians(mesh, values)
    ux, uy = jac[:, 0, 0], jac[:, 0, 1]
    vx, vy = jac[:, 1, 0], jac[:, 1, 1]
    return 0.5 * ((ux + vy) + 1j * (vx - uy))


def run(output_dir: Path, n: int = 32, steps: int = 2) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    b = 0.18 + 0.07j
    a = 1.0 - b
    current = np.column_stack((
        (a * source + b * np.conjugate(source)).real,
        (a * source + b * np.conjugate(source)).imag,
    ))
    triangles = source[mesh.faces]
    variation = 0.04 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.38 - 0.42j) ** 2 / 0.12)
    records = []
    t0 = time.perf_counter()
    for iteration in range(steps):
        image = current[:, 0] + 1j * current[:, 1]
        fz = _face_fz(mesh, current)
        velocity = all_vertex_near_far_bhf_variation(
            source,
            image,
            mesh.faces,
            fz,
            variation,
            near_order=8,
        )
        delta = np.column_stack((velocity.real, velocity.imag))
        bound = maximum_safe_step(mesh, current, delta, min_det_margin=0.15)
        step = min(0.2, 0.9 * bound if np.isfinite(bound) else 0.2)
        current = current + step * delta
        report = audit_injectivity(mesh, current, rectangle=False)
        records.append(
            {
                "iteration": iteration,
                "safe_bound": float(bound),
                "accepted_step": float(step),
                "min_determinant": float(report.minimum_signed_area_ratio),
                "flipped_faces": len(report.flipped_faces),
                "certified": bool(report.certified),
                "velocity_max": float(np.max(np.linalg.norm(delta, axis=1))),
            }
        )
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - t0,
        "records": records,
        "scope": "adaptive nonlinear explicit BHF flow with Duffy-corrected vertex near fields",
        "limitation": "regular-grid planar control only; full sphere atlas gluing, edge/vertex PV cancellation, and convergence to a prescribed Beltrami target remain open",
    }
    (output_dir / "bhf_near_far_flow_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, n=args.n, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
