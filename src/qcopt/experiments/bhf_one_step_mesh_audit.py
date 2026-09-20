"""One-step arbitrary-base BHF flow plus independent PL orientation audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import triangle_bhf_variation
from qcopt.mesh import structured_rectangle
from qcopt.beltrami import face_jacobians


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(64, 64)
    z = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    faces = mesh.faces
    tri = z[faces]
    b = 0.18 + 0.07j
    a = 1.0 - b
    tri_image = a * tri + b * np.conjugate(tri)
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    nu = 0.16 * np.exp(-np.abs(np.mean(tri, axis=1) - 0.38 - 0.42j) ** 2 / 0.12)
    t0 = time.perf_counter()
    velocity = triangle_bhf_variation(z, a * z + b * np.conjugate(z), tri, tri_image, fz, nu)
    velocity_xy = np.column_stack((velocity.real, velocity.imag))
    records = []
    for dt in (0.01, 0.05, 0.10):
        # Use the exact current affine map in complex form and add the BHF
        # residual.
        current = np.column_stack(( (a*z+b*np.conjugate(z)).real, (a*z+b*np.conjugate(z)).imag ))
        candidate = current + dt * velocity_xy
        jac = face_jacobians(mesh, candidate)
        det = np.linalg.det(jac)
        records.append({
            "dt": dt,
            "min_determinant": float(np.min(det)),
            "flipped_faces": int(np.sum(det <= 0.0)),
            "finite": bool(np.all(np.isfinite(candidate))),
        })
    result = {
        "grid": "64x64 cells / 8192 faces",
        "vertex_count": mesh.n_vertices,
        "velocity_seconds": time.perf_counter() - t0,
        "records": records,
        "scope": "one explicit arbitrary-base BHF step with piecewise-affine quadrature",
        "limitations": [
            "step sizes are diagnostic and are not an adaptive determinant-root controller",
            "the BHF velocity uses fixed quadrature rather than exact triangle integrals",
            "this does not establish global nonlinear flow convergence",
        ],
    }
    (output_dir / "bhf_one_step_mesh_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
