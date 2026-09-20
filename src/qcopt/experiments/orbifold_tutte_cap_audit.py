"""Spherical-cap Tutte baseline and chart-orientation audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.sphere import sphere_face_orientation, uv_sphere_mesh
from qcopt.forward.sphere_charts import stereographic_forward, stereographic_inverse
from qcopt.forward.tutte import tutte_embedding
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import TriMesh


def run(output_dir: Path, n_lon: int = 256, n_lat: int = 128) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = uv_sphere_mesh(n_lon, n_lat)
    keep = np.all(vertices[faces, 2] >= -1e-12, axis=1)
    cap_faces = faces[keep]
    used = np.unique(cap_faces)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    cap_faces = remap[cap_faces]
    chart = stereographic_forward(vertices[used], pole="south")
    planar = np.column_stack((chart.real, chart.imag))
    cross = (
        (planar[cap_faces[:, 1], 0] - planar[cap_faces[:, 0], 0])
        * (planar[cap_faces[:, 2], 1] - planar[cap_faces[:, 0], 1])
        - (planar[cap_faces[:, 1], 1] - planar[cap_faces[:, 0], 1])
        * (planar[cap_faces[:, 2], 0] - planar[cap_faces[:, 0], 0])
    )
    if np.min(cross) < 0.0:
        cap_faces = cap_faces[:, [0, 2, 1]]
    mesh = TriMesh(planar, cap_faces)
    loop = mesh.boundary_loops[0]
    angles = np.angle(chart[loop])
    target = np.column_stack((np.cos(angles), np.sin(angles)))
    t0 = time.perf_counter()
    mapped_planar = tutte_embedding(mesh, target)
    solve_seconds = time.perf_counter() - t0
    planar_report = audit_injectivity(mesh, mapped_planar)
    # The south-chart orientation convention reverses the planar orientation
    # relative to outward sphere orientation; conjugating the output restores it.
    outward_planar = mapped_planar.copy()
    outward_planar[:, 1] *= -1.0
    mapped_sphere = stereographic_inverse(
        outward_planar[:, 0] + 1j * outward_planar[:, 1], pole="south"
    )
    result = {
        "n_lon": n_lon,
        "n_lat": n_lat,
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "boundary_vertices": int(len(loop)),
        "solve_seconds": solve_seconds,
        "planar_tutte_certified_before_chart_orientation_fix": planar_report.certified,
        "sphere_min_orientation_after_chart_fix": float(np.min(sphere_face_orientation(mapped_sphere, cap_faces))),
        "sphere_norm_error": float(np.max(np.abs(np.linalg.norm(mapped_sphere, axis=1) - 1.0))),
        "scope": "spherical-cap orbifold/Tutte baseline, not a full closed-sphere solver",
    }
    (output_dir / "orbifold_tutte_cap_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
