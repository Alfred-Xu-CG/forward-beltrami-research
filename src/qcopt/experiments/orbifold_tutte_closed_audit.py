"""Closed-sphere two-cap Tutte seam/orientation audit."""

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


def _solve_cap(vertices: np.ndarray, faces: np.ndarray, hemisphere: str):
    pole = "south" if hemisphere == "north" else "north"
    keep = np.all(
        vertices[faces, 2] >= -1e-12 if hemisphere == "north" else vertices[faces, 2] <= 1e-12,
        axis=1,
    )
    cap_faces = faces[keep]
    used = np.unique(cap_faces)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    cap_faces = remap[cap_faces]
    chart = stereographic_forward(vertices[used], pole=pole)
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
    target_chart = stereographic_forward(vertices[used[loop]], pole=pole)
    target = np.column_stack((target_chart.real, target_chart.imag))
    mapped_planar = tutte_embedding(mesh, target)
    planar_report = audit_injectivity(mesh, mapped_planar)
    # Both stereographic charts use a coordinate convention opposite to the
    # outward sphere orientation; conjugation restores the spherical sign.
    mapped_planar[:, 1] *= -1.0
    mapped_sphere = stereographic_inverse(
        mapped_planar[:, 0] + 1j * mapped_planar[:, 1], pole=pole
    )
    return used, cap_faces, mapped_sphere, planar_report


def run(output_dir: Path, n_lon: int = 256, n_lat: int = 128) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = uv_sphere_mesh(n_lon, n_lat)
    t0 = time.perf_counter()
    north = _solve_cap(vertices, faces, "north")
    south = _solve_cap(vertices, faces, "south")
    elapsed = time.perf_counter() - t0
    north_map = {int(i): p for i, p in zip(north[0], north[2])}
    south_map = {int(i): p for i, p in zip(south[0], south[2])}
    shared = sorted(set(north_map) & set(south_map))
    seam_error = max(np.linalg.norm(north_map[i] - south_map[i]) for i in shared)
    result = {
        "n_lon": n_lon,
        "n_lat": n_lat,
        "source_vertices": int(len(vertices)),
        "north_faces": int(len(north[1])),
        "south_faces": int(len(south[1])),
        "seam_vertices": int(len(shared)),
        "elapsed_seconds": elapsed,
        "north_planar_certified": north[3].certified,
        "south_planar_certified": south[3].certified,
        "north_min_spherical_orientation": float(np.min(sphere_face_orientation(north[2], north[1]))),
        "south_min_spherical_orientation": float(np.min(sphere_face_orientation(south[2], south[1]))),
        "seam_max_mismatch": float(seam_error),
        "norm_error_north": float(np.max(np.abs(np.linalg.norm(north[2], axis=1) - 1.0))),
        "norm_error_south": float(np.max(np.abs(np.linalg.norm(south[2], axis=1) - 1.0))),
        "scope": "closed sphere assembled from two positive-Tutte caps; arbitrary orbifold cone data not included",
    }
    (output_dir / "orbifold_tutte_closed_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-lon", type=int, default=256)
    parser.add_argument("--n-lat", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n_lon, args.n_lat), indent=2))


if __name__ == "__main__":
    main()
