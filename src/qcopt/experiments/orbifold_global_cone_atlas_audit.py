"""Global two-chart cone-angle atlas audit on a closed spherical mesh.

This is a hard, analytic control: independent north/south radial cone maps
share the equator exactly.  It tests the global atlas bookkeeping and topology
gates that a future orbifold/BHF decoder must satisfy; it is not an arbitrary
Beltrami solver.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.sphere import sphere_face_orientation, uv_sphere_mesh


def _cone_map(vertices: np.ndarray, alpha_n: float, alpha_s: float) -> np.ndarray:
    """Map colatitude by power laws on the two hemispheres."""
    p = np.asarray(vertices, dtype=np.float64)
    theta = np.arccos(np.clip(p[:, 2], -1.0, 1.0))
    out_theta = theta.copy()
    north = theta <= 0.5 * np.pi
    south = ~north
    out_theta[north] = 0.5 * np.pi * (theta[north] / (0.5 * np.pi)) ** alpha_n
    south_phi = np.pi - theta[south]
    out_theta[south] = np.pi - 0.5 * np.pi * (south_phi / (0.5 * np.pi)) ** alpha_s
    longitude = np.arctan2(p[:, 1], p[:, 0])
    out = np.column_stack(
        (
            np.sin(out_theta) * np.cos(longitude),
            np.sin(out_theta) * np.sin(longitude),
            np.cos(out_theta),
        )
    )
    # Keep the two poles exact, avoiding any longitude dependence at sin(theta)=0.
    out[np.abs(theta) < 1e-14] = (0.0, 0.0, 1.0)
    out[np.abs(theta - np.pi) < 1e-14] = (0.0, 0.0, -1.0)
    return np.ascontiguousarray(out)


def _cone_inverse(vertices: np.ndarray, alpha_n: float, alpha_s: float) -> np.ndarray:
    """Inverse of :func:`_cone_map` for points on the unit sphere."""
    p = np.asarray(vertices, dtype=np.float64)
    theta = np.arccos(np.clip(p[:, 2], -1.0, 1.0))
    in_theta = theta.copy()
    north = theta <= 0.5 * np.pi
    south = ~north
    in_theta[north] = 0.5 * np.pi * (theta[north] / (0.5 * np.pi)) ** (1.0 / alpha_n)
    south_phi = np.pi - theta[south]
    in_theta[south] = np.pi - 0.5 * np.pi * (south_phi / (0.5 * np.pi)) ** (1.0 / alpha_s)
    longitude = np.arctan2(p[:, 1], p[:, 0])
    out = np.column_stack(
        (
            np.sin(in_theta) * np.cos(longitude),
            np.sin(in_theta) * np.sin(longitude),
            np.cos(in_theta),
        )
    )
    out[np.abs(theta) < 1e-14] = (0.0, 0.0, 1.0)
    out[np.abs(theta - np.pi) < 1e-14] = (0.0, 0.0, -1.0)
    return np.ascontiguousarray(out)


def run(output_dir: Path, n_lon: int = 512, n_lat: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source, faces = uv_sphere_mesh(n_lon, n_lat)
    t0 = time.perf_counter()
    cases = []
    for alpha_n, alpha_s in ((0.65, 1.40), (1.40, 0.65), (0.65, 0.65)):
        mapped = _cone_map(source, alpha_n, alpha_s)
        recovered = _cone_inverse(mapped, alpha_n, alpha_s)
        theta = np.arccos(np.clip(source[:, 2], -1.0, 1.0))
        seam = np.abs(theta - 0.5 * np.pi) <= 1e-13
        orientation = sphere_face_orientation(mapped, faces)
        cases.append(
            {
                "alpha_north": alpha_n,
                "alpha_south": alpha_s,
                "seam_vertices": int(np.count_nonzero(seam)),
                "seam_displacement_max": float(np.max(np.linalg.norm(mapped[seam] - source[seam], axis=1))),
                "sphere_norm_error": float(np.max(np.abs(np.linalg.norm(mapped, axis=1) - 1.0))),
                "inverse_roundtrip_max": float(np.max(np.linalg.norm(recovered - source, axis=1))),
                "min_spherical_orientation": float(np.min(orientation)),
                "flipped_faces": int(np.count_nonzero(orientation <= 0.0)),
            }
        )
    elapsed = time.perf_counter() - t0
    result = {
        "n_lon": n_lon,
        "n_lat": n_lat,
        "source_vertices": int(len(source)),
        "faces": int(len(faces)),
        "elapsed_seconds": elapsed,
        "cases": cases,
        "scope": "global two-cap radial cone atlas control; not an arbitrary-mu orbifold/BHF solver",
    }
    (output_dir / "orbifold_global_cone_atlas_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-lon", type=int, default=512)
    parser.add_argument("--n-lat", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n_lon, args.n_lat), indent=2))


if __name__ == "__main__":
    main()
