"""Realistic-resolution sweep of two-chart radial cone exponents."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.experiments.orbifold_global_cone_atlas_audit import _cone_inverse, _cone_map
from qcopt.forward.sphere import sphere_face_orientation, uv_sphere_mesh


def run(output_dir: Path, n_lon: int = 512, n_lat: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source, faces = uv_sphere_mesh(n_lon, n_lat)
    exponent_pairs = ((0.35, 1.80), (0.50, 2.00), (0.80, 1.20), (1.20, 0.80), (1.80, 0.35))
    records = []
    started = time.perf_counter()
    for alpha_n, alpha_s in exponent_pairs:
        mapped = _cone_map(source, alpha_n, alpha_s)
        recovered = _cone_inverse(mapped, alpha_n, alpha_s)
        theta = np.arccos(np.clip(source[:, 2], -1.0, 1.0))
        seam = np.abs(theta - 0.5 * np.pi) <= 1e-13
        orientation = sphere_face_orientation(mapped, faces)
        records.append(
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
    result = {
        "n_lon": n_lon,
        "n_lat": n_lat,
        "source_vertices": int(len(source)),
        "faces": int(len(faces)),
        "exponent_pairs": [list(pair) for pair in exponent_pairs],
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "realistic two-chart analytic cone atlas exponent sweep",
        "limitation": "analytic radial transition control; does not prove arbitrary orbifold transition laws or nonlinear BHF coupling",
    }
    (output_dir / "orbifold_cone_exponent_sweep_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
