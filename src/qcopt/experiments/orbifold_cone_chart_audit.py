"""Realistic-resolution local orbifold cone-angle chart audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.orbifold import polar_disk_mesh, radial_cone_inverse, radial_cone_map


def main() -> None:
    started = time.perf_counter()
    vertices, faces = polar_disk_mesh(256, 512)
    records = []
    for alpha in (0.65, 1.4):
        mapped = radial_cone_map(vertices, alpha)
        recovered = radial_cone_inverse(mapped, alpha)
        p0, p1, p2 = mapped[faces].transpose(1, 0, 2)
        determinant = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        records.append({
            "angle_scale": alpha,
            "max_round_trip_error": float(np.max(np.abs(recovered - vertices))),
            "min_face_determinant": float(np.min(determinant)),
            "flipped_faces": int(np.count_nonzero(determinant <= 0.0)),
            "boundary_error": float(np.max(np.abs(mapped[-512:] - vertices[-512:]))),
        })
    result = {
        "radial_rings": 256,
        "angular_samples": 512,
        "vertices": len(vertices),
        "faces": len(faces),
        "records": records,
        "scope": "local cone-point hard chart with explicit inverse",
        "limitation": "does not yet glue arbitrary cone angles into a closed orbifold sphere or couple to BHF",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path("D:/QC_optimization/artifacts/orbifold_cone_chart_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "orbifold_cone_chart_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
