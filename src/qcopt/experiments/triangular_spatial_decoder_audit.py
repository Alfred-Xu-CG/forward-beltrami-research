"""High-resolution audit of the spatially varying triangular hard decoder."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.coupled_decoder import triangular_spatial_monotone_inverse, triangular_spatial_monotone_map
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(20260919)
    n = 512
    mesh = structured_rectangle(n, n)
    points = mesh.vertices.copy()
    x_inc = np.exp(0.8 * rng.normal(size=96))
    y_inc = np.exp(0.8 * rng.normal(size=(96, 128)))
    mapped = triangular_spatial_monotone_map(points, x_inc, y_inc)
    inverse_points = triangular_spatial_monotone_inverse(mapped, x_inc, y_inc)
    report = audit_injectivity(mesh, mapped, rectangle=True)
    vectors = np.stack((mapped[mesh.faces[:, 1]] - mapped[mesh.faces[:, 0]], mapped[mesh.faces[:, 2]] - mapped[mesh.faces[:, 0]]), axis=-1)
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "vertices": len(points),
        "x_control_intervals": len(x_inc),
        "y_control_columns": y_inc.shape[0],
        "y_control_intervals": y_inc.shape[1],
        "max_round_trip_error": float(np.max(np.abs(inverse_points - points))),
        "min_increment": float(min(np.min(x_inc), np.min(y_inc))),
        "min_determinant": float(np.min(np.linalg.det(vectors))),
        "flipped_faces": len(report.flipped_faces),
        "injectivity_certified": bool(report.certified),
        "elapsed_seconds": time.perf_counter() - started,
        "limitation": "triangular hard decoder, not arbitrary Beltrami expressivity or a full diffusion model",
    }
    output = Path("D:/QC_optimization/artifacts/triangular_spatial_decoder_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "triangular_spatial_decoder_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
