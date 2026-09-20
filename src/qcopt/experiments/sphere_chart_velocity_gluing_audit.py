"""High-resolution velocity gluing audit for the two stereographic charts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.sphere import uv_sphere_mesh
from qcopt.forward.sphere_charts import (
    stereographic_forward,
    stereographic_transition,
    stereographic_transition_velocity,
)


def main() -> None:
    started = time.perf_counter()
    vertices, _ = uv_sphere_mesh(512, 256)
    north = stereographic_forward(vertices[1:-1], pole="north")
    south = stereographic_forward(vertices[1:-1], pole="south")
    # Exclude the numerically ill-conditioned tails while retaining a large
    # overlap region on a realistic 130k-vertex sphere.
    overlap = (np.abs(north) > 0.08) & (np.abs(north) < 12.5)
    north = north[overlap]
    south = south[overlap]
    transition = stereographic_transition(north, from_pole="north", to_pole="south")
    north_velocity = north.copy()  # d z / d(log scale) = z
    south_velocity = stereographic_transition_velocity(
        north,
        north_velocity,
        from_pole="north",
        to_pole="south",
    )
    round_trip_velocity = stereographic_transition_velocity(
        south,
        south_velocity,
        from_pole="south",
        to_pole="north",
    )
    result = {
        "longitude_samples": 512,
        "latitude_bands": 256,
        "sphere_vertices": int(len(vertices)),
        "overlap_vertices": int(np.count_nonzero(overlap)),
        "coordinate_transition_max_error": float(np.max(np.abs(transition - south))),
        "south_analytic_velocity_max_error": float(np.max(np.abs(south_velocity + south))),
        "velocity_round_trip_max_error": float(np.max(np.abs(round_trip_velocity - north_velocity))),
        "min_overlap_radius": float(np.min(np.abs(north))),
        "max_overlap_radius": float(np.max(np.abs(north))),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output = Path("D:/QC_optimization/artifacts/sphere_chart_velocity_gluing_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "sphere_chart_velocity_gluing_audit.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
