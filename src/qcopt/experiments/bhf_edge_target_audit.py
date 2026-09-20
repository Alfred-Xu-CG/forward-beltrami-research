"""Convergence audit for the BHF near/far rule at an interior mesh edge."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import edge_point_near_far_bhf_variation
from qcopt.mesh import structured_rectangle


def _run(n: int) -> dict[str, float | int]:
    mesh = structured_rectangle(n, n)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    coefficient = 0.19 + 0.06j
    image = (1.0 - coefficient) * source + coefficient * np.conjugate(source)
    fz = np.full(mesh.n_faces, 1.0 - coefficient, dtype=np.complex128)
    variation = 0.11 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.48 - 0.43j) ** 2 / 0.11)
    # Interior diagonal of the cell whose lower-left vertex is near the center.
    i = n // 2
    j = n // 2
    edge = (j * (n + 1) + i, (j + 1) * (n + 1) + i + 1)
    records = []
    for order in (8, 16, 24):
        started = time.perf_counter()
        value = edge_point_near_far_bhf_variation(
            source, image, mesh.faces, fz, variation, edge, 0.41, near_order=order
        )
        records.append({"order": order, "real": value.real, "imag": value.imag, "elapsed_seconds": time.perf_counter() - started})
    return {
        "grid_cells_per_axis": n,
        "faces": mesh.n_faces,
        "order8_to_order16_abs": abs(complex(records[1]["real"], records[1]["imag"]) - complex(records[0]["real"], records[0]["imag"])),
        "order16_to_order24_abs": abs(complex(records[2]["real"], records[2]["imag"]) - complex(records[1]["real"], records[1]["imag"])),
        "records": records,
    }


def main() -> None:
    started = time.perf_counter()
    result = {"cases": [_run(64), _run(128)], "elapsed_seconds": 0.0,
              "limitation": "edge-local Duffy control; global multi-edge PV cancellation and scalable nonlinear BHF remain open"}
    result["elapsed_seconds"] = time.perf_counter() - started
    output = Path("D:/QC_optimization/artifacts/bhf_edge_target_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "bhf_edge_target_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
