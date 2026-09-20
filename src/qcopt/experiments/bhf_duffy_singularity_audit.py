"""Mesh-scale local Duffy quadrature audit for a BHF vertex singularity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import duffy_triangle_kernel_integral
from qcopt.mesh import structured_rectangle


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(64, 64)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    faces = mesh.faces
    tri = source[faces]
    b = 0.21 + 0.09j
    a = 1.0 - b
    image_tri = a * tri + b * np.conjugate(tri)
    nu = 0.12 * np.exp(-np.abs(np.mean(tri, axis=1) - 0.42 - 0.37j) ** 2 / 0.15)
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    target_vertex = 32 * 65 + 32
    target = a * source[target_vertex] + b * np.conjugate(source[target_vertex])
    records = []
    for order in (8, 16):
        t0 = time.perf_counter()
        total = 0.0 + 0.0j
        for index, face in enumerate(faces):
            local = np.flatnonzero(face == target_vertex)
            vertex = int(local[0]) if local.size else 0
            total += nu[index] * fz[index] ** 2 * duffy_triangle_kernel_integral(
                tri[index], image_tri[index], target, vertex=vertex, order=order
            )
        records.append({
            "order": order,
            "elapsed_seconds": time.perf_counter() - t0,
            "velocity_real": float((-total / np.pi).real),
            "velocity_imag": float((-total / np.pi).imag),
        })
    difference = abs(complex(records[-1]["velocity_real"], records[-1]["velocity_imag"]) - complex(records[-2]["velocity_real"], records[-2]["velocity_imag"]))
    result = {
        "mesh": "64x64 cells / 8192 faces",
        "records": records,
        "order16_minus_order8_abs": float(difference),
        "scope": "Duffy treatment of the simple BHF kernel pole at a mesh vertex",
        "limitation": "local near-field primitive; a production solver still needs a global near/far-field assembly and chart gluing",
    }
    (output_dir / "bhf_duffy_singularity_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
