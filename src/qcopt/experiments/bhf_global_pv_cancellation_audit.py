"""Global incident-face PV cancellation/convergence audit for BHF targets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import edge_point_near_far_bhf_variation, vertex_near_far_bhf_variation
from qcopt.mesh import structured_rectangle


def _field(n: int):
    mesh = structured_rectangle(n, n)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    coefficient = 0.23 + 0.08j
    image = (1.0 - coefficient) * source + coefficient * np.conjugate(source)
    fz = np.full(mesh.n_faces, 1.0 - coefficient, dtype=np.complex128)
    centers = np.mean(triangles, axis=1)
    variation = 0.11 * np.exp(-np.abs(centers - 0.48 - 0.43j) ** 2 / 0.13)
    variation *= 1.0 + 0.35 * np.cos(7.0 * centers.real - 4.0 * centers.imag)
    i = n // 2
    j = n // 2
    vertex = j * (n + 1) + i
    edge = (vertex, (j + 1) * (n + 1) + i + 1)
    return mesh, source, image, fz, variation, vertex, edge


def _target_records(kind: str, n: int, source, image, faces, fz, variation, target, orders):
    records = []
    for order in orders:
        t0 = time.perf_counter()
        if kind == "vertex":
            value = vertex_near_far_bhf_variation(
                source, image, faces, fz, variation, int(target), near_order=order
            )
        else:
            value = edge_point_near_far_bhf_variation(
                source, image, faces, fz, variation, target, 0.41, near_order=order
            )
        records.append({
            "order": order,
            "velocity_real": float(value.real),
            "velocity_imag": float(value.imag),
            "elapsed_seconds": time.perf_counter() - t0,
        })
    values = [complex(r["velocity_real"], r["velocity_imag"]) for r in records]
    return {
        "target_kind": kind,
        "records": records,
        "order8_to_order16_abs": float(abs(values[1] - values[0])),
        "order16_to_order24_abs": float(abs(values[2] - values[1])),
        "finite": bool(np.all(np.isfinite(np.asarray(values))))
    }


def run(output_dir: Path, sizes: tuple[int, ...] = (64, 128, 256, 512)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n in sizes:
        mesh, source, image, fz, variation, vertex, edge = _field(n)
        records.append({
            "grid": f"{n}x{n} cells",
            "faces": int(mesh.n_faces),
            "incident_vertex_faces": int(np.sum(np.any(mesh.faces == vertex, axis=1))),
            "incident_edge_faces": int(np.sum(np.sum(np.isin(mesh.faces, edge), axis=1) == 2)),
            "vertex": _target_records("vertex", n, source, image, mesh.faces, fz, variation, vertex, (8, 16, 24)),
            "edge": _target_records("edge", n, source, image, mesh.faces, fz, variation, edge, (8, 16, 24)),
        })
    result = {
        "records": records,
        "scope": "incident-face Duffy replacement for vertex and interior-edge BHF targets, with global far-field assembly",
        "interpretation": "order convergence across both sides of an interior edge and all incident vertex faces is a numerical PV-cancellation control, not a proof of the nonlinear BHF principal-value theorem",
        "limitation": "full adaptive global PV quadrature, nonlinear time integration, and atlas coupling remain open",
    }
    (output_dir / "bhf_global_pv_cancellation_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes)), indent=2))


if __name__ == "__main__":
    main()
