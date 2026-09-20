"""Realistic-mesh BHF near/far assembly audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import duffy_triangle_kernel_integral, vertex_near_far_bhf_variation
from qcopt.mesh import structured_rectangle


def _data(n: int):
    mesh = structured_rectangle(n, n)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    b = 0.21 + 0.09j
    a = 1.0 - b
    image = a * source + b * np.conjugate(source)
    image_triangles = image[mesh.faces]
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    variation = 0.12 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.42 - 0.37j) ** 2 / 0.15)
    return mesh, source, triangles, image, image_triangles, fz, variation


def _all_duffy(triangles, image_triangles, image, faces, fz, variation, target_vertex, order):
    target = image[target_vertex]
    total = 0.0 + 0.0j
    for index, face in enumerate(faces):
        local = np.flatnonzero(face == target_vertex)
        vertex = int(local[0]) if local.size else 0
        total += variation[index] * fz[index] ** 2 * duffy_triangle_kernel_integral(
            triangles[index], image_triangles[index], target, vertex=vertex, order=order
        )
    return -total / np.pi


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n in (64, 128):
        mesh, source, triangles, image, image_triangles, fz, variation = _data(n)
        stride = max(1, n // 32)
        targets = [j * (n + 1) + i for j in range(stride, n, stride) for i in range(stride, n, stride)]
        targets = targets[:16]
        t0 = time.perf_counter()
        values8 = np.asarray([
            vertex_near_far_bhf_variation(source, image, mesh.faces, fz, variation, v, near_order=8)
            for v in targets
        ])
        seconds8 = time.perf_counter() - t0
        t1 = time.perf_counter()
        values16 = np.asarray([
            vertex_near_far_bhf_variation(source, image, mesh.faces, fz, variation, v, near_order=16)
            for v in targets
        ])
        seconds16 = time.perf_counter() - t1
        record = {
            "grid": f"{n}x{n} cells",
            "faces": int(mesh.n_faces),
            "targets": len(targets),
            "order8_seconds": seconds8,
            "order16_seconds": seconds16,
            "max_order16_minus_order8": float(np.max(np.abs(values16 - values8))),
            "finite": bool(np.all(np.isfinite(values8)) and np.all(np.isfinite(values16))),
        }
        if n == 64:
            t2 = time.perf_counter()
            full = _all_duffy(triangles, image_triangles, image, mesh.faces, fz, variation, targets[0], order=16)
            record["all_duffy_single_target_seconds"] = time.perf_counter() - t2
            record["near_far_vs_all_duffy_abs"] = float(abs(values16[0] - full))
        records.append(record)
    result = {
        "records": records,
        "scope": "Duffy-corrected incident faces plus degree-five far-field assembly at interior mesh vertices",
        "limitation": "only vertex targets are covered; a full production solver still needs arbitrary target near-field rules, adaptive nonlinear integration, and atlas gluing",
    }
    (output_dir / "bhf_near_far_assembly_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
