"""Realistic rectangle-side boundary and holonomy reconstruction audit.

The manufactured map keeps each boundary side on its corresponding side of
the unit rectangle while allowing a non-identity tangential parameterization.
Its facewise Beltrami field is reconstructed through the dual-holonomy/primal
edge procedure, and the boundary winding is sampled independently.  This is
numerical evidence for the PL degree argument, not an exact-predicate proof.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle

from qcopt.experiments.holonomy_reconstruction_audit import (
    _align_error,
    _integrate_edges,
    _propagate_face_scales,
)


def side_sliding_map(vertices: np.ndarray) -> np.ndarray:
    """Construct a smooth positive map whose boundary stays on square sides."""

    x = vertices[:, 0]
    y = vertices[:, 1]
    # Bottom/top tangential motion; the normal coordinate remains y.
    u = x + np.sin(np.pi * x) * (0.08 * (1.0 - y) - 0.05 * y)
    # Left/right tangential motion; the normal coordinate remains x.
    v = y + np.sin(np.pi * y) * (0.07 * (1.0 - x) - 0.04 * x)
    # Interior-only coupling vanishes on every boundary side.
    u += 0.025 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(2.0 * np.pi * y)
    v += 0.020 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(2.0 * np.pi * x)
    return np.column_stack((u, v))


def _polygon_winding(polygon: np.ndarray, point: np.ndarray) -> float:
    shifted = polygon - point[None, :]
    cross = shifted[:, 0] * np.roll(shifted[:, 1], -1) - shifted[:, 1] * np.roll(shifted[:, 0], -1)
    dot = np.sum(shifted * np.roll(shifted, -1, axis=0), axis=1)
    return float(np.sum(np.arctan2(cross, dot)) / (2.0 * np.pi))


def run(output_dir: Path, n: int = 512) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh = structured_rectangle(n, n)
    target = side_sliding_map(mesh.vertices)
    mu = face_beltrami(mesh, target)
    scales, holonomy, edges = _propagate_face_scales(mesh.vertices, mesh.faces, mu)
    recovered, closure = _integrate_edges(mesh.vertices, mesh.faces, mu, scales, edges)
    recovered_xy = np.column_stack((recovered.real, recovered.imag))
    report = audit_injectivity(mesh, target, rectangle=True)
    loop = mesh.boundary_loops[0]
    polygon = target[loop]
    sample_points = np.asarray(
        [[0.20, 0.23], [0.50, 0.50], [0.77, 0.61], [0.35, 0.74]], dtype=np.float64
    )
    winding = np.asarray([_polygon_winding(polygon, point) for point in sample_points])
    # A complex similarity is the only gauge left by face-scale propagation.
    aligned_error = _align_error(recovered, target)
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "boundary_vertices": int(len(loop)),
        "max_abs_mu": float(np.max(np.abs(mu))),
        "dual_cycle_count": int(len(holonomy)),
        "dual_holonomy_max_relative": float(np.max(holonomy)) if len(holonomy) else 0.0,
        "primal_cycle_count": int(len(closure)),
        "primal_edge_closure_max": float(np.max(closure)) if len(closure) else 0.0,
        "reconstruction_aligned_relative_l2": aligned_error,
        "boundary_side_sliding": True,
        "boundary_winding_samples": winding.tolist(),
        "boundary_winding_all_one": bool(np.allclose(winding, 1.0, atol=1e-10)),
        "injectivity_certified": bool(report.certified),
        "boundary_orientation_ok": bool(report.boundary_orientation_ok),
        "rectangle_side_violations": list(report.rectangle_side_violations),
        "flipped_faces": int(len(report.flipped_faces)),
        "minimum_signed_area_ratio": float(report.minimum_signed_area_ratio),
        "finite": bool(np.all(np.isfinite(recovered_xy))),
        "scope": "R2 rectangle-side boundary homeomorphism plus dual-holonomy reconstruction",
        "interpretation": "Positive P1 faces, simple ordered rectangle boundary, and sampled winding 1 are consistent with the disk degree argument; this remains floating-point evidence rather than an exact global theorem.",
        "limitation": "The audit does not prove arbitrary target fields admit an R2 projection or uniqueness of the nonlinear projection.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "compatibility_boundary_theorem_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
