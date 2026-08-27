"""Transition-aware two-chart cylinder registration experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..beltrami import face_beltrami, qc_dilation
from ..constraints import fixed_vertex_constraints
from ..injectivity import audit_injectivity
from ..mesh import TriMesh, structured_rectangle
from ..multichart import (
    AffineTransition,
    Atlas,
    Chart,
    Overlap,
    build_affine_compatibility,
    raw_coordinate_equality_constraints,
)
from ..multichart_solver import solve_coupled_lsqc


def run_multichart_registration(
    output_directory: str | Path,
    *,
    nx: int = 12,
    ny: int = 6,
    make_figure: bool = True,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    atlas = _cylinder_atlas(nx, ny)
    target_maps = _target_maps(atlas)
    mus = {
        name: face_beltrami(atlas.chart(name).mesh, target)
        for name, target in target_maps.items()
    }
    landmarks = np.array([0, nx // 2, (ny // 2) * (nx + 1) + nx // 2, (ny + 1) * (nx + 1) - 1])
    local_landmarks = fixed_vertex_constraints(
        atlas.chart("c0").mesh.n_vertices,
        landmarks,
        target_maps["c0"][landmarks],
    )
    embedded_landmarks = atlas.embed_chart_constraints("c0", local_landmarks)
    correct_constraints = embedded_landmarks.stack(build_affine_compatibility(atlas))
    correct = solve_coupled_lsqc(atlas, mus, correct_constraints)

    wrong_constraints = embedded_landmarks.stack(raw_coordinate_equality_constraints(atlas))
    wrong = solve_coupled_lsqc(atlas, mus, wrong_constraints)
    physical_compatibility = build_affine_compatibility(atlas)
    wrong_vector = atlas.pack_maps(wrong.maps)
    wrong_physical_seam = float(
        np.linalg.norm(
            physical_compatibility.C @ wrong_vector - physical_compatibility.d,
            ord=np.inf,
        )
    )
    landmark_error = correct.maps["c0"][landmarks] - target_maps["c0"][landmarks]
    chart_certificates = {
        name: audit_injectivity(atlas.chart(name).mesh, values, rectangle=True).certified
        for name, values in correct.maps.items()
    }
    max_angular_distortion = max(
        float(np.max(qc_dilation(face_beltrami(atlas.chart(name).mesh, values))))
        for name, values in correct.maps.items()
    )
    metrics: dict[str, object] = {
        "compatibility_equation": "g_d(tau_S(p)) = tau_T(g_c(p))",
        "transition_aware_seam_residual": correct.seam_residual,
        "raw_coordinate_equality_physical_seam_residual": wrong_physical_seam,
        "algebraic_residual": correct.algebraic_residual,
        "constraint_residual": correct.constraint_residual,
        "max_angular_distortion": max_angular_distortion,
        "landmark_rmse": float(np.sqrt(np.mean(landmark_error**2))),
        "chart_certificates": chart_certificates,
        "charts": len(atlas.charts),
        "overlap_pairs": int(sum(len(overlap.indices_c) for overlap in atlas.overlaps)),
        "scope": "matched synthetic cylinder atlas with known affine transitions",
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        output / "maps.npz",
        faces=atlas.chart("c0").mesh.faces,
        source_c0=atlas.chart("c0").mesh.vertices,
        source_c1=atlas.chart("c1").mesh.vertices,
        target_c0=target_maps["c0"],
        target_c1=target_maps["c1"],
        correct_c0=correct.maps["c0"],
        correct_c1=correct.maps["c1"],
        wrong_c0=wrong.maps["c0"],
        wrong_c1=wrong.maps["c1"],
    )
    if make_figure:
        _plot(output / "comparison.png", atlas, correct.maps, wrong.maps)
    return metrics


def _cylinder_atlas(nx: int, ny: int) -> Atlas:
    base = structured_rectangle(nx, ny)
    shifted = TriMesh(base.vertices + np.array([1.0, 0.0]), base.faces)
    chart0, chart1 = Chart("c0", base), Chart("c1", shifted)
    source_transition = AffineTransition("c0", "c1", np.eye(2), np.array([1.0, 0.0]))
    target_transition = AffineTransition("c0", "c1", np.eye(2), np.array([1.0, 0.0]))
    indices = np.arange(base.n_vertices)
    overlap = Overlap(
        "c0", "c1", indices, indices, source_transition, target_transition
    )
    return Atlas((chart0, chart1), (overlap,))


def _target_maps(atlas: Atlas) -> dict[str, np.ndarray]:
    base = atlas.chart("c0").mesh.vertices
    x, y = base[:, 0], base[:, 1]
    target0 = np.column_stack(
        (
            x + 0.08 * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y),
            y + 0.055 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y),
        )
    )
    return {"c0": target0, "c1": target0 + np.array([1.0, 0.0])}


def _plot(path, atlas, correct, wrong):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, name in enumerate(("c0", "c1")):
        faces = atlas.chart(name).mesh.faces
        axes[0, column].triplot(correct[name][:, 0], correct[name][:, 1], faces, linewidth=0.4)
        axes[0, column].set_title(f"transition-aware {name}")
        axes[1, column].triplot(wrong[name][:, 0], wrong[name][:, 1], faces, linewidth=0.4)
        axes[1, column].set_title(f"wrong raw equality {name}")
        for row in range(2):
            axes[row, column].set_aspect("equal")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=6)
    arguments = parser.parse_args()
    run_multichart_registration(arguments.output, nx=arguments.nx, ny=arguments.ny)


if __name__ == "__main__":
    main()
