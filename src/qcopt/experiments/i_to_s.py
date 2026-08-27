"""Large-distortion analytic I-to-S registration benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..autograd import lbs_from_raw, lsqc_from_raw
from ..beltrami import face_beltrami, qc_dilation
from ..constraints import rectangle_sliding_constraints, two_pin_constraints
from ..injectivity import audit_injectivity
from ..mesh import structured_rectangle
from ..optimize import OptimizationResult, run_direct_optimization, run_mu_optimization
from ..registration import i_field, registration_loss, s_field, soft_dice


def run_i_to_s(
    output_directory: str | Path,
    *,
    nx: int = 20,
    ny: int = 20,
    iterations: int = 100,
    seed: int = 20260828,
    make_figure: bool = True,
) -> list[dict[str, object]]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mesh = structured_rectangle(nx, ny)
    source_points = torch.tensor(
        np.array(mesh.vertices, copy=True), dtype=torch.double
    )
    source_values = i_field(source_points).detach()

    def data_loss(uv: torch.Tensor) -> torch.Tensor:
        return registration_loss(source_values, s_field(uv))

    results: dict[str, OptimizationResult] = {}
    raw_initial = torch.zeros(mesh.n_faces, 2, dtype=torch.double)
    fixed_constraints = rectangle_sliding_constraints(mesh)
    results["mu_lbs_fixed"] = run_mu_optimization(
        mesh,
        raw_initial,
        lambda raw: lbs_from_raw(raw, mesh, fixed_constraints, k_max=0.92),
        data_loss,
        iterations=iterations,
        learning_rate=0.08,
        rectangle=True,
    )
    free_constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    results["mu_lsqc_free"] = run_mu_optimization(
        mesh,
        raw_initial,
        lambda raw: lsqc_from_raw(
            raw, mesh, free_constraints, k_max=0.92, weighted=False
        ),
        data_loss,
        iterations=iterations,
        learning_rate=0.08,
        rectangle=False,
    )
    for method in ("none", "lim_style", "slim_style", "amips_style"):
        results[f"direct_{method}"] = run_direct_optimization(
            mesh,
            mesh.vertices,
            data_loss,
            method=method,
            iterations=iterations,
            learning_rate=0.04,
            barrier_weight=0.005,
            rectangle=True,
        )

    metrics = [
        _method_metrics(mesh, name, result, source_values)
        for name, result in results.items()
    ]
    payload = {
        "configuration": {
            "nx": nx,
            "ny": ny,
            "iterations": iterations,
            "seed": seed,
            "source": "analytic_thick_I",
            "target": "analytic_thick_S",
        },
        "methods": metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_trajectories(output / "trajectories.csv", results)
    np.savez_compressed(
        output / "maps.npz",
        vertices=mesh.vertices,
        faces=mesh.faces,
        **{name: result.uv for name, result in results.items()},
    )
    if make_figure:
        _plot_comparison(output / "comparison.png", mesh, results, metrics)
    return metrics


def _method_metrics(mesh, method, result, source_values):
    rectangle = method != "mu_lsqc_free"
    report = audit_injectivity(mesh, result.uv, rectangle=rectangle)
    uv_tensor = torch.tensor(result.uv, dtype=torch.double)
    target_values = s_field(uv_tensor)
    try:
        measured_mu = face_beltrami(mesh, result.uv)
        maximum_dilation = float(np.max(qc_dilation(measured_mu)))
    except ValueError:
        maximum_dilation = float("inf")
    return {
        "method": method,
        "data_loss": float(registration_loss(source_values, target_values)),
        "soft_dice": float(soft_dice(source_values, target_values)),
        "flipped_faces": len(report.flipped_faces),
        "boundary_intersections": len(report.boundary_intersections),
        "bad_branch_vertices": len(report.bad_branch_vertices),
        "minimum_signed_area_ratio": report.minimum_signed_area_ratio,
        "max_qc_dilation": maximum_dilation,
        "certified": report.certified,
        "elapsed_seconds": result.elapsed_seconds,
        "iterations": len(result.history),
        "rejected_steps": sum(not bool(item["accepted"]) for item in result.history),
        "total_backtracks": sum(int(item["backtracks"]) for item in result.history),
    }


def _write_trajectories(path: Path, results: dict[str, OptimizationResult]) -> None:
    fieldnames = [
        "method",
        "iteration",
        "loss",
        "objective",
        "accepted",
        "backtracks",
        "certified",
        "minimum_signed_area_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for method, result in results.items():
            for item in result.history:
                writer.writerow({"method": method, **item})


def _plot_comparison(path: Path, mesh, results, metrics) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    metrics_by_method = {item["method"]: item for item in metrics}
    for axis, (method, result) in zip(axes.flat, results.items(), strict=True):
        axis.triplot(
            result.uv[:, 0], result.uv[:, 1], mesh.faces, color="#406080", linewidth=0.35
        )
        item = metrics_by_method[method]
        axis.set_title(
            f"{method}\nDice={item['soft_dice']:.3f}, flips={item['flipped_faces']}"
        )
        axis.set_aspect("equal")
        axis.set_xlim(-0.2, 1.2)
        axis.set_ylim(-0.2, 1.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    run_i_to_s(
        arguments.output,
        nx=arguments.nx,
        ny=arguments.ny,
        iterations=arguments.iterations,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
