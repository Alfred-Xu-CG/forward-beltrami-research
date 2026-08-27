"""Prescribed face-area deformation benchmark."""

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

from ..autograd import lbs_from_raw
from ..beltrami import face_beltrami, face_jacobians, qc_dilation
from ..constraints import rectangle_sliding_constraints
from ..density import (
    AreaTarget,
    area_loss,
    manufactured_area_target,
    stress_area_target,
    torch_area_ratios,
)
from ..energies import symmetric_dirichlet_energy, torch_face_jacobians
from ..injectivity import audit_injectivity
from ..mesh import structured_rectangle
from ..optimize import OptimizationResult, run_direct_optimization, run_mu_optimization


def run_density_equalizing(
    output_directory: str | Path,
    *,
    nx: int = 16,
    ny: int = 16,
    iterations: int = 100,
    seed: int = 20260828,
    mu_regularizer_weight: float = 0.002,
    target_names: tuple[str, ...] = ("manufactured", "checkerboard", "spike", "random"),
    make_figure: bool = True,
) -> list[dict[str, object]]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mesh = structured_rectangle(nx, ny)
    targets = [_make_target(mesh, name, seed) for name in target_names]
    constraints = rectangle_sliding_constraints(mesh)
    raw_initial = torch.zeros(mesh.n_faces, 2, dtype=torch.double)
    all_results: dict[tuple[str, str], OptimizationResult] = {}
    metrics: list[dict[str, object]] = []
    for target in targets:
        target_tensor = torch.tensor(target.factors, dtype=torch.double)

        def loss_function(uv: torch.Tensor, factors=target_tensor) -> torch.Tensor:
            return area_loss(torch_area_ratios(mesh, uv), factors)

        def mu_loss_function(uv: torch.Tensor, factors=target_tensor) -> torch.Tensor:
            return area_loss(
                torch_area_ratios(mesh, uv), factors
            ) + mu_regularizer_weight * symmetric_dirichlet_energy(
                torch_face_jacobians(mesh, uv)
            )

        methods: dict[str, OptimizationResult] = {}
        methods["mu_lbs_fixed"] = run_mu_optimization(
            mesh,
            raw_initial,
            lambda raw: lbs_from_raw(raw, mesh, constraints, k_max=0.92),
            mu_loss_function,
            iterations=iterations,
            learning_rate=0.06,
            rectangle=True,
        )
        for method in ("none", "lim_style", "slim_style", "amips_style"):
            methods[f"direct_{method}"] = run_direct_optimization(
                mesh,
                mesh.vertices,
                loss_function,
                method=method,
                iterations=iterations,
                learning_rate=0.04,
                barrier_weight=0.005,
                rectangle=True,
            )
        for method, result in methods.items():
            all_results[(target.name, method)] = result
            metrics.append(_metrics(mesh, target, method, result))

    payload = {
        "configuration": {
            "nx": nx,
            "ny": ny,
            "iterations": iterations,
            "seed": seed,
            "targets": list(target_names),
            "mu_map_regularizer": "symmetric_dirichlet",
            "mu_map_regularizer_weight": mu_regularizer_weight,
        },
        "results": metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_trajectories(output / "trajectories.csv", all_results)
    arrays = {
        f"{target}__{method}": result.uv
        for (target, method), result in all_results.items()
    }
    arrays.update({f"target_factors__{target.name}": target.factors for target in targets})
    np.savez_compressed(
        output / "maps.npz", vertices=mesh.vertices, faces=mesh.faces, **arrays
    )
    if make_figure:
        _plot(output / "comparison.png", mesh, targets, all_results)
    return metrics


def _make_target(mesh, name, seed) -> AreaTarget:
    if name == "manufactured":
        return manufactured_area_target(mesh)
    return stress_area_target(mesh, name, seed=seed)


def _metrics(mesh, target, method, result):
    ratios = np.linalg.det(face_jacobians(mesh, result.uv))
    positive = np.all(ratios > 0.0)
    log_rmse = (
        float(np.sqrt(np.mean((np.log(ratios) - np.log(target.factors)) ** 2)))
        if positive
        else float("inf")
    )
    relative_rmse = float(
        np.sqrt(np.mean(((ratios - target.factors) / target.factors) ** 2))
    )
    report = audit_injectivity(mesh, result.uv, rectangle=True)
    try:
        maximum_dilation = float(np.max(qc_dilation(face_beltrami(mesh, result.uv))))
    except ValueError:
        maximum_dilation = float("inf")
    return {
        "target": target.name,
        "feasibility_status": target.feasibility_status,
        "method": method,
        "log_area_rmse": log_rmse,
        "relative_area_rmse": relative_rmse,
        "weighted_area_sum": float(np.sum(mesh.areas * ratios)),
        "target_weighted_area_sum": float(np.sum(mesh.areas * target.factors)),
        "flipped_faces": len(report.flipped_faces),
        "minimum_signed_area_ratio": report.minimum_signed_area_ratio,
        "max_qc_dilation": maximum_dilation,
        "certified": report.certified,
        "elapsed_seconds": result.elapsed_seconds,
        "iterations": len(result.history),
        "rejected_steps": sum(not bool(item["accepted"]) for item in result.history),
    }


def _write_trajectories(path, results):
    fields = [
        "target",
        "method",
        "iteration",
        "loss",
        "objective",
        "accepted",
        "backtracks",
        "certified",
        "minimum_signed_area_ratio",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for (target, method), result in results.items():
            for item in result.history:
                writer.writerow({"target": target, "method": method, **item})


def _plot(path, mesh, targets, results):
    methods = [
        "mu_lbs_fixed",
        "direct_none",
        "direct_lim_style",
        "direct_slim_style",
        "direct_amips_style",
    ]
    figure, axes = plt.subplots(
        len(targets), len(methods), figsize=(15, 3 * len(targets)), squeeze=False
    )
    for row, target in enumerate(targets):
        for column, method in enumerate(methods):
            axis = axes[row, column]
            uv = results[(target.name, method)].uv
            axis.triplot(uv[:, 0], uv[:, 1], mesh.faces, linewidth=0.3)
            axis.set_title(f"{target.name}: {method}", fontsize=8)
            axis.set_aspect("equal")
            axis.set_xlim(-0.15, 1.15)
            axis.set_ylim(-0.15, 1.15)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    run_density_equalizing(
        arguments.output,
        nx=arguments.nx,
        ny=arguments.ny,
        iterations=arguments.iterations,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
