"""Manufactured reconstruction, adjoint, and sparse scaling validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..adjoint import lbs_mu_vjp, lsqc_mu_vjp
from ..beltrami import face_beltrami
from ..constraints import fixed_vertex_constraints, two_pin_constraints
from ..lbs import solve_lbs
from ..lsqc import solve_lsqc
from ..mesh import structured_rectangle


def run_solver_validation(
    output_directory: str | Path,
    *,
    grid_sizes: tuple[int, ...] = (4, 8, 16, 24),
    seed: int = 20260828,
    make_figure: bool = True,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mesh = structured_rectangle(max(3, min(grid_sizes)), max(3, min(grid_sizes)))
    boundary = mesh.boundary_loops[0]
    matrix = np.array([[1.35, 0.22], [-0.12, 0.76]])
    target = mesh.vertices @ matrix.T + np.array([0.17, -0.23])
    manufactured_mu = face_beltrami(mesh, target)
    fixed = fixed_vertex_constraints(mesh.n_vertices, boundary, target[boundary])
    pins = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        target[[0, mesh.n_vertices - 1]],
    )
    lbs_manufactured = solve_lbs(mesh, manufactured_mu, fixed)
    lsqc_manufactured = solve_lsqc(mesh, manufactured_mu, pins)
    reconstruction_errors = {
        "lbs": float(np.max(np.abs(lbs_manufactured.uv - target))),
        "lsqc_unweighted": float(np.max(np.abs(lsqc_manufactured.uv - target))),
    }

    validation_mu = 0.06 * (
        rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces)
    )
    uv_bar = rng.normal(size=(mesh.n_vertices, 2))
    direction = rng.normal(size=(mesh.n_faces, 2))
    direction /= np.linalg.norm(direction)
    epsilons = np.asarray([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    sweeps = []

    lbs_result = solve_lbs(mesh, validation_mu, fixed)
    lbs_adjoint = lbs_mu_vjp(mesh, validation_mu, lbs_result, uv_bar)
    sweeps.append(
        _gradient_sweep(
            "lbs",
            validation_mu,
            direction,
            lbs_adjoint.gradient,
            lambda value: float(np.sum(solve_lbs(mesh, value, fixed).uv * uv_bar)),
            epsilons,
            lbs_adjoint.residual,
        )
    )
    for weighted, name in ((False, "lsqc_unweighted"), (True, "lsqc_weighted")):
        result = solve_lsqc(mesh, validation_mu, pins, weighted=weighted)
        adjoint = lsqc_mu_vjp(
            mesh, validation_mu, result, uv_bar, weighted=weighted
        )
        sweeps.append(
            _gradient_sweep(
                name,
                validation_mu,
                direction,
                adjoint.gradient,
                lambda value, flag=weighted: float(
                    np.sum(solve_lsqc(mesh, value, pins, weighted=flag).uv * uv_bar)
                ),
                epsilons,
                adjoint.residual,
            )
        )

    scaling = [_scaling_case(size) for size in grid_sizes]
    maximum_forward_residual = max(
        [lbs_manufactured.primal_residual, lsqc_manufactured.primal_residual]
        + [item["maximum_forward_residual"] for item in scaling]
    )
    maximum_adjoint_residual = max(float(item["adjoint_residual"]) for item in sweeps)
    maximum_stable_gradient_error = max(
        float(item["stable_window_median_relative_error"]) for item in sweeps
    )
    maximum_reconstruction_error = max(reconstruction_errors.values())
    report: dict[str, object] = {
        "configuration": {"grid_sizes": list(grid_sizes), "seed": seed},
        "reconstruction_errors": reconstruction_errors,
        "gradient_sweeps": sweeps,
        "scaling": scaling,
        "acceptance": {
            "forward_residual_below_1e-10": maximum_forward_residual < 1e-10,
            "adjoint_residual_below_1e-10": maximum_adjoint_residual < 1e-10,
            "stable_gradient_relative_error_below_1e-5": maximum_stable_gradient_error < 1e-5,
            "manufactured_reconstruction_below_1e-8": maximum_reconstruction_error < 1e-8,
        },
        "maxima": {
            "forward_residual": maximum_forward_residual,
            "adjoint_residual": maximum_adjoint_residual,
            "stable_gradient_relative_error": maximum_stable_gradient_error,
            "manufactured_reconstruction_error": maximum_reconstruction_error,
        },
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_scaling(output / "scaling.csv", scaling)
    if make_figure:
        _plot_sweeps(output / "gradient_sweep.png", sweeps)
    return report


def _gradient_sweep(name, mu, direction, gradient, evaluate, epsilons, residual):
    prediction = float(np.sum(gradient * direction))
    errors = []
    finite_differences = []
    complex_direction = direction[:, 0] + 1j * direction[:, 1]
    for epsilon in epsilons:
        finite = (evaluate(mu + epsilon * complex_direction) - evaluate(mu - epsilon * complex_direction)) / (
            2.0 * epsilon
        )
        finite_differences.append(float(finite))
        errors.append(
            abs(prediction - finite) / max(1.0, abs(prediction), abs(finite))
        )
    stable_error = float(np.median(np.asarray(errors)[2:5]))
    return {
        "solver": name,
        "epsilons": epsilons.tolist(),
        "finite_differences": finite_differences,
        "adjoint_prediction": prediction,
        "relative_errors": [float(value) for value in errors],
        "stable_window_median_relative_error": stable_error,
        "adjoint_residual": float(residual),
    }


def _scaling_case(size: int) -> dict[str, object]:
    mesh = structured_rectangle(size, size)
    mu = np.zeros(mesh.n_faces, dtype=np.complex128)
    boundary = mesh.boundary_loops[0]
    fixed = fixed_vertex_constraints(mesh.n_vertices, boundary, mesh.vertices[boundary])
    pins = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    start = perf_counter()
    lbs = solve_lbs(mesh, mu, fixed)
    lbs_seconds = perf_counter() - start
    start = perf_counter()
    lsqc = solve_lsqc(mesh, mu, pins)
    lsqc_seconds = perf_counter() - start
    return {
        "grid_size": size,
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "lbs_system_dimension": lbs.state.system.shape[0],
        "lbs_nnz": int(lbs.state.system.nnz),
        "lbs_seconds": lbs_seconds,
        "lsqc_system_dimension": lsqc.state.system.shape[0],
        "lsqc_nnz": int(lsqc.state.system.nnz),
        "lsqc_seconds": lsqc_seconds,
        "maximum_forward_residual": max(lbs.primal_residual, lsqc.primal_residual),
    }


def _write_scaling(path, scaling):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scaling[0].keys()))
        writer.writeheader()
        writer.writerows(scaling)


def _plot_sweeps(path, sweeps):
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for sweep in sweeps:
        axis.loglog(
            sweep["epsilons"], sweep["relative_errors"], marker="o", label=sweep["solver"]
        )
    axis.axhline(1e-5, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("central-difference step")
    axis.set_ylabel("relative directional-gradient error")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_solver_validation(arguments.output)


if __name__ == "__main__":
    main()
