"""Benchmark augmented-reference and hard-pin fast weighted LSQC backends."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence, TypeVar

import numpy as np

from ..adjoint import (
    _lsqc_augmented_mu_contraction,
    _weighted_lsqc_mu_contraction,
    lsqc_fast_mu_vjp,
    lsqc_mu_vjp,
)
from ..constraints import two_pin_constraints
from ..lsqc import (
    assemble_lsqc_operator,
    assemble_weighted_lsqc_hessian,
    solve_lsqc,
    solve_lsqc_fast,
)
from ..mesh import structured_rectangle

T = TypeVar("T")


def _timed(callback: Callable[[], T], repeats: int) -> tuple[float, T]:
    samples: list[float] = []
    value: T | None = None
    for _ in range(repeats):
        start = perf_counter()
        value = callback()
        samples.append(perf_counter() - start)
    assert value is not None
    return float(np.median(samples)), value


def run_lsqc_benchmark(
    output_directory: str | Path,
    *,
    grid_sizes: Sequence[int] = (12, 24, 36, 48),
    repeats: int = 3,
    seed: int = 20260831,
) -> dict[str, object]:
    """Run deterministic weighted LSQC forward/backward scaling measurements."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not grid_sizes or any(int(size) < 1 for size in grid_sizes):
        raise ValueError("grid_sizes must contain positive integers")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases: list[dict[str, int | float]] = []

    for raw_size in grid_sizes:
        size = int(raw_size)
        mesh = structured_rectangle(size, size)
        constraints = two_pin_constraints(
            mesh.n_vertices,
            [0, mesh.n_vertices - 1],
            mesh.vertices[[0, mesh.n_vertices - 1]],
        )
        mu = 0.08 * (
            rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces)
        )
        uv_bar = rng.normal(size=(mesh.n_vertices, 2))

        # Untimed warm-up isolates imports and one-time runtime initialization.
        augmented_result = solve_lsqc(mesh, mu, constraints, weighted=True)
        fast_result = solve_lsqc_fast(mesh, mu, constraints)
        lsqc_mu_vjp(mesh, mu, augmented_result, uv_bar, weighted=True)
        lsqc_fast_mu_vjp(mesh, mu, fast_result, uv_bar)

        augmented_assembly, _ = _timed(
            lambda: assemble_lsqc_operator(mesh, mu, weighted=True), repeats
        )
        fast_assembly, _ = _timed(
            lambda: assemble_weighted_lsqc_hessian(mesh, mu), repeats
        )
        augmented_forward, augmented_result = _timed(
            lambda: solve_lsqc(mesh, mu, constraints, weighted=True), repeats
        )
        fast_forward, fast_result = _timed(
            lambda: solve_lsqc_fast(mesh, mu, constraints), repeats
        )
        augmented_backward, augmented_adjoint = _timed(
            lambda: lsqc_mu_vjp(
                mesh, mu, augmented_result, uv_bar, weighted=True
            ),
            repeats,
        )
        fast_backward, fast_adjoint = _timed(
            lambda: lsqc_fast_mu_vjp(mesh, mu, fast_result, uv_bar), repeats
        )

        augmented_rhs = np.zeros_like(augmented_result.state.solution)
        augmented_rhs[augmented_result.state.coordinate_slice] = uv_bar.T.reshape(-1)
        augmented_adjoint_solve, augmented_adjoint_vector = _timed(
            lambda: augmented_result.state.factor.solve(augmented_rhs, trans="T"),
            repeats,
        )
        augmented_contraction, _ = _timed(
            lambda: _lsqc_augmented_mu_contraction(
                mesh,
                mu,
                augmented_result,
                augmented_adjoint_vector,
                True,
            ),
            repeats,
        )
        full_fast_rhs = uv_bar.T.reshape(-1)
        fast_rhs = full_fast_rhs[fast_result.state.free_indices]
        fast_adjoint_solve, fast_adjoint_vector = _timed(
            lambda: fast_result.state.factor.solve(fast_rhs, trans="T"), repeats
        )
        fast_adjoint_coordinates = np.zeros_like(fast_result.state.coordinates)
        fast_adjoint_coordinates[fast_result.state.free_indices] = fast_adjoint_vector
        fast_contraction, _ = _timed(
            lambda: _weighted_lsqc_mu_contraction(
                mesh,
                mu,
                fast_result.state.coordinates,
                fast_adjoint_coordinates,
            ),
            repeats,
        )

        def run_augmented_step():
            result = solve_lsqc(mesh, mu, constraints, weighted=True)
            return result, lsqc_mu_vjp(
                mesh, mu, result, uv_bar, weighted=True
            )

        def run_fast_step():
            result = solve_lsqc_fast(mesh, mu, constraints)
            return result, lsqc_fast_mu_vjp(mesh, mu, result, uv_bar)

        augmented_training_step, _ = _timed(run_augmented_step, repeats)
        fast_training_step, _ = _timed(run_fast_step, repeats)

        cases.append(
            {
                "grid_size": size,
                "n_vertices": mesh.n_vertices,
                "n_faces": mesh.n_faces,
                "augmented_system_size": int(augmented_result.state.system.shape[0]),
                "fast_system_size": int(fast_result.state.system.shape[0]),
                "augmented_system_nnz": int(augmented_result.state.system.nnz),
                "fast_system_nnz": int(fast_result.state.system.nnz),
                "augmented_factor_nnz": int(
                    augmented_result.state.factor.L.nnz
                    + augmented_result.state.factor.U.nnz
                ),
                "fast_factor_nnz": int(
                    fast_result.state.factor.L.nnz + fast_result.state.factor.U.nnz
                ),
                "augmented_assembly_seconds": augmented_assembly,
                "fast_assembly_seconds": fast_assembly,
                "augmented_forward_seconds": augmented_forward,
                "fast_forward_seconds": fast_forward,
                "augmented_backward_seconds": augmented_backward,
                "fast_backward_seconds": fast_backward,
                "augmented_adjoint_solve_seconds": augmented_adjoint_solve,
                "fast_adjoint_solve_seconds": fast_adjoint_solve,
                "augmented_contraction_seconds": augmented_contraction,
                "fast_contraction_seconds": fast_contraction,
                "augmented_training_step_seconds": augmented_training_step,
                "fast_training_step_seconds": fast_training_step,
                "forward_speedup": augmented_forward / fast_forward,
                "backward_speedup": augmented_backward / fast_backward,
                "training_step_speedup": (
                    augmented_training_step / fast_training_step
                ),
                "max_map_difference": float(
                    np.max(np.abs(augmented_result.uv - fast_result.uv))
                ),
                "max_gradient_difference": float(
                    np.max(
                        np.abs(augmented_adjoint.gradient - fast_adjoint.gradient)
                    )
                ),
                "augmented_forward_residual": augmented_result.primal_residual,
                "fast_forward_residual": fast_result.primal_residual,
                "augmented_adjoint_residual": augmented_adjoint.residual,
                "fast_adjoint_residual": fast_adjoint.residual,
            }
        )

    report: dict[str, object] = {
        "seed": int(seed),
        "repeats": int(repeats),
        "grid_sizes": [int(size) for size in grid_sizes],
        "cases": cases,
    }
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    return report


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[12, 24, 36, 48])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parse_args(arguments)
    run_lsqc_benchmark(
        parsed.output,
        grid_sizes=parsed.grid_sizes,
        repeats=parsed.repeats,
        seed=parsed.seed,
    )


if __name__ == "__main__":
    main()
