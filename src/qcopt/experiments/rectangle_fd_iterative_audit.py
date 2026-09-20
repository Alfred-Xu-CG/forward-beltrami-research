"""Realistic-resolution ILU-GMRES versus direct rectangle BVP audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.rectangle_fd import rectangle_beltrami_fd
from qcopt.forward.rectangle_fd_iterative import solve_rectangle_fd_gmres


def _case(n: int, variable: bool):
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    truth = xx + 0.12 * np.sin(np.pi * xx) ** 2 * np.sin(np.pi * yy) + 1j * (
        yy + 0.08 * np.sin(np.pi * xx) * np.sin(np.pi * yy) ** 2
    )
    if not variable:
        coefficient = 0.18 + 0.11j
        truth = xx + 1j * yy + coefficient * (xx - 1j * yy)
        return np.full((n, n), coefficient, dtype=np.complex128), truth
    spacing = 1.0 / (n - 1)
    fx = (truth[:, 2:] - truth[:, :-2]) / (2.0 * spacing)
    fy = (truth[2:, :] - truth[:-2, :]) / (2.0 * spacing)
    fz = 0.5 * (fx[1:-1, :] - 1j * fy[:, 1:-1])
    fbar = 0.5 * (fx[1:-1, :] + 1j * fy[:, 1:-1])
    coefficient = np.zeros((n, n), dtype=np.complex128)
    coefficient[1:-1, 1:-1] = fbar / fz
    return coefficient, truth


def run(output_dir: Path, sizes: tuple[int, ...] = (128, 256, 512)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for variable in (False, True):
        for n in sizes if not variable else sizes[:2]:
            coefficient, boundary = _case(n, variable)
            rng = np.random.default_rng(20260919 + n + int(variable))
            transpose_rhs = np.zeros_like(boundary)
            transpose_rhs[1:-1, 1:-1] = rng.normal(size=(n - 2, n - 2)) + 1j * rng.normal(size=(n - 2, n - 2))
            started = time.perf_counter()
            iterative = solve_rectangle_fd_gmres(
                coefficient, boundary, rtol=1e-10, restart=80, maxiter=1000,
                drop_tol=1e-4, fill_factor=10.0, transpose_rhs=transpose_rhs,
            )
            iterative_elapsed = time.perf_counter() - started
            direct_started = time.perf_counter()
            direct = rectangle_beltrami_fd(coefficient, boundary, method="direct")
            direct_elapsed = time.perf_counter() - direct_started
            difference = iterative.map - direct.map
            records.append({
                "variable": variable,
                "n": n,
                "unknowns": int((n - 2) ** 2),
                "iterative_total_seconds": iterative_elapsed,
                "assembly_seconds": iterative.assembly_seconds,
                "preconditioner_seconds": iterative.preconditioner_seconds,
                "solve_seconds": iterative.solve_seconds,
                "direct_seconds": direct_elapsed,
                "iterations": iterative.iterations,
                "info": iterative.info,
                "equation_residual": iterative.equation_residual,
                "direct_equation_residual": direct.max_equation_residual,
                "solution_relative_difference": float(np.linalg.norm(difference) / max(np.linalg.norm(direct.map), 1e-30)),
                "transpose_iterations": iterative.transpose_iterations,
                "transpose_info": iterative.transpose_info,
                "transpose_seconds": iterative.transpose_seconds,
                "transpose_relative_residual": iterative.transpose_residual,
                "finite": bool(np.all(np.isfinite(iterative.map))),
            })
    result = {
        "records": records,
        "scope": "rectangle Beltrami sparse ILU-GMRES and transpose implicit-layer control",
        "limitation": "still a central-difference Dirichlet BVP; no boundary-free arbitrary-mu decoder or higher-order discretization",
    }
    (output_dir / "rectangle_fd_iterative_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes)), indent=2))


if __name__ == "__main__":
    main()
