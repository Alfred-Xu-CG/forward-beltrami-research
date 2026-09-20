"""Realistic-resolution boundary-adapted conductivity rectangle audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.rectangle_conductivity import rectangle_beltrami_conductivity
from qcopt.forward.rectangle_fd import rectangle_beltrami_fd


def _manufactured(n: int, amplitude: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    truth = xx + amplitude * np.sin(2.0 * np.pi * yy) + 1j * yy
    beta = amplitude * np.pi * np.cos(2.0 * np.pi * yy)
    mu = 1j * beta / (1.0 - 1j * beta)
    return mu.astype(np.complex128), truth.astype(np.complex128)


def run(output_dir: Path, sizes: tuple[int, ...] = (65, 129, 257)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n in sizes:
        mu, truth = _manufactured(n)
        t0 = time.perf_counter()
        conductivity = rectangle_beltrami_conductivity(mu, truth)
        conductivity_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        collocated = rectangle_beltrami_fd(mu, truth, method="direct")
        collocated_seconds = time.perf_counter() - t1
        collocated_finite = bool(np.all(np.isfinite(collocated.map)))
        records.append(
            {
                "n_nodes_per_axis": n,
                "cells": (n - 1) ** 2,
                "faces": 2 * (n - 1) ** 2,
                "conductivity_seconds": conductivity_seconds,
                "conductivity_rms_error": float(np.sqrt(np.mean(np.abs(conductivity.map - truth) ** 2))),
                "conductivity_max_error": float(np.max(np.abs(conductivity.map - truth))),
                "conductivity_equation_residual": conductivity.equation_residual,
                "conductivity_boundary_tangential_mismatch": conductivity.boundary_imag_mismatch,
                "conductivity_min_triangle_determinant": conductivity.min_triangle_determinant,
                "collocated_fd_seconds": collocated_seconds,
                "collocated_fd_finite": collocated_finite,
                "collocated_fd_matrix_singular": not collocated_finite,
                "collocated_fd_max_error": (
                    float(np.max(np.abs(collocated.map - truth))) if collocated_finite else None
                ),
                "collocated_fd_equation_residual": (
                    collocated.max_equation_residual if collocated_finite else None
                ),
            }
        )
    result = {
        "records": records,
        "scope": "nonperiodic boundary-adapted conductivity FEM versus collocated central-difference rectangle BVP",
        "manufactured_map": "f(x,y)=x+0.08 sin(2 pi y)+i y",
        "limitation": "Dirichlet reference/control; imaginary recovery is a weak projection and does not provide an arbitrary-mu boundary-free decoder",
    }
    (output_dir / "rectangle_conductivity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[65, 129, 257])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes)), indent=2))


if __name__ == "__main__":
    main()
