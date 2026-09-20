"""Heterogeneous random-field stress test for the matrix-free Neumann preconditioner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_gmres import periodic_beltrami_gmres


def _smooth_random_field(n: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Make a bounded complex field from a small random Fourier dictionary."""
    x = np.arange(n, dtype=np.float64)[None, :] / n
    y = np.arange(n, dtype=np.float64)[:, None] / n
    field = np.zeros((n, n), dtype=np.complex128)
    modes = ((1, 0), (0, 1), (1, 1), (2, -1), (2, 2), (3, 1), (1, -3))
    for kx, ky in modes:
        coefficient = rng.normal() + 1j * rng.normal()
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field += coefficient * np.exp(2j * np.pi * (kx * x + ky * y) + 1j * phase)
    scale = np.max(np.abs(field))
    return np.ascontiguousarray(amplitude * field / max(scale, np.finfo(float).eps))


def run(output_dir: Path, n: int = 256, cases: int = 6) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    records = []
    for index in range(cases):
        amplitude = float(rng.uniform(0.20, 0.72))
        mu = _smooth_random_field(n, amplitude, rng)
        cold_started = time.perf_counter()
        cold = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=160)
        cold_seconds = time.perf_counter() - cold_started
        case = {
            "case": index,
            "amplitude": amplitude,
            "cold_iterations": cold.iterations,
            "cold_seconds": cold_seconds,
            "cold_residual": cold.operator_residual,
        }
        for order in (1, 2, 3):
            started = time.perf_counter()
            result = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=160, preconditioner_order=order)
            elapsed = time.perf_counter() - started
            case[f"order{order}"] = {
                "iterations": result.iterations,
                "elapsed_seconds": elapsed,
                "speedup": cold_seconds / elapsed,
                "iteration_reduction": cold.iterations - result.iterations,
                "residual": result.operator_residual,
                "root_difference": float(np.max(np.abs(result.h - cold.h))),
                "converged": bool(result.converged),
            }
        records.append(case)
    speedups = {str(order): [r[f"order{order}"]["speedup"] for r in records] for order in (1, 2, 3)}
    result = {
        "grid": f"{n}x{n}",
        "cases": cases,
        "records": records,
        "speedup_summary": {
            f"order{order}_median": float(np.median(speedups[str(order)])) for order in (1, 2, 3)
        },
        "scope": "cost-inclusive true matrix-free Neumann preconditioner on heterogeneous random smooth fields",
        "limitation": "not a learned preconditioner and not a guarantee for rough or near-degenerate coefficients",
        "elapsed_seconds": float(sum(r["cold_seconds"] + sum(r[f"order{o}"]["elapsed_seconds"] for o in (1, 2, 3)) for r in records)),
    }
    (output_dir / "neumann_preconditioner_random_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--cases", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.cases), indent=2))


if __name__ == "__main__":
    main()
