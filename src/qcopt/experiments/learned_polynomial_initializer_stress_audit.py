"""Held-out heterogeneous stress audit for the polynomial GMRES initializer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling import periodic_beurling_apply
from qcopt.forward.beurling_gmres import periodic_beltrami_gmres, periodic_polynomial_initial_guess


def _case(n: int, seed: int, amplitude: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    phase1, phase2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
    field = (
        np.exp(2j * np.pi * ((seed % 7 + 1) * xx - (seed % 5 + 2) * yy) + 1j * phase1)
        + 0.6 * np.exp(2j * np.pi * ((seed % 4 + 2) * xx + (seed % 6 + 1) * yy) + 1j * phase2)
    )
    field /= np.max(np.abs(field))
    return amplitude * field


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [_case(n, seed, 0.16 + 0.025 * (seed % 6)) for seed in range(1, 13)]
    features = []
    targets = []
    training_records = []
    for mu in train:
        bmu = periodic_beurling_apply(mu)
        features.append(np.column_stack((mu.ravel(), (mu * bmu).ravel())))
        solved = periodic_beltrami_gmres(mu, rtol=1e-9)
        targets.append(solved.h.ravel())
        training_records.append({"max_abs_mu": float(np.max(np.abs(mu))), "iterations": solved.iterations})
    matrix = np.vstack(features)
    target = np.concatenate(targets)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    held_out = []
    for seed, amplitude in ((101, 0.24), (102, 0.38), (103, 0.52), (104, 0.62)):
        mu = _case(n, seed, amplitude)
        t0 = time.perf_counter()
        cold = periodic_beltrami_gmres(mu, rtol=1e-9)
        cold_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        guess = periodic_polynomial_initial_guess(mu, (coefficients[0], coefficients[1]))
        seeded = periodic_beltrami_gmres(mu, rtol=1e-9, initial_guess=guess)
        seeded_seconds = time.perf_counter() - t1
        held_out.append(
            {
                "seed": seed,
                "amplitude": amplitude,
                "max_abs_mu": float(np.max(np.abs(mu))),
                "cold_iterations": cold.iterations,
                "seeded_iterations": seeded.iterations,
                "iteration_reduction": cold.iterations - seeded.iterations,
                "cold_seconds": cold_seconds,
                "seeded_seconds_including_initializer": seeded_seconds,
                "net_wall_speedup": cold_seconds / seeded_seconds,
                "root_difference": float(np.max(np.abs(cold.h - seeded.h))),
                "seeded_residual": seeded.operator_residual,
            }
        )
    result = {
        "grid": f"{n}x{n}",
        "training_examples": len(train),
        "learned_coefficients": [[float(c.real), float(c.imag)] for c in coefficients],
        "training_records": training_records,
        "held_out": held_out,
        "mean_net_wall_speedup": float(np.mean([r["net_wall_speedup"] for r in held_out])),
        "mean_iteration_reduction": float(np.mean([r["iteration_reduction"] for r in held_out])),
        "scope": "heterogeneous multi-frequency polynomial GMRES initializer stress audit",
        "limitation": "training solve cost is excluded from inference timing; this remains a torus initializer and not a topology certificate",
    }
    (output_dir / "learned_polynomial_initializer_stress_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, n=args.n), indent=2))


if __name__ == "__main__":
    main()
