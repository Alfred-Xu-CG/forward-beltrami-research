"""Learned polynomial GMRES initializer with end-to-end cost accounting."""

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
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return amplitude * np.exp(2j * np.pi * ((seed + 1) * xx - (seed + 2) * yy) + 1j * phase)


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_mu = [_case(n, seed, 0.18 + 0.015 * seed) for seed in range(1, 7)]
    features = []
    targets = []
    for mu in train_mu:
        bmu = periodic_beurling_apply(mu)
        features.append(np.column_stack((mu.ravel(), (mu * bmu).ravel())))
        targets.append(periodic_beltrami_gmres(mu, rtol=1e-9).h.ravel())
    matrix = np.vstack(features)
    target = np.concatenate(targets)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    test_mu = _case(n, 21, 0.43)
    t0 = time.perf_counter()
    cold = periodic_beltrami_gmres(test_mu, rtol=1e-9)
    cold_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    guess = periodic_polynomial_initial_guess(test_mu, (coefficients[0], coefficients[1]))
    seeded = periodic_beltrami_gmres(test_mu, rtol=1e-9, initial_guess=guess)
    seeded_time = time.perf_counter() - t1
    result = {
        "grid": f"{n}x{n}",
        "training_examples": len(train_mu),
        "learned_coefficients": [[float(c.real), float(c.imag)] for c in coefficients],
        "cold": {"iterations": cold.iterations, "seconds": cold_time, "residual": cold.operator_residual},
        "polynomial_seeded": {"iterations": seeded.iterations, "seconds_including_seed": seeded_time, "residual": seeded.operator_residual},
        "iteration_reduction": cold.iterations - seeded.iterations,
        "root_difference": float(np.max(np.abs(cold.h - seeded.h))),
        "net_wall_speedup": float(cold_time / seeded_time),
        "scope": "two-feature learned polynomial initializer with residual-certified GMRES root",
        "limitation": "training solve cost is excluded from inference timing and the method remains a torus initializer, not a learned preconditioner",
    }
    (output_dir / "learned_polynomial_initializer_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()

