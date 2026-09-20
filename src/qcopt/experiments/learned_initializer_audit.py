"""Toy correctness-preserving learned initializer audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_gmres import periodic_beltrami_gmres, periodic_neumann_initial_guess


def run(output_dir: Path, n: int = 128) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    training_mu: list[np.ndarray] = []
    training_h: list[np.ndarray] = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        mu = 0.25 * np.exp(2j * np.pi * ((seed + 1) * xx - (seed + 2) * yy) + 1j * phase)
        training_mu.append(mu)
        training_h.append(periodic_beltrami_gmres(mu, rtol=1e-10).h)
    numerator = sum(np.vdot(mu, h) for mu, h in zip(training_mu, training_h))
    denominator = sum(np.vdot(mu, mu) for mu in training_mu)
    alpha = numerator / denominator
    xt = np.arange(n, dtype=np.float64) / n
    tx, ty = np.meshgrid(xt, xt, indexing="xy")
    test_mu = 0.42 * np.exp(2j * np.pi * (3.0 * tx - 2.0 * ty))
    cold = periodic_beltrami_gmres(test_mu, rtol=1e-10)
    warm = periodic_beltrami_gmres(test_mu, rtol=1e-10, initial_guess=alpha * test_mu)
    neumann_guess = periodic_neumann_initial_guess(test_mu, order=2)
    neumann = periodic_beltrami_gmres(test_mu, rtol=1e-10, initial_guess=neumann_guess)
    result = {
        "grid": n,
        "learned_complex_alpha": [float(alpha.real), float(alpha.imag)],
        "cold_iterations": cold.iterations,
        "warm_iterations": warm.iterations,
        "cold_residual": cold.operator_residual,
        "warm_residual": warm.operator_residual,
        "root_difference": float(np.max(np.abs(cold.h - warm.h))),
        "neumann_order2_iterations": neumann.iterations,
        "neumann_order2_residual": neumann.operator_residual,
        "neumann_root_difference": float(np.max(np.abs(cold.h - neumann.h))),
    }
    (output_dir / "learned_initializer_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
