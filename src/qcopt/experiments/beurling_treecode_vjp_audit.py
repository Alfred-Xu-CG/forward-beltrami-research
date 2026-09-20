"""High-resolution VJP audit for the non-periodic scattered Beurling treecode."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_treecode import treecode_beurling_apply, treecode_beurling_values_vjp


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(20260919)
    count = 4096
    source = 0.05 + 0.42 * rng.random(count) + 1j * (0.05 + 0.9 * rng.random(count))
    target = 0.55 + 0.4 * rng.random(count) + 1j * (0.05 + 0.9 * rng.random(count))
    values = rng.normal(size=count) + 1j * rng.normal(size=count)
    weights = (0.4 + 0.8 * rng.random(count)) / count
    cotangent = rng.normal(size=count) + 1j * rng.normal(size=count)
    theta, order = 0.22, 12
    forward_started = time.perf_counter()
    tree_forward = treecode_beurling_apply(source, values, weights, target_points=target, theta=theta, order=order)
    forward_seconds = time.perf_counter() - forward_started
    direct = direct_beurling_apply(source, values, weights, target_points=target, block_size=512)
    direct_error = float(np.linalg.norm(tree_forward - direct) / np.linalg.norm(direct))
    vjp_started = time.perf_counter()
    gradient = treecode_beurling_values_vjp(source, values, weights, cotangent, target_points=target, theta=theta, order=order)
    vjp_seconds = time.perf_counter() - vjp_started
    direction = rng.normal(size=count) + 1j * rng.normal(size=count)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = treecode_beurling_apply(source, values + eps * direction, weights, target_points=target, theta=theta, order=order)
    minus = treecode_beurling_apply(source, values - eps * direction, weights, target_points=target, theta=theta, order=order)
    finite_directional = float(np.real(np.vdot(cotangent, (plus - minus) / (2.0 * eps))))
    predicted_directional = float(np.real(np.vdot(gradient, direction)))
    result = {
        "points": count,
        "theta": theta,
        "order": order,
        "forward_seconds": forward_seconds,
        "vjp_seconds": vjp_seconds,
        "direct_forward_relative_error": direct_error,
        "finite_directional": finite_directional,
        "predicted_directional": predicted_directional,
        "vjp_directional_abs_error": abs(finite_directional - predicted_directional),
        "scope": "nonperiodic scattered treecode forward and source-value VJP",
        "limitation": "treecode approximation, not a production FMM or singular quadrature theorem",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path("D:/QC_optimization/artifacts/beurling_treecode_vjp_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "beurling_treecode_vjp_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
