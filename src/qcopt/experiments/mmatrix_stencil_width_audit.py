"""High-resolution positive-wide-stencil sweep for a rotating Beltrami tensor.

This is a cone-coverage experiment: increasing the integer stencil radius adds
more mesh-realizable directions, so it can distinguish a finite-direction
approximation barrier from a tensor that is intrinsically non-monotone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import batch_positive_directional_conductances, integer_wide_stencil_directions
from qcopt.forward.mmatrix_smooth import smooth_positive_directional_conductances


def _integer_vectors(max_step: int) -> tuple[tuple[int, int], ...]:
    vectors: list[tuple[int, int]] = []
    for p in range(max_step + 1):
        for q in range(-max_step, max_step + 1):
            if p == 0 and q <= 0:
                continue
            if p == 0 and q == 0:
                continue
            if np.gcd(p, abs(q)) != 1:
                continue
            vectors.append((p, q))
    return tuple(vectors)


def _periodic_derivative(field: np.ndarray, axis: int) -> np.ndarray:
    n = field.shape[axis]
    frequencies = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    shape = [1] * field.ndim
    shape[axis] = n
    transformed = np.fft.fft(field, axis=axis)
    return np.fft.ifft(1j * frequencies.reshape(shape) * transformed, axis=axis).real


def _manufactured(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    u = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy) + 0.4 * np.cos(6.0 * np.pi * xx + 0.2) * np.sin(2.0 * np.pi * yy)
    ux = 2.0 * np.pi * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy) - 0.4 * 6.0 * np.pi * np.sin(6.0 * np.pi * xx + 0.2) * np.sin(2.0 * np.pi * yy)
    uy = -2.0 * np.pi * np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy) + 0.4 * 2.0 * np.pi * np.cos(6.0 * np.pi * xx + 0.2) * np.cos(2.0 * np.pi * yy)
    gradient = np.stack((ux, uy), axis=-1)
    rho = 0.30 + 0.08 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    angle = 0.35 + 0.55 * np.sin(2.0 * np.pi * xx) + 0.35 * np.cos(2.0 * np.pi * yy)
    mu = rho * np.exp(1j * angle)
    denominator = 1.0 - rho * rho
    tensor = np.empty(xx.shape + (2, 2), dtype=np.float64)
    tensor[..., 0, 0] = (1.0 - 2.0 * mu.real + rho * rho) / denominator
    tensor[..., 0, 1] = tensor[..., 1, 0] = -2.0 * mu.imag / denominator
    tensor[..., 1, 1] = (1.0 + 2.0 * mu.real + rho * rho) / denominator
    return u, gradient, tensor


def _truth_operator(tensor: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    flux_x = tensor[..., 0, 0] * gradient[..., 0] + tensor[..., 0, 1] * gradient[..., 1]
    flux_y = tensor[..., 1, 0] * gradient[..., 0] + tensor[..., 1, 1] * gradient[..., 1]
    return _periodic_derivative(flux_x, 1) + _periodic_derivative(flux_y, 0)


def _discrete_operator(u: np.ndarray, weights: np.ndarray, vectors: tuple[tuple[int, int], ...]) -> np.ndarray:
    n = u.shape[0]
    h2 = (1.0 / n) ** 2
    output = np.zeros_like(u)
    for index, (p, q) in enumerate(vectors):
        length2 = float(p * p + q * q)
        plus_weights = 0.5 * (weights[..., index] + np.roll(weights[..., index], shift=(-q, -p), axis=(0, 1)))
        minus_weights = 0.5 * (weights[..., index] + np.roll(weights[..., index], shift=(q, p), axis=(0, 1)))
        plus = np.roll(u, shift=(-q, -p), axis=(0, 1))
        minus = np.roll(u, shift=(q, p), axis=(0, 1))
        output += (plus_weights * (plus - u) - minus_weights * (u - minus)) / (h2 * length2)
    return output


def run(output_dir: Path, sizes: tuple[int, ...], max_steps: tuple[int, ...]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for n in sizes:
        u, gradient, tensor = _manufactured(n)
        truth = _truth_operator(tensor, gradient)
        for max_step in max_steps:
            started = time.perf_counter()
            directions = integer_wide_stencil_directions(max_step)
            vectors = _integer_vectors(max_step)
            fit = batch_positive_directional_conductances(tensor, directions)
            discrete = _discrete_operator(u, fit.values, vectors)
            residual = fit.residual
            weight_variation = np.sqrt(
                0.5 * (
                    np.mean(np.diff(fit.values, axis=0) ** 2)
                    + np.mean(np.diff(fit.values, axis=1) ** 2)
                )
            )
            records.append({
                "n": n,
                "max_step": max_step,
                "direction_count": len(vectors),
                "fit_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "fit_residual_p95": float(np.quantile(residual, 0.95)),
                "fit_residual_max": float(np.max(residual)),
                "exact_fit_fraction": float(np.mean(residual < 1e-10)),
                "weight_neighbor_rms": float(weight_variation),
                "weight_neighbor_max": float(
                    max(
                        np.max(np.abs(np.diff(fit.values, axis=0))),
                        np.max(np.abs(np.diff(fit.values, axis=1))),
                    )
                ),
                "operator_l2_error": float(np.sqrt(np.mean((discrete - truth) ** 2))),
                "elapsed_seconds": time.perf_counter() - started,
            })
            if max_step >= 2:
                smooth_started = time.perf_counter()
                smooth_values, smooth_residual = smooth_positive_directional_conductances(
                    tensor, directions, neighbor_weight=1.0, norm_weight=1e-8
                )
                smooth_discrete = _discrete_operator(u, smooth_values, vectors)
                records.append({
                    "n": n,
                    "max_step": max_step,
                    "selection": "raster_neighbor_continuation",
                    "direction_count": len(vectors),
                    "fit_residual_rms": float(np.sqrt(np.mean(smooth_residual * smooth_residual))),
                    "fit_residual_p95": float(np.quantile(smooth_residual, 0.95)),
                    "fit_residual_max": float(np.max(smooth_residual)),
                    "exact_fit_fraction": float(np.mean(smooth_residual < 1e-10)),
                    "weight_neighbor_rms": float(
                        np.sqrt(0.5 * (np.mean(np.diff(smooth_values, axis=0) ** 2) + np.mean(np.diff(smooth_values, axis=1) ** 2)))
                    ),
                    "weight_neighbor_max": float(
                        max(np.max(np.abs(np.diff(smooth_values, axis=0))), np.max(np.abs(np.diff(smooth_values, axis=1))))
                    ),
                    "operator_l2_error": float(np.sqrt(np.mean((smooth_discrete - truth) ** 2))),
                    "elapsed_seconds": time.perf_counter() - smooth_started,
                })
    result = {
        "sizes": list(sizes),
        "max_steps": list(max_steps),
        "records": records,
        "scope": "realistic periodic rotating-Beltrami positive integer wide-stencil cone sweep",
        "limitation": "periodic scalar conductivity control; exact positive vector-map QC injectivity and boundary closure remain separate gates",
    }
    (output_dir / "mmatrix_stencil_width_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--max-steps", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes), tuple(args.max_steps)), indent=2))


if __name__ == "__main__":
    main()
