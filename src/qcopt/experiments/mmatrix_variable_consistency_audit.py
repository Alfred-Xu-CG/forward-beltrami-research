"""Variable-conductivity consistency audit for positive wide stencils."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import (
    batch_positive_directional_conductances,
    integer_wide_stencil_directions,
)


def _periodic_derivative(field: np.ndarray, axis: int) -> np.ndarray:
    n = field.shape[axis]
    frequencies = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    shape = [1] * field.ndim
    shape[axis] = n
    transformed = np.fft.fft(field, axis=axis)
    return np.fft.ifft(1j * frequencies.reshape(shape) * transformed, axis=axis).real


def _manufactured(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    u = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy) + 0.4 * np.cos(6.0 * np.pi * xx + 0.2) * np.sin(2.0 * np.pi * yy)
    ux = 2.0 * np.pi * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy) - 0.4 * 6.0 * np.pi * np.sin(6.0 * np.pi * xx + 0.2) * np.sin(2.0 * np.pi * yy)
    uy = -2.0 * np.pi * np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy) + 0.4 * 2.0 * np.pi * np.cos(6.0 * np.pi * xx + 0.2) * np.cos(2.0 * np.pi * yy)
    return xx, yy, u, np.stack((ux, uy), axis=-1)


def _cone_tensor(xx: np.ndarray, yy: np.ndarray, directions: np.ndarray) -> np.ndarray:
    weights = np.stack(
        [
            1.2 + 0.15 * np.sin(2.0 * np.pi * xx) + 0.10 * np.cos(2.0 * np.pi * yy),
            0.9 + 0.12 * np.cos(2.0 * np.pi * xx) + 0.08 * np.sin(2.0 * np.pi * yy),
            0.7 + 0.08 * np.sin(2.0 * np.pi * (xx + yy)),
        ],
        axis=-1,
    )
    return np.einsum("...m,mi,mj->...ij", weights, directions, directions)


def _rotating_beltrami_tensor(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    rho = 0.30 + 0.08 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    angle = 0.35 + 0.55 * np.sin(2.0 * np.pi * xx) + 0.35 * np.cos(2.0 * np.pi * yy)
    mu = rho * np.exp(1j * angle)
    denominator = 1.0 - rho * rho
    return np.stack(
        (
            ((1.0 - 2.0 * mu.real + rho * rho) / denominator),
            (-2.0 * mu.imag / denominator),
            ((1.0 + 2.0 * mu.real + rho * rho) / denominator),
        ),
        axis=-1,
    ).reshape(xx.shape + (3,))


def _tensor_from_components(components: np.ndarray) -> np.ndarray:
    tensor = np.empty(components.shape[:-1] + (2, 2), dtype=np.float64)
    tensor[..., 0, 0] = components[..., 0]
    tensor[..., 0, 1] = components[..., 1]
    tensor[..., 1, 0] = components[..., 1]
    tensor[..., 1, 1] = components[..., 2]
    return tensor


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


def run(output_dir: Path, n_values: tuple[int, ...] = (128, 256, 512)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    integer_vectors = ((0, 1), (1, 0), (1, 1))
    full_directions = integer_wide_stencil_directions(2)
    directions = full_directions[[0, 3, 4]]
    records = []
    for case in ("cone_exact", "rotating_beltrami"):
        resolution_records = []
        for n in n_values:
            started = time.perf_counter()
            xx, yy, u, gradient = _manufactured(n)
            tensor = _cone_tensor(xx, yy, directions) if case == "cone_exact" else _tensor_from_components(_rotating_beltrami_tensor(xx, yy))
            fit = batch_positive_directional_conductances(tensor, directions)
            discrete = _discrete_operator(u, fit.values, integer_vectors)
            truth = _truth_operator(tensor, gradient)
            resolution_records.append(
                {
                    "n": n,
                    "fit_residual_rms": float(np.sqrt(np.mean(fit.residual * fit.residual))),
                    "fit_residual_max": float(np.max(fit.residual)),
                    "operator_l2_error": float(np.sqrt(np.mean((discrete - truth) ** 2))),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        orders = [
            float(np.log(old["operator_l2_error"] / new["operator_l2_error"]) / np.log(2.0))
            for old, new in zip(resolution_records, resolution_records[1:])
        ]
        records.append({"case": case, "resolution_records": resolution_records, "empirical_orders": orders})
    result = {
        "n_values": list(n_values),
        "directions": directions.tolist(),
        "records": records,
        "scope": "periodic variable-conductivity positive-wide-stencil consistency; cone-exact versus rotating Beltrami tensor",
        "limitation": "periodic regular grid control; positive cone fit is not a general unstructured-mesh QC decoder theorem",
    }
    (output_dir / "mmatrix_variable_consistency_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes)), indent=2))


if __name__ == "__main__":
    main()
