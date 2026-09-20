"""Separate fixed-point convergence from discrete Beltrami residual floors.

The sweep uses the same periodic Fourier Beurling operator for the fixed-point
iteration and for the spectral derivative control, then evaluates the returned
map with ordinary second-order central differences on the interior.  The last
quantity exposes the discretization/derivative floor that a small fixed-point
residual does not certify away.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..forward.beurling import (
    periodic_beurling_apply,
    periodic_beltrami_map_from_h,
    periodic_cauchy_inverse,
    periodic_frequency_grid,
)


def _spectral_beltrami_residual(h: np.ndarray, mu: np.ndarray) -> float:
    """Residual using the Fourier derivatives consistent with the operator."""

    ny, nx = h.shape
    frequencies = periodic_frequency_grid(nx, ny)
    fft_h = np.fft.fft2(h)
    dz_symbol = 0.5j * (frequencies.kx - 1j * frequencies.ky)
    dbar_symbol = 0.5j * (frequencies.kx + 1j * frequencies.ky)
    mean = np.mean(h)
    correction = periodic_cauchy_inverse(h - mean)
    fft_correction = np.fft.fft2(correction)
    fz = 1.0 + np.fft.ifft2(fft_correction * dz_symbol)
    fbar = mean + np.fft.ifft2(fft_correction * dbar_symbol)
    return float(np.max(np.abs(fbar - mu * fz)))


def _central_difference_residual(h: np.ndarray, mu: np.ndarray) -> float:
    """Residual after sampling the quasi-periodic map on the physical grid."""

    mapped, _, _ = periodic_beltrami_map_from_h(h)
    ny, nx = mapped.shape
    dx, dy = 1.0 / nx, 1.0 / ny
    fx = (mapped[1:-1, 2:] - mapped[1:-1, :-2]) / (2.0 * dx)
    fy = (mapped[2:, 1:-1] - mapped[:-2, 1:-1]) / (2.0 * dy)
    fz = 0.5 * (fx - 1j * fy)
    fbar = 0.5 * (fx + 1j * fy)
    residual = fbar - mu[1:-1, 1:-1] * fz
    return float(np.max(np.abs(residual)))


def run(
    output_dir: Path,
    *,
    n: int = 512,
    amplitude: float = 0.35,
    iteration_points: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64),
) -> dict:
    if n < 8 or amplitude <= 0.0 or amplitude >= 1.0:
        raise ValueError("n must be >= 8 and amplitude must lie in (0, 1)")
    if not iteration_points or any(k < 0 for k in iteration_points):
        raise ValueError("iteration_points must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    envelope = np.exp(-((xx - 0.50) ** 2 + (yy - 0.47) ** 2) / (2.0 * 0.13**2))
    phase = np.exp(2j * np.pi * (0.8 * xx - 0.55 * yy))
    mu = (amplitude * envelope * phase).astype(np.complex128)
    np.savez_compressed(output_dir / "data.npz", mu=mu)

    wanted = set(int(k) for k in iteration_points)
    h = np.zeros_like(mu)
    records: list[dict[str, float | int]] = []
    for iteration in range(max(wanted) + 1):
        if iteration in wanted:
            fixed = mu * (1.0 + periodic_beurling_apply(h))
            fixed_point_residual = float(np.max(np.abs(fixed - h)))
            started = time.perf_counter()
            spectral = _spectral_beltrami_residual(h, mu)
            spectral_seconds = time.perf_counter() - started
            started = time.perf_counter()
            central = _central_difference_residual(h, mu)
            central_seconds = time.perf_counter() - started
            records.append(
                {
                    "iterations": int(iteration),
                    "fixed_point_residual": fixed_point_residual,
                    "spectral_beltrami_residual": spectral,
                    "central_difference_beltrami_residual": central,
                    "spectral_seconds": spectral_seconds,
                    "central_difference_seconds": central_seconds,
                }
            )
        h = mu * (1.0 + periodic_beurling_apply(h))
    if max(wanted) in wanted and max(wanted) == 0:
        pass
    result = {
        "schema": "forward-beltrami-beurling-fixed-point-residual-floor-v1",
        "n": n,
        "amplitude": amplitude,
        "iteration_points": [int(k) for k in iteration_points],
        "records": records,
        "interpretation": "The spectral residual follows the fixed-point/operator consistency, while the central-difference residual exposes the sampled-map discretization floor.",
        "limitation": "This is a periodic torus operator audit; it does not impose rectangle boundary data or prove whole-plane convergence.",
    }
    (output_dir / "beurling_fixed_point_residual_floor_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_resolution_sweep(
    output_dir: Path,
    *,
    resolutions: tuple[int, ...] = (256, 512, 1024),
    amplitude: float = 0.35,
    iteration_points: tuple[int, ...] = (0, 8, 16),
) -> dict:
    """Repeat the floor audit at realistic resolutions and collect the tail."""

    if not resolutions or any(int(n) < 8 for n in resolutions):
        raise ValueError("resolutions must contain integers >= 8")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for resolution in resolutions:
        result = run(
            output_dir / f"n{int(resolution)}",
            n=int(resolution),
            amplitude=amplitude,
            iteration_points=iteration_points,
        )
        tail = result["records"][-1]
        runs.append(
            {
                "n": int(resolution),
                "iterations": int(tail["iterations"]),
                "fixed_point_residual": float(tail["fixed_point_residual"]),
                "spectral_beltrami_residual": float(tail["spectral_beltrami_residual"]),
                "central_difference_beltrami_residual": float(tail["central_difference_beltrami_residual"]),
            }
        )
    result = {
        "schema": "forward-beltrami-beurling-fixed-point-residual-floor-resolution-v1",
        "amplitude": amplitude,
        "iteration_points": [int(k) for k in iteration_points],
        "runs": runs,
        "interpretation": "The converged spectral residual is near machine precision while the central-difference residual decreases with mesh refinement, exposing a separate discretization floor.",
    }
    (output_dir / "resolution_sweep.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--iteration-points", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.resolutions is not None:
        result = run_resolution_sweep(
            args.output_dir,
            resolutions=tuple(args.resolutions),
            amplitude=args.amplitude,
            iteration_points=tuple(args.iteration_points),
        )
    else:
        result = run(args.output_dir, n=args.n, amplitude=args.amplitude, iteration_points=tuple(args.iteration_points))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
