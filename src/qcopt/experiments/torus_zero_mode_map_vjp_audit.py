"""Realistic-resolution audit of the affine torus map/period VJP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_implicit import torus_affine_map_vjp


def run(output_dir: Path, n: int = 512) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    points = (xx + 1j * yy).ravel()
    cotangent = (
        np.sin(2.0 * np.pi * xx) + 1j * np.cos(2.0 * np.pi * yy)
    ).ravel()
    mu = 0.23 + 0.17j
    direction = -0.07 + 0.11j
    gp, gq = -0.4 + 0.2j, 0.13 - 0.31j

    def loss(value: complex) -> float:
        mapped = points + value * np.conjugate(points)
        px, py = 1.0 + value, 1j * (1.0 - value)
        return float(
            np.real(
                np.sum(np.conjugate(cotangent) * mapped)
                + np.conjugate(gp) * px
                + np.conjugate(gq) * py
            )
        )

    t0 = time.perf_counter()
    gradient = torus_affine_map_vjp(
        points, cotangent, grad_period_x=gp, grad_period_y=gq
    )
    vjp_seconds = time.perf_counter() - t0
    eps = 1e-7
    finite_difference = (loss(mu + eps * direction) - loss(mu - eps * direction)) / (2.0 * eps)
    directional_vjp = float(np.real(np.conjugate(gradient) * direction))
    result = {
        "grid": f"{n}x{n}",
        "points": int(points.size),
        "vjp_seconds": vjp_seconds,
        "gradient": [float(gradient.real), float(gradient.imag)],
        "finite_difference_directional": finite_difference,
        "vjp_directional": directional_vjp,
        "absolute_directional_error": abs(finite_difference - directional_vjp),
        "scope": "constant/affine torus zero-mode map plus quasi-period VJP",
        "limitation": "does not differentiate a spatially varying nonlinear torus solve or arbitrary triangulated-torus decoder",
    }
    (output_dir / "torus_zero_mode_map_vjp_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, n=args.n), indent=2))


if __name__ == "__main__":
    main()
