"""Reflection-compatible rectangle extension audit for the Beurling route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.beurling import (
    periodic_beltrami_fixed_point,
    periodic_beltrami_map_from_h,
    periodic_lift_derivatives,
)
from qcopt.forward.beurling_reflection import reflection_extend_mu


def run(output_dir: Path, nx: int = 256, ny: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    envelope = np.exp(-((xx - 0.47) ** 2 + (yy - 0.54) ** 2) / 0.16)
    real_part = 0.30 * envelope * (1.0 + 0.12 * np.cos(4.0 * np.pi * xx - 2.0 * np.pi * yy))
    imaginary_part = (
        0.12
        * envelope
        * np.sin(2.0 * np.pi * xx)
        * np.sin(2.0 * np.pi * yy)
        * (16.0 * xx * (1.0 - xx) * yy * (1.0 - yy))
    )
    mu = (real_part + 1j * imaginary_part).astype(np.complex128)
    extension = reflection_extend_mu(mu)
    result = periodic_beltrami_fixed_point(extension, max_iter=500, tolerance=1e-10)
    mapped, period_x, period_y = periodic_beltrami_map_from_h(result.h)
    fz, fbar = periodic_lift_derivatives(result.h)
    # ``periodic_lift_derivatives`` differentiates only the zero-mean
    # correction; the map-level lift contributes mean(h) to f_bar.
    fbar_map = fbar + np.mean(result.h)
    determinant = np.abs(fz) ** 2 - np.abs(fbar) ** 2
    equation_residual = fbar_map - extension * fz
    base = np.s_[:ny, :nx]
    metrics = {
        "base_grid": f"{nx}x{ny}",
        "extended_grid": f"{2 * nx}x{2 * ny}",
        "max_abs_mu": float(np.max(np.abs(mu))),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "fixed_point_residual": float(result.residual),
        "equation_residual_linf_base": float(np.max(np.abs(equation_residual[base]))),
        "minimum_determinant_base": float(np.min(determinant[base])),
        "horizontal_axis_imag_linf": float(np.max(np.abs(mapped[0, :nx].imag))),
        "vertical_axis_real_linf": float(np.max(np.abs(mapped[:ny, 0].real))),
        "period_x": [float(period_x.real), float(period_x.imag)],
        "period_y": [float(period_y.real), float(period_y.imag)],
        "scope": "reflection-compatible doubled-periodic rectangle control",
        "limitation": "covers only reflection-symmetric boundary data; not all rectangle boundary homeomorphisms or a production boundary solver",
    }
    (output_dir / "beurling_reflection_rectangle_audit.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields.npz", mu=mu, extension=extension, mapped=mapped)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.nx, args.ny), indent=2))


if __name__ == "__main__":
    main()
