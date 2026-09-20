"""Mesh-native integer wide-stencil audit for anisotropic Beltrami tensors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import (
    beltrami_conductivity,
    integer_wide_stencil_conductances,
    integer_wide_stencil_directions,
)


def run(output_dir: Path, n: int = 256, sample_stride: int = 8) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mu_field = (0.96 * np.tanh(
        2.2 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    )).astype(np.complex128)
    mu_field *= np.exp(1j * (0.75 * np.sin(2.0 * np.pi * yy) + 0.25 * np.cos(2.0 * np.pi * xx)))
    samples = mu_field[::sample_stride, ::sample_stride].ravel()
    records: list[dict] = []
    for max_step in (1, 2, 3, 4, 6):
        directions = integer_wide_stencil_directions(max_step)
        residuals = []
        positive = []
        t0 = time.perf_counter()
        for coefficient in samples:
            fit = integer_wide_stencil_conductances(
                beltrami_conductivity(complex(coefficient)), max_step=max_step
            )
            residuals.append(fit.residual)
            positive.append(bool(np.all(fit.values >= -1e-12)))
        records.append(
            {
                "max_step": max_step,
                "directions": int(len(directions)),
                "samples": int(len(samples)),
                "elapsed_seconds": time.perf_counter() - t0,
                "mean_residual": float(np.mean(residuals)),
                "max_residual": float(np.max(residuals)),
                "all_nonnegative": bool(all(positive)),
                "scope": "integer directions correspond to regular-grid wide edges",
            }
        )
    result = {
        "field_grid": [n, n],
        "sample_stride": sample_stride,
        "records": records,
        "scope": "adaptive mesh-native wide-stencil tensor decomposition",
        "limitation": "wide stencils require regular-grid connectivity and boundary closure; this is not yet a general unstructured-triangulation QC decoder",
    }
    (output_dir / "mmatrix_wide_stencil_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--sample-stride", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, n=args.n, sample_stride=args.sample_stride), indent=2))


if __name__ == "__main__":
    main()
