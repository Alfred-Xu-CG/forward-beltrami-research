"""Finite-direction adaptive-stencil audit on a realistic coefficient field."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import beltrami_conductivity, dictionary_conductances


def run(output_dir: Path, n: int = 256, sample_stride: int = 4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mu_field = 0.93 * np.tanh(2.0 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy))
    mu_field = mu_field * np.exp(1j * (0.7 * np.sin(2.0 * np.pi * yy) + 0.2 * np.cos(2.0 * np.pi * xx)))
    samples = mu_field[::sample_stride, ::sample_stride].ravel()
    records: list[dict] = []
    for n_directions in (8, 32, 128):
        residuals = []
        t0 = time.perf_counter()
        for coefficient in samples:
            residuals.append(
                dictionary_conductances(
                    beltrami_conductivity(complex(coefficient)),
                    n_directions=n_directions,
                ).residual
            )
        records.append(
            {
                "n_directions": n_directions,
                "samples": len(samples),
                "elapsed_seconds": time.perf_counter() - t0,
                "mean_residual": float(np.mean(residuals)),
                "max_residual": float(np.max(residuals)),
            }
        )
    result = {
        "field_grid": [n, n],
        "sample_stride": sample_stride,
        "records": records,
        "scope": "nonnegative finite-direction tensor fit; residual is the M-matrix stencil mismatch",
    }
    (output_dir / "mmatrix_dictionary_audit.json").write_text(
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
