"""Frequency/near-Nyquist error audit for compact-kernel particle mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_particle_mesh import particle_mesh_beurling_apply


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    xx, yy = np.meshgrid((np.arange(n) + 0.23) / n, (np.arange(n) + 0.37) / n, indexing="xy")
    points = (xx + 1j * yy).ravel()
    weights = np.full(points.shape, 1.0 / points.size)
    records = []
    for kx, ky in ((3, -2), (40, -35), (90, -80), (120, -115)):
        values = np.exp(2j * np.pi * (kx * points.real + ky * points.imag))
        expected = ((kx - 1j * ky) / (kx + 1j * ky)) * values
        result = particle_mesh_beurling_apply(
            points, values, weights, grid_shape=(n, n), kernel="cubic", deconvolve=True
        )
        records.append({
            "kx": kx,
            "ky": ky,
            "mean_abs_error": float(np.mean(np.abs(result - expected))),
            "max_abs_error": float(np.max(np.abs(result - expected))),
        })
    result = {
        "grid": f"{n}x{n}",
        "records": records,
        "scope": "periodic Fourier-mode dispersion of cubic B-spline particle mesh",
        "conclusion": "high-order gridding is highly accurate for low frequencies but deteriorates near Nyquist; it is not a uniformly accurate singular-integral solver",
    }
    (output_dir / "beurling_particle_mesh_dispersion_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()

