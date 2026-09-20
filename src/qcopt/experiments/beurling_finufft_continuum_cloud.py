"""Generate a smooth uniform quadrature cloud for a FINUFFT control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def run(output: Path, side: int = 256, targets: int = 4096) -> None:
    axis_x = np.linspace(0.02, 0.46, side, endpoint=False) + 0.44 / (2.0 * side)
    axis_y = np.linspace(0.02, 0.98, side, endpoint=False) + 0.96 / (2.0 * side)
    xx, yy = np.meshgrid(axis_x, axis_y, indexing="xy")
    points = (xx + 1j * yy).reshape(-1)
    rng = np.random.default_rng(20260919)
    target_xy = rng.random((targets, 2))
    target_xy[:, 0] = 0.74 + 0.44 * target_xy[:, 0]
    target_xy[:, 1] = 0.02 + 0.96 * target_xy[:, 1]
    target_points = target_xy[:, 0] + 1j * target_xy[:, 1]
    values = (
        np.exp(2j * np.pi * (2.0 * points.real - points.imag))
        + 0.35 * np.exp(2j * np.pi * (points.real + 2.0 * points.imag))
        + 0.2 * (points.real - 0.25) * (points.imag - 0.5)
    )
    weights = np.full(points.shape, (0.44 * 0.96) / points.size, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, points=points, targets=target_points, values=values, weights=weights)
    print({"points": int(points.size), "targets": int(target_points.size), "weight_sum": float(weights.sum()), "output": str(output)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--targets", type=int, default=4096)
    args = parser.parse_args()
    run(args.output, args.side, args.targets)


if __name__ == "__main__":
    main()
