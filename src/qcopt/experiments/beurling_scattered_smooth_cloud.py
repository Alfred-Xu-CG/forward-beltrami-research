"""Create a smooth-density version of a scattered Beurling audit cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def run(input_path: Path, output_path: Path) -> None:
    data = np.load(input_path)
    points = np.asarray(data["points"], dtype=np.complex128)
    values = (
        np.exp(2j * np.pi * (2.0 * points.real - 1.0 * points.imag))
        + 0.35 * np.exp(2j * np.pi * (1.0 * points.real + 2.0 * points.imag))
        + 0.2 * (points.real - 0.25) * (points.imag - 0.5)
    )
    np.savez(
        output_path,
        points=points,
        targets=np.asarray(data["targets"], dtype=np.complex128),
        values=values,
        weights=np.asarray(data["weights"], dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
