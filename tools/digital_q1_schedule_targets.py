"""Make a fixed-boundary analytic shear target for the F1/F2 schedule ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def create_sine_shear_target(path: Path, *, side: int, amplitude: float) -> dict:
    if side < 3 or side % 2 != 1 or not math.isfinite(amplitude) or (
        abs(amplitude) * math.pi >= 1
    ):
        raise ValueError("need odd side >= 3 and |amplitude| pi < 1")
    axis = np.arange(side, dtype=np.float64) / (side - 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    reference = np.stack((xx, yy), axis=-1)[None].astype(np.float32)
    teacher = np.stack((
        xx + amplitude * np.sin(math.pi * xx) * np.sin(math.pi * yy), yy,
    ), axis=-1)[None].astype(np.float32)
    np.savez_compressed(
        path,
        teacher_vertices=teacher,
        vertices=teacher,
        boundary_reference=reference,
        post_affine_matrix=np.eye(2, dtype=np.float32),
        post_affine_offset=np.zeros(2, dtype=np.float32),
        analytic_amplitude=np.float64(amplitude),
    )
    certificate = certify_q1_binary_map(path)
    if not certificate["valid"]:
        raise ArithmeticError("analytic target lost saved-binary Q1 positivity")
    return {
        "target": str(path), "side": side, "amplitude": amplitude,
        "continuum_jacobian_lower_bound": 1 - abs(amplitude) * math.pi,
        "saved_binary_valid": certificate["valid"],
        "saved_binary_nonpositive_corners": certificate["nonpositive_corners"],
        "center_target_horizontal_displacement": float(
            teacher[0, side // 2, side // 2, 0] - reference[0, side // 2, side // 2, 0]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--amplitude", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_sine_shear_target(
        args.output, side=args.side, amplitude=args.amplitude,
    ), indent=2))


if __name__ == "__main__":
    main()
