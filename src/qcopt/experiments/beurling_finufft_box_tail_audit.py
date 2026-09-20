"""Probe finite-box versus Fourier-cutoff error in a scattered FINUFFT audit.

This standalone script is copied to the AI host because FINUFFT is installed
there.  It keeps the source/target cloud and the independent direct reference
fixed, then increases both box length and Fourier mode count so that the
physical cutoff remains approximately constant.  The experiment is a
diagnostic of the whole-plane truncation model, not a rectangle boundary
solver.
"""

from __future__ import annotations

import argparse
import json
import time

import finufft
import numpy as np


def apply(data: dict[str, np.ndarray], modes: int, box_length: float, eps: float) -> np.ndarray:
    points = data["points"]
    targets = data["targets"]
    charge = data["values"] * data["weights"]
    scale = 2.0 * np.pi / box_length
    coeff = finufft.nufft2d1(
        scale * points.real,
        scale * points.imag,
        charge,
        (modes, modes),
        isign=-1,
        eps=eps,
    )
    indices = np.arange(-modes // 2, modes // 2, dtype=np.float64)
    kx, ky = np.meshgrid(indices, indices, indexing="ij")
    denominator = kx + 1j * ky
    symbol = np.zeros_like(denominator, dtype=np.complex128)
    nonzero = np.abs(denominator) > 0.0
    symbol[nonzero] = (kx[nonzero] - 1j * ky[nonzero]) / denominator[nonzero]
    filtered = symbol * coeff / (box_length * box_length)
    return finufft.nufft2d2(
        scale * targets.real,
        scale * targets.imag,
        filtered,
        isign=1,
        eps=eps,
    )


def run(input_path: str, reference_path: str, output_path: str, eps: float = 1.0e-10) -> dict:
    data = dict(np.load(input_path))
    reference = np.load(reference_path)
    # Keep pi*modes/L approximately fixed while increasing the free-space box.
    cases = ((8.0, 2048), (12.0, 3072), (16.0, 4096))
    records: list[dict[str, object]] = []
    for box_length, modes in cases:
        started = time.perf_counter()
        output = apply(data, modes, box_length, eps)
        seconds = time.perf_counter() - started
        error = output - reference
        records.append(
            {
                "box_length": box_length,
                "modes": modes,
                "physical_kmax": float(np.pi * modes / box_length),
                "seconds": seconds,
                "relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(reference)),
                "max_abs_error": float(np.max(np.abs(error))),
                "output_l2": float(np.linalg.norm(output)),
                "finite": bool(np.all(np.isfinite(output))),
            }
        )
    result = {
        "records": records,
        "source_count": int(data["points"].size),
        "target_count": int(data["targets"].size),
        "scope": "fixed smooth-cloud Beurling FINUFFT box-tail audit",
        "limitation": "whole-plane truncated Fourier diagnostic; it does not close singular PV or rectangle-boundary quadrature",
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eps", type=float, default=1.0e-10)
    args = parser.parse_args()
    run(args.input, args.reference, args.output, args.eps)


if __name__ == "__main__":
    main()
