"""High-resolution P1 block-symbol and continuous-dispersion audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.p1_symbols import p1_beurling_block_symbol


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    projector_error = 0.0
    max_norm = 0.0
    null_modes = 0
    low_errors = []
    high_errors = []
    for ky in range(n):
        sy = ky if ky < n // 2 else ky - n
        for kx in range(n):
            sx = kx if kx < n // 2 else kx - n
            a, b, p, s = p1_beurling_block_symbol(kx, ky, n, n)
            projector = b[:, None] @ p
            projector_error = max(projector_error, float(np.linalg.norm(projector @ projector - projector)))
            max_norm = max(max_norm, float(np.linalg.norm(s, 2)))
            if np.linalg.norm(b) <= 1e-14:
                null_modes += 1
                continue
            numerical_ratio = np.vdot(b, a) / np.vdot(b, b)
            if sx == 0 and sy == 0:
                continue
            continuous_ratio = (1j * sx + sy) / (1j * sx - sy)
            error = abs(numerical_ratio - continuous_ratio)
            if abs(sx) <= n // 8 and abs(sy) <= n // 8:
                low_errors.append(error)
            else:
                high_errors.append(error)
    result = {
        "grid": f"{n}x{n} Fourier modes",
        "elapsed_seconds": time.perf_counter() - t0,
        "null_modes": null_modes,
        "projector_max_idempotence_error": projector_error,
        "max_block_symbol_spectral_norm": max_norm,
        "low_frequency_ratio_mean_error": float(np.mean(low_errors)),
        "low_frequency_ratio_max_error": float(np.max(low_errors)),
        "high_frequency_ratio_mean_error": float(np.mean(high_errors)),
        "high_frequency_ratio_max_error": float(np.max(high_errors)),
        "scope": "P1 two-face weighted pseudoinverse and rank-one block Beurling symbol",
        "limitation": "periodic regular triangulation only; no physical-boundary or unstructured-mesh FFT acceleration",
    }
    (output_dir / "p1_symbol_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()

