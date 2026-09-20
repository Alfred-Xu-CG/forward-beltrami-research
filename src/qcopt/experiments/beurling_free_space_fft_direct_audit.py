"""512² free-space direct GPU reference versus zero-padded FFT outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling import zero_padded_beurling_apply


def generate(output: Path, n: int = 512, target_side: int = 64) -> dict:
    axis = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = (xx + 1j * yy).reshape(-1)
    source = (
        np.exp(-((xx - 0.20) ** 2 + (yy - 0.50) ** 2) / (2.0 * 0.055**2))
        * np.exp(2j * np.pi * (1.5 * xx - 0.75 * yy))
    ).astype(np.complex128)
    # A disjoint target block on the opposite side of the unit square.
    start_x = int(0.70 * n)
    start_y = int(0.25 * n)
    target_x = np.arange(start_x, start_x + target_side, dtype=np.int64)
    target_y = np.arange(start_y, start_y + target_side, dtype=np.int64)
    tx, ty = np.meshgrid(target_x, target_y, indexing="xy")
    target_indices = (ty * n + tx).reshape(-1)
    target_points = points[target_indices]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, n=n, source=source, target_indices=target_indices, target_points=target_points)
    return {"n": n, "source_count": int(points.size), "target_count": int(target_points.size), "output": str(output)}


def compute_fft(data_path: Path, output_dir: Path, factors: tuple[int, ...]) -> dict:
    data = np.load(data_path)
    n = int(data["n"])
    source = np.asarray(data["source"], dtype=np.complex128).reshape(n, n)
    target_indices = np.asarray(data["target_indices"], dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for factor in factors:
        started = time.perf_counter()
        field = zero_padded_beurling_apply(source, padding_factor=factor)
        elapsed = time.perf_counter() - started
        records[str(factor)] = {
            "padding_factor": factor,
            "seconds": elapsed,
            "values_path": str(output_dir / f"fft_padding_{factor}.npy"),
        }
        np.save(output_dir / f"fft_padding_{factor}.npy", field.reshape(-1)[target_indices])
    return records


def compare(
    data_path: Path,
    output_dir: Path,
    factors: tuple[int, ...],
    direct_seconds: float | None = None,
    direct_interactions: int | None = None,
    direct_backend: str | None = None,
) -> dict:
    data = np.load(data_path)
    direct = np.load(output_dir / "direct_gpu.npy")
    records = {}
    for factor in factors:
        candidate = np.load(output_dir / f"fft_padding_{factor}.npy")
        difference = candidate - direct
        records[str(factor)] = {
            "padding_factor": factor,
            "relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(direct), 1e-30)),
            "max_abs_error": float(np.max(np.abs(difference))),
            "candidate_l2": float(np.linalg.norm(candidate)),
            "direct_l2": float(np.linalg.norm(direct)),
        }
    result = {
        "n": int(data["n"]),
        "source_count": int(data["source"].size),
        "target_count": int(data["target_points"].size),
        "records": records,
        "scope": "disjoint 512² source/target free-space direct reference versus zero-padded FFT",
        "interpretation": "The direct GPU sum has no periodic wrapping and no source-target self singularity; residual differences isolate finite-box/periodic-image and Fourier discretization effects.",
        "limitation": "The direct reference is a quadrature control on a bounded source window, not an exact analytic whole-plane solution; rectangle boundary conditions remain separate.",
    }
    if direct_seconds is not None:
        result["direct_reference_seconds"] = float(direct_seconds)
    if direct_interactions is not None:
        result["direct_reference_interactions"] = int(direct_interactions)
    if direct_backend is not None:
        result["direct_reference_backend"] = direct_backend
    (output_dir / "beurling_free_space_fft_direct_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=Path, default=None)
    parser.add_argument("--compute-fft", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--target-side", type=int, default=64)
    parser.add_argument("--factors", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--direct-seconds", type=float, default=None)
    parser.add_argument("--direct-interactions", type=int, default=None)
    parser.add_argument("--direct-backend", type=str, default=None)
    args = parser.parse_args()
    factors = tuple(args.factors)
    if args.generate is not None:
        print(json.dumps(generate(args.generate, args.n, args.target_side), indent=2))
    if args.compute_fft is not None:
        print(json.dumps(compute_fft(args.compute_fft, args.output_dir, factors), indent=2))
    if args.compare is not None:
        print(json.dumps(compare(
            args.compare,
            args.output_dir,
            factors,
            direct_seconds=args.direct_seconds,
            direct_interactions=args.direct_interactions,
            direct_backend=args.direct_backend,
        ), indent=2))


if __name__ == "__main__":
    main()
