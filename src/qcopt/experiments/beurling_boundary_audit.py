"""Measure periodic-image error in FFT Beurling approximations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..forward.beurling import periodic_beurling_apply, zero_padded_beurling_apply


def run_boundary_audit(
    output: str | Path,
    *,
    nx: int = 512,
    ny: int = 512,
    sigma: float = 0.03,
    padding_factors: Iterable[int] = (2, 4),
) -> dict[str, Any]:
    """Compare torus and centered zero-padded whole-plane approximations."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if nx < 2 or ny < 2 or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("invalid grid or Gaussian width")
    factors = tuple(int(factor) for factor in padding_factors)
    if not factors or any(factor < 2 for factor in factors):
        raise ValueError("padding_factors must contain integers at least two")

    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    source = np.exp(-((xx - 0.04) ** 2 + (yy - 0.5) ** 2) / (2.0 * sigma**2)).astype(
        np.complex128
    )
    start = time.perf_counter()
    periodic = periodic_beurling_apply(source)
    metrics: dict[str, Any] = {
        "periodic": {"elapsed_s": time.perf_counter() - start}
    }
    fields: dict[str, np.ndarray] = {"source": source, "periodic": periodic}
    for factor in factors:
        start = time.perf_counter()
        padded = zero_padded_beurling_apply(source, padding_factor=factor)
        difference = periodic - padded
        metrics[f"padding_{factor}"] = {
            "elapsed_s": time.perf_counter() - start,
            "padding_factor": factor,
            "max_abs_difference": float(np.max(np.abs(difference))),
            "l2_difference": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
            "opposite_boundary_difference": float(
                np.abs(difference[ny // 2, -1])
            ),
        }
        fields[f"padding_{factor}"] = padded
    config = {
        "schema": "forward-beltrami-beurling-boundary-audit-v1",
        "nx": nx,
        "ny": ny,
        "sigma": sigma,
        "padding_factors": list(factors),
        "source_center": [0.04, 0.5],
    }
    _write_json(output / "config.json", config)
    _write_json(output / "metrics.json", metrics)
    np.savez_compressed(output / "fields.npz", **fields)
    manifest = {
        "schema": "forward-beltrami-beurling-boundary-manifest-v1",
        "files": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {"config": config, "metrics": metrics, "manifest": manifest}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--padding-factors", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    run_boundary_audit(
        args.output,
        nx=args.nx,
        ny=args.ny,
        sigma=args.sigma,
        padding_factors=args.padding_factors,
    )


if __name__ == "__main__":
    main()
