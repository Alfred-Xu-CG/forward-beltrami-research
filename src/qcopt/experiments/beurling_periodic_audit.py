"""Periodic fixed-point audit separating solver convergence from BC consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..forward.beurling import periodic_beltrami_fixed_point, periodic_lift_derivatives


def run_periodic_audit(
    output: str | Path,
    *,
    nx: int = 512,
    ny: int = 512,
    amplitude: float = 0.2,
    tolerance: float = 1e-10,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Run mean-zero and constant-coefficient periodic fixed-point cases."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if nx < 2 or ny < 2 or not 0.0 < amplitude < 1.0:
        raise ValueError("invalid grid or amplitude")
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mean_zero = amplitude * np.exp(2j * np.pi * (2.0 * xx - yy))
    constant = np.full((ny, nx), amplitude + 0.1j, dtype=np.complex128)
    metrics = {
        "mean_zero": _solve_case(mean_zero, max_iter, tolerance),
        "constant": _solve_case(constant, max_iter, tolerance),
    }
    config = {
        "schema": "forward-beltrami-periodic-audit-v1",
        "nx": nx,
        "ny": ny,
        "amplitude": amplitude,
        "tolerance": tolerance,
        "max_iter": max_iter,
    }
    _write_json(output / "config.json", config)
    _write_json(output / "metrics.json", metrics)
    manifest = {
        "schema": "forward-beltrami-periodic-audit-manifest-v1",
        "files": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {"config": config, "metrics": metrics, "manifest": manifest}


def _solve_case(mu: np.ndarray, max_iter: int, tolerance: float) -> dict[str, Any]:
    result = periodic_beltrami_fixed_point(
        mu, max_iter=max_iter, tolerance=tolerance
    )
    fz, fbar = periodic_lift_derivatives(result.h)
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "fixed_point_residual": result.residual,
        "equation_linf": float(np.max(np.abs(fbar - mu * fz))),
        "mean_abs_h": float(np.abs(np.mean(result.h))),
        "max_abs_mu": float(np.max(np.abs(mu))),
    }


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
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    run_periodic_audit(
        args.output,
        nx=args.nx,
        ny=args.ny,
        amplitude=args.amplitude,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
