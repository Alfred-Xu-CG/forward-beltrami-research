"""Run every benchmark and produce a hashed reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

from .audit_results import audit_artifacts
from .density_equalizing import run_density_equalizing
from .i_to_s import run_i_to_s
from .multichart_registration import run_multichart_registration
from .validate_solver import run_solver_validation


def run_all(
    output_directory: str | Path, *, quick: bool = False, seed: int = 20260828
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if quick:
        run_solver_validation(output / "solver_validation", grid_sizes=(2, 3), seed=seed)
        run_i_to_s(output / "i_to_s", nx=3, ny=3, iterations=2, seed=seed)
        run_density_equalizing(
            output / "density",
            nx=3,
            ny=3,
            iterations=2,
            seed=seed,
            target_names=("manufactured", "checkerboard"),
        )
        run_multichart_registration(output / "multichart", nx=3, ny=2)
    else:
        run_solver_validation(
            output / "solver_validation", grid_sizes=(8, 16, 32, 48, 64), seed=seed
        )
        run_i_to_s(output / "i_to_s", nx=12, ny=12, iterations=40, seed=seed)
        run_density_equalizing(
            output / "density", nx=10, ny=10, iterations=40, seed=seed
        )
        run_multichart_registration(output / "multichart", nx=12, ny=6)
    audit = audit_artifacts(output)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "mkl_threading_layer": "SEQUENTIAL",
    }
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest: dict[str, object] = {
        "mode": "quick" if quick else "full",
        "seed": seed,
        "environment": environment,
        "audit_status": audit["status"],
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    run_all(arguments.output, quick=arguments.quick, seed=arguments.seed)


if __name__ == "__main__":
    main()
