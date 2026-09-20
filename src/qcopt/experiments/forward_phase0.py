"""Reproducible Phase 0 manufactured-map benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..forward.benchmarks import affine_map, manufactured_mu, smooth_twist_map
from ..forward.operators import evaluate_forward_metrics
from ..mesh import structured_rectangle


def run_phase0(
    output: str | Path,
    *,
    nx: int = 256,
    ny: int = 256,
    amplitude: float = 0.08,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the shared manufactured benchmark and write only explicit artifacts."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")

    mesh = structured_rectangle(nx, ny)
    source = mesh.vertices
    affine = affine_map(
        source,
        np.asarray([[1.1, 0.12], [0.04, 0.95]], dtype=np.float64),
        np.asarray([0.05, -0.03], dtype=np.float64),
    )
    smooth_twist = smooth_twist_map(source, amplitude=amplitude)
    maps = {"source": source, "affine": affine, "smooth_twist": smooth_twist}
    metrics = {
        "affine": asdict(
            evaluate_forward_metrics(mesh, affine, manufactured_mu(mesh, affine))
        ),
        "smooth_twist": asdict(
            evaluate_forward_metrics(
                mesh, smooth_twist, manufactured_mu(mesh, smooth_twist)
            )
        ),
    }
    config = {
        "schema": "forward-beltrami-phase0-v1",
        "seed": int(seed),
        "nx": int(nx),
        "ny": int(ny),
        "amplitude": float(amplitude),
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
    }
    _write_json(output / "config.json", config)
    _write_json(output / "metrics.json", metrics)
    np.savez_compressed(output / "maps.npz", **maps)
    manifest = {
        "schema": "forward-beltrami-phase0-manifest-v1",
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
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_phase0(
        args.output,
        nx=args.nx,
        ny=args.ny,
        amplitude=args.amplitude,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
