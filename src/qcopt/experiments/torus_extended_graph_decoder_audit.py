"""Extended positive-direction graph decoder audit on a periodic torus."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import beltrami_conductivity, edge_direction_conductances
from qcopt.forward.torus_tutte import periodic_positive_graph_embedding, periodic_torus_face_determinants

from .torus_cross_direction_decoder_audit import _case, _face_mu, _target


def run(output_dir: Path, n: int = 512) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = np.asarray(((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2)), dtype=np.int64)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    target = _target(n)
    mu = _face_mu(target, n).reshape(n, n)
    weights = np.empty((len(directions), n, n), dtype=np.float64)
    residuals = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            fit = edge_direction_conductances(beltrami_conductivity(complex(mu[j, i])), unit)
            weights[:, j, i] = np.maximum(fit.values, 1e-8)
            residuals[j, i] = fit.residual
    started = time.perf_counter()
    solved = periodic_positive_graph_embedding(n, n, directions, weights)
    elapsed = time.perf_counter() - started
    record = _case("axis_diagonal_two_step", solved, target, n, elapsed)
    record.update(
        {
            "direction_count": int(len(directions)),
            "directions": directions.tolist(),
            "fit_residual_mean": float(np.mean(residuals)),
            "fit_residual_p95": float(np.quantile(residuals, 0.95)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
        }
    )
    result = {
        "grid": f"{n}x{n} periodic vertices",
        "vertices": n * n,
        "periodic_faces": 2 * n * n,
        "records": [record],
        "scope": "positive periodic graph decoder with primitive directions through two lattice steps",
        "interpretation": "An extended positive cone tests whether diagonal-only coupling is the remaining local expressivity bottleneck while preserving periodic graph decoding and hard lifted-face orientation.",
        "limitation": "regular-grid periodic graph control; arbitrary triangulated torus, global variable-coefficient consistency, and arbitrary-mu inversion remain open",
    }
    (output_dir / "torus_extended_graph_decoder_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
