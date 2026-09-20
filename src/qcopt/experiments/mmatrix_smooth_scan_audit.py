"""Check scan-order sensitivity of the spatially continuous cone selector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.experiments.mmatrix_stencil_width_audit import _discrete_operator, _manufactured, _integer_vectors, _truth_operator
from qcopt.forward.mmatrix import integer_wide_stencil_directions
from qcopt.forward.mmatrix_smooth import smooth_positive_directional_conductances


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    u, gradient, tensor = _manufactured(n)
    truth = _truth_operator(tensor, gradient)
    directions = integer_wide_stencil_directions(2)
    vectors = _integer_vectors(2)
    records = []
    for name, reverse_rows, reverse_columns in (
        ("forward", False, False),
        ("reverse_columns", False, True),
    ):
        started = time.perf_counter()
        values, residual = smooth_positive_directional_conductances(
            tensor,
            directions,
            reverse_rows=reverse_rows,
            reverse_columns=reverse_columns,
        )
        discrete = _discrete_operator(u, values, vectors)
        records.append({
            "selection": name,
            "n": n,
            "operator_l2_error": float(np.sqrt(np.mean((discrete - truth) ** 2))),
            "fit_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "fit_residual_max": float(np.max(residual)),
            "weight_neighbor_rms": float(np.sqrt(0.5 * (np.mean(np.diff(values, axis=0) ** 2) + np.mean(np.diff(values, axis=1) ** 2)))),
            "weight_neighbor_max": float(max(np.max(np.abs(np.diff(values, axis=0))), np.max(np.abs(np.diff(values, axis=1))))),
            "elapsed_seconds": time.perf_counter() - started,
        })
    result = {
        "records": records,
        "scope": "scan-order sensitivity of exact positive wide-stencil conductance continuation",
        "limitation": "local raster continuation is a canonicalization diagnostic, not a scan-order-independent global optimization theorem",
    }
    (output_dir / "mmatrix_smooth_scan_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
