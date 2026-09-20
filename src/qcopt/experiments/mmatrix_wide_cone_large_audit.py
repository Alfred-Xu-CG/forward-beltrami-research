"""Large-radius positive-cone/operator audit without spatial smoothing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import batch_positive_directional_conductances, integer_wide_stencil_directions

from .mmatrix_stencil_width_audit import _discrete_operator, _integer_vectors, _manufactured, _truth_operator


def run(output_dir: Path, n: int = 512, max_steps: tuple[int, ...] = (4, 6, 8)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    u, gradient, tensor = _manufactured(n)
    truth = _truth_operator(tensor, gradient)
    records: list[dict[str, object]] = []
    for max_step in max_steps:
        started = time.perf_counter()
        directions = integer_wide_stencil_directions(max_step)
        vectors = _integer_vectors(max_step)
        fit = batch_positive_directional_conductances(tensor, directions)
        discrete = _discrete_operator(u, fit.values, vectors)
        records.append(
            {
                "n": n,
                "max_step": max_step,
                "direction_count": len(vectors),
                "fit_residual_rms": float(np.sqrt(np.mean(fit.residual * fit.residual))),
                "fit_residual_p95": float(np.quantile(fit.residual, 0.95)),
                "exact_fit_fraction": float(np.mean(fit.residual < 1e-10)),
                "weight_neighbor_rms": float(np.sqrt(0.5 * (np.mean(np.diff(fit.values, axis=0) ** 2) + np.mean(np.diff(fit.values, axis=1) ** 2)))),
                "operator_l2_error": float(np.sqrt(np.mean((discrete - truth) ** 2))),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    result = {
        "grid": f"{n}x{n} periodic cells",
        "records": records,
        "scope": "512² rotating-Beltrami positive integer wide-stencil cone sweep without spatial conductance smoothing",
        "interpretation": "The sweep isolates finite-direction cone coverage from the separate spatial-selection regularization problem.",
        "limitation": "periodic scalar conductivity control; no vector-map injectivity, boundary closure, or global QC decoder theorem",
    }
    (output_dir / "mmatrix_wide_cone_large_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--max-steps", type=int, nargs="+", default=[4, 6, 8])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, tuple(args.max_steps)), indent=2))


if __name__ == "__main__":
    main()
