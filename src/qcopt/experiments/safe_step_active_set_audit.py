"""Realistic-mesh audit of the nonsmooth active-face safe-step map."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.safe_step import (
    maximum_safe_step,
    maximum_safe_step_directional_derivative,
    maximum_safe_step_with_active_face,
)
from qcopt.mesh import structured_rectangle


def run(output_dir: Path, n: int = 256, trials: int = 128, eps: float = 1e-3) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    rng = np.random.default_rng(20260919)
    switched = []
    fixed = []
    started = time.perf_counter()
    for trial in range(trials):
        delta = 0.12 * rng.normal(size=mesh.vertices.shape)
        d_uv = 0.02 * rng.normal(size=mesh.vertices.shape)
        d_delta = 0.02 * rng.normal(size=mesh.vertices.shape)
        uv = mesh.vertices.copy()
        base_step, base_active = maximum_safe_step_with_active_face(
            mesh, uv, delta, min_det_margin=0.05
        )
        plus_step, plus_active = maximum_safe_step_with_active_face(
            mesh, uv + eps * d_uv, delta + eps * d_delta, min_det_margin=0.05
        )
        minus_step, minus_active = maximum_safe_step_with_active_face(
            mesh, uv - eps * d_uv, delta - eps * d_delta, min_det_margin=0.05
        )
        finite = (plus_step - minus_step) / (2.0 * eps)
        try:
            analytic = maximum_safe_step_directional_derivative(
                mesh, uv, delta, d_uv, d_delta, min_det_margin=0.05
            )
            analytic_finite = bool(np.isfinite(analytic))
        except ValueError:
            analytic = float("nan")
            analytic_finite = False
        record = {
            "trial": trial,
            "base_step": float(base_step),
            "base_active_face": int(base_active),
            "plus_active_face": int(plus_active),
            "minus_active_face": int(minus_active),
            "active_switch": bool(plus_active != minus_active),
            "finite_difference_derivative": float(finite),
            "analytic_fixed_active_derivative": float(analytic),
            "analytic_finite": analytic_finite,
            "derivative_abs_error": float(abs(analytic - finite)) if analytic_finite else None,
        }
        (switched if record["active_switch"] else fixed).append(record)
        if len(switched) >= 8 and trial >= 16:
            break
    result = {
        "grid": f"{n}x{n} cells",
        "faces": int(mesh.n_faces),
        "trials_requested": trials,
        "trials_run": len(switched) + len(fixed),
        "active_switch_count": len(switched),
        "fixed_active_count": len(fixed),
        "switch_examples": switched[:8],
        "fixed_examples": fixed[:8],
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "256^2 determinant-root safe-step active-set switching audit",
        "interpretation": "the active-face derivative is valid inside a fixed active set; active-face switches expose a nonsmooth boundary requiring generalized/KKT differentiation",
        "limitation": "random perturbation scan, not a complete Clarke generalized-Jacobian theorem or solver-level active-set treatment",
    }
    (output_dir / "safe_step_active_set_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.trials, args.eps), indent=2))


if __name__ == "__main__":
    main()
