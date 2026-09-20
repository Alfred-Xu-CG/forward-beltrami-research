"""Cost-inclusive truncated-Neumann GMRES preconditioner audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_gmres import periodic_beltrami_gmres


def main() -> None:
    started = time.perf_counter()
    n = 256
    x = np.arange(n, dtype=np.float64)[None, :] / n
    y = np.arange(n, dtype=np.float64)[:, None] / n
    records = []
    for amplitude in (0.25, 0.4, 0.55, 0.7):
        mu = amplitude * np.exp(2j * np.pi * (2.0 * x - y + 0.25 * np.sin(2.0 * np.pi * y) + 0.15 * np.cos(4.0 * np.pi * x)))
        cold_started = time.perf_counter()
        cold = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=120)
        cold_seconds = time.perf_counter() - cold_started
        case = {"amplitude": amplitude, "cold_iterations": cold.iterations, "cold_seconds": cold_seconds}
        for order in (1, 2, 3):
            pre_started = time.perf_counter()
            preconditioned = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=120, preconditioner_order=order)
            elapsed = time.perf_counter() - pre_started
            case[f"order{order}"] = {
                "iterations": preconditioned.iterations,
                "elapsed_seconds": elapsed,
                "speedup": cold_seconds / elapsed,
                "iteration_reduction": cold.iterations - preconditioned.iterations,
                "residual": preconditioned.operator_residual,
                "root_difference": float(np.max(np.abs(preconditioned.h - cold.h))),
                "converged": bool(preconditioned.converged),
            }
        records.append(case)
    result = {
        "grid": "256x256",
        "records": records,
        "scope": "true matrix-free GMRES preconditioning, cost included in wall time",
        "limitation": "transparent truncated-Neumann baseline, not a learned/general robust preconditioner",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path("D:/QC_optimization/artifacts/neumann_preconditioner_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "neumann_preconditioner_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
