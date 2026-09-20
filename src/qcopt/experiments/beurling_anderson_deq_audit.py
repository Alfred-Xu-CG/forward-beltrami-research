"""Anderson fixed-point/DEQ versus GMRES audit for periodic Beltrami solves."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling import periodic_beltrami_map_from_h
from qcopt.forward.beurling_gmres import periodic_beltrami_anderson, periodic_beltrami_gmres
from qcopt.forward.beurling_implicit import periodic_beltrami_implicit_vjp


def main() -> None:
    started = time.perf_counter()
    n = 128
    x = np.arange(n, dtype=np.float64)[None, :] / n
    y = np.arange(n, dtype=np.float64)[:, None] / n
    records = []
    for amplitude in (0.25, 0.4, 0.55):
        mu = amplitude * np.exp(2j * np.pi * (2.0 * x - y + 0.2 * np.sin(2.0 * np.pi * y)))
        anderson_started = time.perf_counter()
        fixed = periodic_beltrami_anderson(mu, depth=6, rtol=1e-9, maxiter=200)
        anderson_seconds = time.perf_counter() - anderson_started
        gmres_started = time.perf_counter()
        linear = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=200)
        gmres_seconds = time.perf_counter() - gmres_started
        mapped, period_x, period_y = periodic_beltrami_map_from_h(fixed.h)
        cotangent = np.exp(2j * np.pi * (x + 3.0 * y))
        grad_h, adjoint_residual, adjoint_info = periodic_beltrami_implicit_vjp(mu, fixed.h, cotangent, rtol=1e-9, maxiter=200)
        records.append({
            "amplitude": amplitude,
            "anderson_iterations": fixed.iterations,
            "anderson_seconds": anderson_seconds,
            "anderson_residual": fixed.fixed_point_residual,
            "anderson_converged": fixed.converged,
            "gmres_iterations": linear.iterations,
            "gmres_seconds": gmres_seconds,
            "gmres_residual": linear.operator_residual,
            "root_difference": float(np.max(np.abs(fixed.h - linear.h))),
            "period_x_abs": abs(period_x),
            "period_y_abs": abs(period_y),
            "map_finite": bool(np.all(np.isfinite(mapped))),
            "deq_adjoint_residual": float(adjoint_residual),
            "deq_adjoint_info": int(adjoint_info),
            "cotangent_norm": float(np.linalg.norm(grad_h)),
        })
    result = {
        "grid": "128x128",
        "records": records,
        "scope": "short-memory Anderson forward equilibrium with implicit adjoint reusing the fixed-point Jacobian",
        "limitation": "Anderson is not uniformly faster than GMRES and the adjoint is audited through the linear Beltrami implicit solve, not a general nonlinear KKT layer",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path("D:/QC_optimization/artifacts/beurling_anderson_deq_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "beurling_anderson_deq_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
