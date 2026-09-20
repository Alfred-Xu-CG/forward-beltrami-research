"""Audit spatially varying torus zero-mode lift and composed implicit VJP."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling import periodic_beltrami_map_from_h
from qcopt.forward.beurling_gmres import periodic_beltrami_gmres
from qcopt.forward.beurling_implicit import periodic_beltrami_implicit_vjp, periodic_beltrami_map_h_vjp


def _objective(mu: np.ndarray, cotangent: np.ndarray) -> tuple[float, np.ndarray, object]:
    solved = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=120)
    mapped, _, _ = periodic_beltrami_map_from_h(solved.h)
    value = float(np.real(np.sum(np.conjugate(cotangent) * mapped)))
    return value, mapped, solved


def main() -> None:
    started = time.perf_counter()
    ny = nx = 256
    y = np.arange(ny, dtype=np.float64)[:, None] / ny
    x = np.arange(nx, dtype=np.float64)[None, :] / nx
    mu = 0.22 + 0.06j + 0.10 * np.exp(2j * np.pi * (2.0 * x - y)) + 0.05j * np.exp(2j * np.pi * (x + 3.0 * y))
    cotangent = 0.4 + np.exp(2j * np.pi * (3.0 * x - 2.0 * y)) + 0.4j * np.exp(2j * np.pi * (x + y))
    value, mapped, solved = _objective(mu, cotangent)
    grad_h = periodic_beltrami_map_h_vjp(solved.h, cotangent)
    gradient, adjoint_residual, adjoint_info = periodic_beltrami_implicit_vjp(mu, solved.h, grad_h, rtol=1e-9, maxiter=120)
    # A constant perturbation directly probes the spatially varying solve's
    # mean/period channel; it is not removed by the periodic inverse.
    direction = np.ones((ny, nx), dtype=np.complex128)
    direction /= np.linalg.norm(direction)
    eps = 2e-5
    plus = _objective(mu + eps * direction, cotangent)[0]
    minus = _objective(mu - eps * direction, cotangent)[0]
    finite_directional = float((plus - minus) / (2.0 * eps))
    predicted_directional = float(np.real(np.sum(np.conjugate(gradient) * direction)))
    h_mean = complex(np.mean(solved.h))
    result = {
        "grid": "256x256",
        "gmres_converged": bool(solved.converged),
        "gmres_residual": float(solved.operator_residual),
        "h_mean": [h_mean.real, h_mean.imag],
        "period_x": [(1.0 + h_mean).real, (1.0 + h_mean).imag],
        "period_y": [(1j * (1.0 - h_mean)).real, (1j * (1.0 - h_mean)).imag],
        "map_objective": value,
        "adjoint_residual": float(adjoint_residual),
        "adjoint_info": int(adjoint_info),
        "finite_directional": finite_directional,
        "predicted_directional": predicted_directional,
        "directional_abs_error": abs(finite_directional - predicted_directional),
        "map_finite": bool(np.all(np.isfinite(mapped))),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output = Path("D:/QC_optimization/artifacts/torus_spatial_zero_mode_map_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "torus_spatial_zero_mode_map_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
