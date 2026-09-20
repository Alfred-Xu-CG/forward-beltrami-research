"""Reproducible CPU benchmark for the Phase III MBM implicit VJP prototype.

Run this command in a fresh process for each resolution when RSS deltas are
needed::

    python -m qcopt.experiments.phase3_mbm_vjp_benchmark --n 129

The reported RSS is a process-resident-set delta, not allocator peak.  The
independent residual evaluator is deliberately separate from the autograd
forward implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import psutil
import scipy
import torch

from qcopt.forward.mbm_lbs import _structured_geometry, solve_mbm_lbs
from qcopt.forward.mbm_lbs_implicit import mbm_lbs_torch_implicit


def _induced_face_mu_accuracy(coefficient: np.ndarray, independent) -> dict[str, float]:
    """Measure induced P1 Beltrami error against face-averaged input data.

    This deliberately uses an independent vectorized face-gradient path rather
    than the conjugacy residual accumulated inside ``solve_mbm_lbs``.
    """
    triangles, gradients, _areas, face_mu = _structured_geometry(coefficient)
    faces = np.asarray(triangles, dtype=np.int64)
    u_local = independent.u.reshape(-1)[faces]
    v_local = independent.v.reshape(-1)[faces]
    grad_u = np.einsum("fi,fij->fj", u_local, gradients)
    grad_v = np.einsum("fi,fij->fj", v_local, gradients)
    ux, uy = grad_u[:, 0], grad_u[:, 1]
    vx, vy = grad_v[:, 0], grad_v[:, 1]
    fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
    fbar = 0.5 * ((ux - vy) + 1j * (vx + uy))
    induced = fbar / fz
    difference = induced - np.asarray(face_mu, dtype=np.complex128)
    return {
        "induced_mu_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "induced_mu_max_error": float(np.max(np.abs(difference))),
        "induced_mu_max_abs": float(np.max(np.abs(induced))),
    }

def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                check=True,
                capture_output=True,
                text=True,
            )
            value = completed.stdout.strip()
            if value:
                return value
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.uname().processor


def _smooth_coefficient(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    x = xx / (n - 1) - 0.5
    y = yy / (n - 1) - 0.5
    return 0.25 * np.exp(-(x * x + y * y) / 0.18) * np.exp(0.7j)


def run(n: int) -> dict[str, object]:
    if n < 3:
        raise ValueError("n must be at least 3")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    coefficient = _smooth_coefficient(n)
    real = torch.tensor(coefficient.real, requires_grad=True)
    imag = torch.tensor(coefficient.imag, requires_grad=True)
    process = psutil.Process()
    gc.collect()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    mapped = mbm_lbs_torch_implicit(real, imag)
    forward_seconds = time.perf_counter() - started
    loss = torch.sum(mapped * mapped)
    started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - started
    rss_after = process.memory_info().rss
    independent = solve_mbm_lbs(coefficient)
    induced_accuracy = _induced_face_mu_accuracy(coefficient, independent)
    return {
        "n_vertices_per_axis": n,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "rss_delta_bytes": int(rss_after - rss_before),
        "conjugacy_residual": independent.conjugacy_residual,
        "minimum_normalized_face_determinant": independent.min_triangle_determinant,
        **induced_accuracy,
        "finite_output": bool(torch.isfinite(mapped).all()),
        "finite_gradient": bool(torch.isfinite(real.grad).all() and torch.isfinite(imag.grad).all()),
        "cpu": _cpu_name(),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "psutil": psutil.__version__,
    }


def directional_gradient_check() -> dict[str, float]:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    n = 7
    real = torch.tensor(0.12 * rng.normal(size=(n, n)), requires_grad=True)
    imag = torch.tensor(0.08 * rng.normal(size=(n, n)), requires_grad=True)
    mapped = mbm_lbs_torch_implicit(real, imag)
    (mapped * mapped).sum().backward()
    direction_real = torch.tensor(rng.normal(size=(n, n)))
    direction_imag = torch.tensor(rng.normal(size=(n, n)))
    predicted = float((real.grad * direction_real + imag.grad * direction_imag).sum())
    epsilon = 1e-6
    with torch.no_grad():
        plus = (mbm_lbs_torch_implicit(real + epsilon * direction_real, imag + epsilon * direction_imag) ** 2).sum()
        minus = (mbm_lbs_torch_implicit(real - epsilon * direction_real, imag - epsilon * direction_imag) ** 2).sum()
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    return {
        "n": n,
        "seed": 3,
        "epsilon": epsilon,
        "predicted": predicted,
        "finite_difference": finite_difference,
        "relative_error": abs(predicted - finite_difference) / max(abs(finite_difference), 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--gradient-check", action="store_true")
    args = parser.parse_args()
    result: dict[str, object] = {"benchmark": run(args.n)}
    if args.gradient_check:
        result["gradient_check"] = directional_gradient_check()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
