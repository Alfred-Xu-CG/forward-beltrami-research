"""Matched nonperiodic rectangle FD-versus-LSQC solver audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.constraints import fixed_vertex_constraints
from qcopt.forward.rectangle_fd import rectangle_beltrami_fd
from qcopt.lsqc import solve_lsqc_fast
from qcopt.mesh import structured_rectangle


def _truth(mesh):
    x, y = mesh.vertices[:, 0], mesh.vertices[:, 1]
    return np.column_stack(
        (
            x + 0.13 * np.sin(np.pi * x) ** 2 * np.sin(2.0 * np.pi * y),
            y + 0.09 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) ** 2,
        )
    )


def _fd_mu(truth: np.ndarray, n_vertices: int) -> np.ndarray:
    spacing = 1.0 / (n_vertices - 1)
    complex_truth = truth[:, 0].reshape(n_vertices, n_vertices) + 1j * truth[:, 1].reshape(n_vertices, n_vertices)
    fx = (complex_truth[1:-1, 2:] - complex_truth[1:-1, :-2]) / (2.0 * spacing)
    fy = (complex_truth[2:, 1:-1] - complex_truth[:-2, 1:-1]) / (2.0 * spacing)
    fz = 0.5 * (fx - 1j * fy)
    fbar = 0.5 * (fx + 1j * fy)
    mu = np.zeros((n_vertices, n_vertices), dtype=np.complex128)
    mu[1:-1, 1:-1] = fbar / fz
    return mu


def _run(n: int) -> dict:
    mesh = structured_rectangle(n, n)
    n_vertices = n + 1
    truth = _truth(mesh)
    truth_complex = truth[:, 0].reshape(n_vertices, n_vertices) + 1j * truth[:, 1].reshape(n_vertices, n_vertices)
    fd_mu = _fd_mu(truth, n_vertices)
    lsqc_mu = face_beltrami(mesh, truth)
    boundary_vertices = np.flatnonzero(
        (mesh.vertices[:, 0] == 0.0)
        | (mesh.vertices[:, 0] == 1.0)
        | (mesh.vertices[:, 1] == 0.0)
        | (mesh.vertices[:, 1] == 1.0)
    )
    constraints = fixed_vertex_constraints(mesh.n_vertices, boundary_vertices, truth[boundary_vertices])
    fd_started = time.perf_counter()
    fd = rectangle_beltrami_fd(fd_mu, truth_complex, method="lsqr", iter_lim=4000)
    fd_seconds = time.perf_counter() - fd_started
    lsqc_started = time.perf_counter()
    lsqc = solve_lsqc_fast(mesh, lsqc_mu, constraints)
    lsqc_seconds = time.perf_counter() - lsqc_started
    lsqc_complex = lsqc.uv[:, 0].reshape(n_vertices, n_vertices) + 1j * lsqc.uv[:, 1].reshape(n_vertices, n_vertices)
    fd_interior = fd.map[1:-1, 1:-1]
    lsqc_interior = lsqc_complex.reshape(n_vertices, n_vertices)[1:-1, 1:-1]
    truth_interior = truth_complex[1:-1, 1:-1]
    return {
        "grid": f"{n}x{n} cells / {n_vertices}x{n_vertices} vertices",
        "faces": int(mesh.n_faces),
        "boundary_vertices": int(len(boundary_vertices)),
        "max_abs_mu_fd": float(np.max(np.abs(fd_mu))),
        "max_abs_mu_lsqc": float(np.max(np.abs(lsqc_mu))),
        "fd_seconds": fd_seconds,
        "lsqc_seconds": lsqc_seconds,
        "fd_equation_residual": fd.max_equation_residual,
        "lsqc_primal_residual": lsqc.primal_residual,
        "fd_map_rms_error": float(np.sqrt(np.mean(np.abs(fd_interior - truth_interior) ** 2))),
        "lsqc_map_rms_error": float(np.sqrt(np.mean(np.abs(lsqc_interior - truth_interior) ** 2))),
        "fd_lsqc_map_rms_difference": float(np.sqrt(np.mean(np.abs(fd_interior - lsqc_interior) ** 2))),
        "lsqc_constraint_residual": lsqc.constraint_residual,
        "finite": bool(np.all(np.isfinite(fd.map)) and np.all(np.isfinite(lsqc.uv))),
    }


def run(output_dir: Path, sizes: tuple[int, ...] = (64, 96)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {
        "records": [_run(n) for n in sizes],
        "scope": "same manufactured rectangle map and boundary data, nonperiodic FD BVP versus facewise LSQC",
        "interpretation": "both solvers see their own discretization-matched mu but identical boundary values; their difference quantifies discretization rather than FFT periodic wrapping",
        "limitation": "not a general rectangle fast solver or a hard-bijective guarantee; FD uses central differences and LSQC uses a P1 face residual",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "rectangle_matched_solver_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 96])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, tuple(args.sizes)), indent=2))


if __name__ == "__main__":
    main()
