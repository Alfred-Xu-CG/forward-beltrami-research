"""Matrix-free nonlinear barrier-KKT forward/adjoint audit."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg

from qcopt.experiments.barrier_kkt_implicit_audit import _problem, _target


torch.set_default_dtype(torch.float64)


def _hvp(q: torch.Tensor, target: torch.Tensor, energy, vector: torch.Tensor) -> torch.Tensor:
    q_eval = q.detach().requires_grad_(True)
    value = energy(q_eval, target)
    gradient = torch.autograd.grad(value, q_eval, create_graph=True)[0]
    return torch.autograd.grad(torch.dot(gradient, vector), q_eval)[0].detach()


def _cg_solve(q: torch.Tensor, target: torch.Tensor, energy, rhs: torch.Tensor, damping: float, rtol: float, maxiter: int):
    if rhs.device.type != "cpu":
        return _cg_solve_torch(q, target, energy, rhs, damping, rtol, maxiter)
    calls = {"hvp": 0}

    def matvec(vector: np.ndarray) -> np.ndarray:
        calls["hvp"] += 1
        vector_t = torch.from_numpy(np.asarray(vector, dtype=np.float64))
        result = _hvp(q, target, energy, vector_t) + damping * vector_t
        return result.numpy()

    operator = LinearOperator((rhs.numel(), rhs.numel()), matvec=matvec, dtype=np.float64)
    solution, info = cg(operator, rhs.numpy(), rtol=rtol, atol=0.0, maxiter=maxiter)
    return torch.from_numpy(np.asarray(solution)), int(info), calls["hvp"]


def _cg_solve_torch(
    q: torch.Tensor,
    target: torch.Tensor,
    energy,
    rhs: torch.Tensor,
    damping: float,
    rtol: float,
    maxiter: int,
):
    """Device-native CG for the Hessian-vector-product KKT solve.

    The CPU path intentionally retains SciPy's mature stopping rule.  This
    implementation keeps every vector and HVP on the input device, which is
    required for a genuine CUDA scaling audit rather than a GPU forward pass
    followed by host-side linear algebra.
    """

    if rhs.ndim != 1 or q.ndim != 1 or rhs.numel() != q.numel():
        raise ValueError("q and rhs must be matching one-dimensional vectors")
    calls = 0

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return _hvp(q, target, energy, vector) + damping * vector

    solution = torch.zeros_like(rhs)
    residual = rhs - matvec(solution)
    rhs_norm = float(torch.linalg.norm(rhs)) if hasattr(torch, "linalg") else float(torch.norm(rhs))
    threshold = rtol * max(rhs_norm, 1.0)
    residual_norm = float(torch.linalg.norm(residual)) if hasattr(torch, "linalg") else float(torch.norm(residual))
    if residual_norm <= threshold:
        return solution, 0, calls
    direction = residual.clone()
    rr = torch.dot(residual, residual)
    for iteration in range(1, maxiter + 1):
        h_direction = matvec(direction)
        denominator = torch.dot(direction, h_direction)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            return solution, iteration, calls
        alpha = rr / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * h_direction
        residual_norm = float(torch.linalg.norm(residual)) if hasattr(torch, "linalg") else float(torch.norm(residual))
        if residual_norm <= threshold:
            return solution, 0, calls
        new_rr = torch.dot(residual, residual)
        direction = residual + (new_rr / rr) * direction
        rr = new_rr
    return solution, maxiter, calls


def _solve_matrixfree(
    q0: torch.Tensor,
    target: torch.Tensor,
    energy,
    max_iter: int = 12,
    cg_maxiter: int = 120,
    cg_rtol: float = 1e-8,
):
    q = q0.detach().clone()
    total_hvp = 0
    for iteration in range(max_iter):
        q_eval = q.detach().requires_grad_(True)
        value = energy(q_eval, target)
        gradient = torch.autograd.grad(value, q_eval)[0].detach()
        grad_norm = float(torch.max(torch.abs(gradient)))
        if grad_norm < 2e-11:
            return q, grad_norm, iteration + 1, total_hvp
        step, info, hvp_count = _cg_solve(q, target, energy, -gradient, 1e-8, cg_rtol, cg_maxiter)
        total_hvp += hvp_count
        if info != 0:
            raise RuntimeError(f"Newton-CG failed with info={info}")
        directional = float(torch.dot(gradient, step))
        alpha = 1.0
        accepted = False
        for _ in range(24):
            trial = q + alpha * step
            trial_value = energy(trial, target)
            if torch.isfinite(trial_value) and float(trial_value) <= float(value) + 1e-4 * alpha * directional:
                q = trial.detach()
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise RuntimeError("matrix-free Newton line search failed")
    q_eval = q.detach().requires_grad_(True)
    gradient = torch.autograd.grad(energy(q_eval, target), q_eval)[0]
    return q, float(torch.max(torch.abs(gradient))), max_iter, total_hvp


def run(
    output_dir: Path,
    n: int = 64,
    beta: float = 1e-3,
    cg_maxiter: int = 120,
    device: str = "cpu",
    newton_maxiter: int = 12,
    cg_rtol: float = 1e-8,
    finite_difference_eps: float = 1e-3,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh, vertices, faces, interior, ncoord, unpack, energy = _problem(n, beta, device=device)
    target = _target(vertices, 0.08)[interior].reshape(-1)
    q0 = vertices[interior].reshape(-1)
    q_star, stationarity, newton_iterations, forward_hvp = _solve_matrixfree(
        q0, target, energy, max_iter=newton_maxiter, cg_maxiter=cg_maxiter, cg_rtol=cg_rtol
    )
    rng = np.random.default_rng(20260919)
    cotangent = torch.from_numpy(rng.normal(size=ncoord)).to(vertices.device)
    adjoint_started = time.perf_counter()
    adjoint, adjoint_info, adjoint_hvp = _cg_solve(
        q_star, target, energy, cotangent, 1e-8, cg_rtol, max(200, cg_maxiter)
    )
    adjoint_seconds = time.perf_counter() - adjoint_started
    direction = torch.from_numpy(np.random.default_rng(20260920).normal(size=ncoord)).to(vertices.device)
    direction /= torch.linalg.norm(direction)
    implicit_gradient = adjoint / ncoord
    eps = finite_difference_eps
    plus, plus_stationarity, plus_iterations, plus_hvp = _solve_matrixfree(
        q_star, target + eps * direction, energy, max_iter=newton_maxiter, cg_maxiter=cg_maxiter, cg_rtol=cg_rtol
    )
    minus, minus_stationarity, minus_iterations, minus_hvp = _solve_matrixfree(
        q_star, target - eps * direction, energy, max_iter=newton_maxiter, cg_maxiter=cg_maxiter, cg_rtol=cg_rtol
    )
    finite_directional = float(torch.dot(cotangent, (plus - minus) / (2.0 * eps)))
    implicit_directional = float(torch.dot(implicit_gradient, direction))
    full = unpack(q_star)
    tri = full[faces]
    det = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    result = {
        "grid": f"{n}x{n} cells",
        "device": str(vertices.device),
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "interior_coordinates": ncoord,
        "barrier_weight": beta,
        "cg_maxiter": cg_maxiter,
        "newton_maxiter": newton_maxiter,
        "cg_rtol": cg_rtol,
        "finite_difference_eps": finite_difference_eps,
        "forward_stationarity_inf": stationarity,
        "forward_newton_iterations": newton_iterations,
        "forward_hvp_calls": forward_hvp,
        "adjoint_cg_info": adjoint_info,
        "adjoint_hvp_calls": adjoint_hvp,
        "adjoint_seconds": adjoint_seconds,
        "finite_directional": finite_directional,
        "implicit_directional": implicit_directional,
        "implicit_directional_abs_error": abs(finite_directional - implicit_directional),
        "plus_stationarity_inf": plus_stationarity,
        "minus_stationarity_inf": minus_stationarity,
        "finite_difference_newton_iterations": plus_iterations + minus_iterations,
        "finite_difference_hvp_calls": plus_hvp + minus_hvp,
        "min_face_determinant": float(det.min()),
        "flipped_faces": int(torch.sum(det <= 0.0)),
        "finite": bool(torch.all(torch.isfinite(q_star)) and torch.all(torch.isfinite(adjoint))),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "matrix-free Newton-CG nonlinear log-det barrier optimum with matrix-free implicit adjoint",
        "interpretation": "the KKT backward uses Hessian-vector products and CG, avoiding dense Hessian materialization and unrolled solver state",
        "limitation": "active-set nonsmoothness, global rectangle theorem, and larger multi-GPU scaling remain open",
    }
    (output_dir / "barrier_kkt_matrixfree_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--cg-maxiter", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--newton-maxiter", type=int, default=12)
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--finite-difference-eps", type=float, default=1e-3)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.beta, args.cg_maxiter, args.device, args.newton_maxiter, args.cg_rtol, args.finite_difference_eps), indent=2))


if __name__ == "__main__":
    main()
