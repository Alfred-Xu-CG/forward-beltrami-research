"""Implicit backward audit for a nonlinear log-determinant barrier optimum."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.mesh import structured_rectangle


torch.set_default_dtype(torch.float64)


def _problem(n: int, beta: float, device: str = "cpu"):
    mesh = structured_rectangle(n, n)
    torch_device = torch.device(device)
    vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).to(torch_device)
    faces = torch.from_numpy(np.array(mesh.faces, copy=True)).to(torch_device)
    boundary = (
        (vertices[:, 0] == 0.0)
        | (vertices[:, 0] == 1.0)
        | (vertices[:, 1] == 0.0)
        | (vertices[:, 1] == 1.0)
    )
    interior = torch.nonzero(~boundary, as_tuple=False).flatten()
    ncoord = int(interior.numel() * 2)

    def unpack(q: torch.Tensor) -> torch.Tensor:
        full = vertices.clone()
        full[interior] = q.reshape(-1, 2)
        return full

    def energy(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        full = unpack(q)
        tri = full[faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        displacement = q - target.reshape(-1)
        # Keep the line-search objective finite even when a trial step crosses
        # the barrier.  The quadratic exterior penalty points the optimizer
        # back toward det>0; the reported stationary point is required to be
        # strictly positive before its Hessian is used for the adjoint.
        safe_det = torch.clamp(det, min=1e-12)
        exterior = beta * 1e6 * torch.relu(1e-6 - det) ** 2
        return 0.5 * torch.mean(displacement * displacement) - beta * torch.mean(torch.log(safe_det)) + torch.mean(exterior)

    return mesh, vertices, faces, interior, ncoord, unpack, energy


def _target(vertices: torch.Tensor, amplitude: float) -> torch.Tensor:
    x, y = vertices[:, 0], vertices[:, 1]
    pi = math.pi
    return torch.stack(
        (
            x + amplitude * torch.sin(2.0 * pi * y) * torch.sin(pi * x),
            y + 0.7 * amplitude * torch.sin(2.0 * pi * x) * torch.sin(pi * y),
        ),
        dim=1,
    )


def _solve(q0: torch.Tensor, target: torch.Tensor, energy, max_iter: int = 24) -> tuple[torch.Tensor, float, int]:
    """Damped Newton solve of the nonlinear barrier stationarity equation."""
    q = q0.detach().clone()
    calls = 0
    for _ in range(max_iter):
        q_eval = q.detach().requires_grad_(True)
        value = energy(q_eval, target)
        grad = torch.autograd.grad(value, q_eval, create_graph=False)[0]
        grad_norm = float(torch.max(torch.abs(grad)))
        calls += 1
        if grad_norm < 1e-12:
            break
        hessian = torch.autograd.functional.hessian(lambda x: energy(x, target), q)
        hessian = 0.5 * (hessian + hessian.T)
        step = torch.linalg.solve(hessian + 1e-10 * torch.eye(q.numel()), -grad)
        directional = float(torch.dot(grad, step))
        accepted = False
        alpha = 1.0
        for _ in range(24):
            trial = q + alpha * step
            trial_value = energy(trial, target)
            if torch.isfinite(trial_value) and float(trial_value) <= float(value) + 1e-4 * alpha * directional:
                q = trial.detach()
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    q_eval = q.detach().requires_grad_(True)
    grad = torch.autograd.grad(energy(q_eval, target), q_eval)[0]
    return q.detach(), float(torch.max(torch.abs(grad))), calls


def run(output_dir: Path, n: int = 32, beta: float = 2e-3) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh, vertices, faces, interior, ncoord, unpack, energy = _problem(n, beta)
    base_target = _target(vertices, 0.12)[interior].reshape(-1)
    q_star, stationarity, forward_calls = _solve(vertices[interior].reshape(-1), base_target, energy)
    cotangent = torch.from_numpy(np.random.default_rng(20260919).normal(size=ncoord))
    grad_loss = cotangent
    hessian = torch.autograd.functional.hessian(lambda q: energy(q, base_target), q_star)
    hessian = 0.5 * (hessian + hessian.T)
    damping = 1e-9
    adjoint = torch.linalg.solve(hessian + damping * torch.eye(ncoord), grad_loss)
    implicit_target_grad = adjoint / ncoord
    direction = torch.from_numpy(np.random.default_rng(20260920).normal(size=ncoord))
    direction = direction / torch.linalg.norm(direction)
    eps = 1e-3
    plus, plus_stationarity, plus_calls = _solve(q_star, base_target + eps * direction, energy, max_iter=180)
    minus, minus_stationarity, minus_calls = _solve(q_star, base_target - eps * direction, energy, max_iter=180)
    finite_directional = float(torch.dot(cotangent, (plus - minus) / (2.0 * eps)))
    predicted_directional = float(torch.dot(implicit_target_grad, direction))
    finite_q_direction = (plus - minus) / (2.0 * eps)
    linear_q_direction = torch.linalg.solve(hessian + damping * torch.eye(ncoord), direction / ncoord)
    full = unpack(q_star)
    tri = full[faces]
    det = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "interior_coordinates": ncoord,
        "barrier_weight": beta,
        "forward_stationarity_inf": stationarity,
        "forward_lbfgs_calls": forward_calls,
        "implicit_hessian_min_eigenvalue": float(torch.linalg.eigvalsh(hessian).min()),
        "finite_directional": finite_directional,
        "implicit_directional": predicted_directional,
        "implicit_directional_abs_error": abs(finite_directional - predicted_directional),
        "q_direction_relative_error": float(torch.linalg.norm(finite_q_direction - linear_q_direction) / torch.linalg.norm(linear_q_direction)),
        "finite_difference_forward_calls": plus_calls + minus_calls,
        "plus_stationarity_inf": plus_stationarity,
        "minus_stationarity_inf": minus_stationarity,
        "min_face_determinant": float(det.min()),
        "flipped_faces": int(torch.sum(det <= 0.0)),
        "finite": bool(torch.all(torch.isfinite(q_star)) and torch.all(torch.isfinite(adjoint))),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "nonlinear log-det barrier stationary point with implicit Hessian/KKT-style adjoint",
        "interpretation": "the adjoint differentiates the converged nonlinear optimum without unrolling its LBFGS iterations",
        "limitation": "dense Hessian prototype through 48²; matrix-free larger-resolution KKT, active-set nonsmoothness, and global rectangle theorem remain open",
    }
    (output_dir / "barrier_kkt_implicit_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--beta", type=float, default=2e-3)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.beta), indent=2))


if __name__ == "__main__":
    main()
