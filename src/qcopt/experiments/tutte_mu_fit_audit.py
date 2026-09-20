"""Direct Beltrami-target fitting with the positive directed Tutte layer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.beltrami import face_beltrami
from qcopt.forward.tutte_directed_implicit import DirectedTutteSystem, directed_tutte_embedding_torch_implicit
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _target_map(mesh):
    target = mesh.vertices.copy()
    x, y = target[:, 0], target[:, 1]
    target[:, 0] += 0.12 * np.sin(np.pi * x) ** 2 * np.sin(np.pi * y)
    target[:, 1] += 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y) ** 2
    return target


def _torch_face_mu(mapped: torch.Tensor, mesh) -> torch.Tensor:
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long, device=mapped.device)
    gradients = torch.as_tensor(np.array(mesh.gradients, copy=True), dtype=mapped.dtype, device=mapped.device)
    local = mapped[faces]
    jac = torch.einsum("fki,fkj->fij", local, gradients)
    ux, uy = jac[:, 0, 0], jac[:, 0, 1]
    vx, vy = jac[:, 1, 0], jac[:, 1, 1]
    fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
    fbar = 0.5 * ((ux - vy) + 1j * (vx + uy))
    return fbar / fz


def run(output_dir: Path, n: int = 128, iterations: int = 20, learning_rate: float = 0.25) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    mesh = structured_rectangle(n, n)
    truth = _target_map(mesh)
    target_mu_np = face_beltrami(mesh, truth)
    target_mu = torch.as_tensor(target_mu_np, dtype=torch.complex128)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.as_tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    history = []
    started = time.perf_counter()
    first_loss = None
    first_mu_loss = None
    finite_grad = True
    for step in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        mapped = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
        induced = _torch_face_mu(mapped, mesh)
        mu_loss = torch.mean(torch.abs(induced - target_mu) ** 2)
        map_loss = torch.mean((mapped - torch.as_tensor(truth)) ** 2)
        loss = mu_loss
        if first_loss is None:
            first_loss = float(loss.detach())
            first_mu_loss = float(mu_loss.detach())
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            finite_grad = False
            break
        optimizer.step()
        with torch.no_grad():
            logits.clamp_(-6.0, 6.0)
        history.append(
            {
                "step": step,
                "mu_mse": float(mu_loss.detach()),
                "map_mse": float(map_loss.detach()),
                "gradient_l2": float(torch.linalg.vector_norm(logits.grad).detach()),
            }
        )
    with torch.no_grad():
        final = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
        final_mu = _torch_face_mu(final, mesh)
        final_mu_mse = float(torch.mean(torch.abs(final_mu - target_mu) ** 2))
        final_map_mse = float(torch.mean((final - torch.as_tensor(truth)) ** 2))
    mapped_np = final.detach().cpu().numpy()
    report = audit_injectivity(mesh, mapped_np, rectangle=True)
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "interior_rows": int(system.n_rows),
        "iterations_requested": iterations,
        "iterations_completed": len(history),
        "learning_rate": learning_rate,
        "target_mu_rms": float(np.sqrt(np.mean(np.abs(target_mu_np) ** 2))),
        "initial_mu_mse": first_mu_loss,
        "final_mu_mse": final_mu_mse,
        "mu_mse_ratio": float(final_mu_mse / max(first_mu_loss or 1.0, 1e-30)),
        "final_map_mse": final_map_mse,
        "finite_gradient": finite_grad,
        "flipped_faces": int(len(report.flipped_faces)),
        "minimum_signed_area_ratio": float(report.minimum_signed_area_ratio),
        "injectivity_certified": bool(report.certified),
        "boundary_orientation_ok": bool(report.boundary_orientation_ok),
        "boundary_intersections": int(len(report.boundary_intersections)),
        "bad_branch_vertices": int(len(report.bad_branch_vertices)),
        "history_tail": history[-5:],
        "scope": "direct induced-Beltrami fitting through a positive directed implicit Tutte decoder",
        "interpretation": "topology is hard-certified by the positive-row decoder while mu fidelity is an independent learned objective",
        "limitation": "finite Adam optimization on one manufactured field; not arbitrary-mu expressivity or a global optimization theorem",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "tutte_mu_fit_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.iterations, args.learning_rate), indent=2))


if __name__ == "__main__":
    main()
