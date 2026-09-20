"""Determinant-aware trust-region optimization of positive Tutte weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.tutte import tutte_embedding_weighted
from qcopt.forward.tutte_weighted_implicit import weighted_tutte_vjp
from qcopt.mesh import structured_rectangle


def _edge_list(mesh):
    edges = set()
    for face in mesh.faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def _truth(mesh):
    truth = mesh.vertices.copy()
    x, y = truth[:, 0], truth[:, 1]
    truth[:, 0] += 0.12 * np.sin(np.pi * x) ** 2 * np.sin(np.pi * y)
    truth[:, 1] += 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y) ** 2
    return truth


def _determinants(mesh, values: np.ndarray) -> np.ndarray:
    p0, p1, p2 = (values[mesh.faces[:, i]] for i in range(3))
    return (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])


def run(
    output_dir: Path,
    n: int = 256,
    iterations: int = 12,
    initial_step: float = 0.5,
    determinant_margin: float = 1e-8,
    max_backtracks: int = 8,
) -> dict:
    if iterations < 1 or initial_step <= 0.0 or determinant_margin <= 0.0:
        raise ValueError("iterations, initial_step, and determinant_margin must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    boundary = mesh.vertices[mesh.boundary_loops[0]].copy()
    target = _truth(mesh)
    edges = _edge_list(mesh)
    log_weights = np.zeros(len(edges), dtype=np.float64)
    records = []
    first_loss = None
    for step in range(1, iterations + 1):
        weights = {edge: float(np.exp(value)) for edge, value in zip(edges, log_weights)}
        current, edge_grad = weighted_tutte_vjp(mesh, boundary, weights, (tutte_embedding_weighted(mesh, boundary, weights) - target) / mesh.n_vertices)
        current_loss = 0.5 * float(np.mean((current - target) ** 2))
        det = _determinants(mesh, current)
        if first_loss is None:
            first_loss = current_loss
        gradient = np.asarray([edge_grad[edge] * weights[edge] for edge in edges], dtype=np.float64)
        scale = max(float(np.max(np.abs(gradient))), 1e-12)
        direction = -gradient / scale
        accepted = False
        accepted_alpha = 0.0
        candidate_loss = current_loss
        candidate_min_det = float(np.min(det))
        for backtrack in range(max_backtracks + 1):
            alpha = initial_step * (0.5 ** backtrack)
            candidate_log = np.clip(log_weights + alpha * direction, -4.0, 4.0)
            candidate_weights = {edge: float(np.exp(value)) for edge, value in zip(edges, candidate_log)}
            candidate = tutte_embedding_weighted(mesh, boundary, candidate_weights)
            candidate_det = _determinants(mesh, candidate)
            trial_loss = 0.5 * float(np.mean((candidate - target) ** 2))
            if trial_loss < current_loss and float(np.min(candidate_det)) >= determinant_margin:
                log_weights = candidate_log
                accepted = True
                accepted_alpha = alpha
                candidate_loss = trial_loss
                candidate_min_det = float(np.min(candidate_det))
                break
        records.append(
            {
                "step": step,
                "loss_before": current_loss,
                "loss_after": candidate_loss,
                "accepted": accepted,
                "alpha": accepted_alpha,
                "backtracks": backtrack if accepted else max_backtracks + 1,
                "min_determinant_before": float(np.min(det)),
                "min_determinant_after": candidate_min_det,
            }
        )
    final_weights = {edge: float(np.exp(value)) for edge, value in zip(edges, log_weights)}
    final = tutte_embedding_weighted(mesh, boundary, final_weights)
    final_det = _determinants(mesh, final)
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "vertices": mesh.n_vertices,
        "edges": len(edges),
        "iterations": iterations,
        "initial_step": initial_step,
        "determinant_margin": determinant_margin,
        "initial_loss": first_loss,
        "final_loss": float(0.5 * np.mean((final - target) ** 2)),
        "loss_ratio": float((0.5 * np.mean((final - target) ** 2)) / max(first_loss, 1e-30)),
        "min_determinant": float(np.min(final_det)),
        "flipped_faces": int(np.sum(final_det <= 0.0)),
        "accepted_steps": int(sum(record["accepted"] for record in records)),
        "monotone_accepted_losses": bool(all(record["loss_after"] <= record["loss_before"] for record in records if record["accepted"])),
        "history": records,
        "scope": "positive-log-weight Tutte optimization with determinant-aware backtracking",
        "limitation": "This is a hard-positive decoder optimizer, not a proof of arbitrary-Beltrami expressivity or a globally optimal weight-learning method.",
    }
    (output_dir / "tutte_weight_learning_safe_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--initial-step", type=float, default=0.5)
    parser.add_argument("--determinant-margin", type=float, default=1e-8)
    parser.add_argument("--max-backtracks", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.iterations, args.initial_step, args.determinant_margin, args.max_backtracks), indent=2))


if __name__ == "__main__":
    main()
