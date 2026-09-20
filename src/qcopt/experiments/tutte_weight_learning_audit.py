"""Positive-edge-weight Tutte expressivity toy using the implicit edge VJP."""

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


def run(output_dir: Path, n: int = 32, iterations: int = 60) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    target_boundary = mesh.vertices[mesh.boundary_loops[0]].copy()
    truth = mesh.vertices.copy()
    x, y = truth[:, 0], truth[:, 1]
    truth[:, 0] += 0.12 * np.sin(np.pi * x) ** 2 * np.sin(np.pi * y)
    truth[:, 1] += 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y) ** 2
    edges = _edge_list(mesh)
    log_weights = np.zeros(len(edges), dtype=np.float64)
    first_loss = None
    m = np.zeros_like(log_weights)
    v = np.zeros_like(log_weights)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    history = []
    for step in range(1, iterations + 1):
        weights = {edge: float(np.exp(value)) for edge, value in zip(edges, log_weights)}
        current = tutte_embedding_weighted(mesh, target_boundary, weights)
        decoded, edge_grad = weighted_tutte_vjp(
            mesh, target_boundary, weights, (current - truth) / mesh.n_vertices
        )
        # Reuse the explicit decode only for the loss gradient's map value;
        # weighted_tutte_vjp returns the same map and its edge VJP.
        loss = 0.5 * float(np.mean((decoded - truth) ** 2))
        if first_loss is None:
            first_loss = loss
        gradient = np.asarray([edge_grad[edge] * weights[edge] for edge in edges])
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * gradient * gradient
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        log_weights -= 0.15 * m_hat / (np.sqrt(v_hat) + eps)
        log_weights = np.clip(log_weights, -4.0, 4.0)
        history.append({"step": step, "loss": loss, "min_weight": float(np.min(np.exp(log_weights))), "max_weight": float(np.max(np.exp(log_weights)))})
    final_weights = {edge: float(np.exp(value)) for edge, value in zip(edges, log_weights)}
    final = tutte_embedding_weighted(mesh, target_boundary, final_weights)
    p0, p1, p2 = (final[mesh.faces[:, i]] for i in range(3))
    det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "edges": len(edges),
        "iterations": iterations,
        "initial_loss": first_loss,
        "final_loss": float(0.5 * np.mean((final - truth) ** 2)),
        "loss_ratio": float((0.5 * np.mean((final - truth) ** 2)) / first_loss),
        "min_determinant": float(np.min(det)),
        "flipped_faces": int(np.sum(det <= 0.0)),
        "history_tail": history[-5:],
        "scope": "positive edge-weight Tutte expressivity optimization driven by implicit VJP",
        "limitation": "small/medium optimization toy; positivity gives the decoder guarantee, not arbitrary-mu representation",
    }
    (output_dir / "tutte_weight_learning_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.iterations), indent=2))


if __name__ == "__main__":
    main()
