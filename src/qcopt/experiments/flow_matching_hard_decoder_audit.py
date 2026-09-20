"""Conditional flow-matching toy with a hard monotone/shear decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.coupled_decoder import coupled_monotone_shear_map
from qcopt.mesh import structured_rectangle


def _target_latent(n_intervals: int, batch: int, device: torch.device) -> torch.Tensor:
    phase = torch.rand(batch, 1, device=device) * 2.0 * np.pi
    grid = torch.linspace(0.0, 1.0, n_intervals, device=device)[None, :]
    x = 0.35 * torch.sin(2.0 * np.pi * grid + phase) + 0.12 * torch.cos(6.0 * np.pi * grid - phase)
    y = 0.28 * torch.cos(2.0 * np.pi * grid - phase) + 0.10 * torch.sin(4.0 * np.pi * grid + phase)
    shear = torch.cat((0.24 * torch.sin(phase), 0.18 * torch.cos(phase)), dim=1)
    return torch.cat((x, y, shear), dim=1)


def _decode(latent: torch.Tensor, points: np.ndarray) -> np.ndarray:
    n = (latent.shape[1] - 2) // 2
    x = torch.nn.functional.softplus(latent[:, :n]).detach().cpu().numpy()[0] + 1e-5
    y = torch.nn.functional.softplus(latent[:, n : 2 * n]).detach().cpu().numpy()[0] + 1e-5
    alpha, beta = latent[0, -2:].detach().cpu().numpy().tolist()
    return coupled_monotone_shear_map(points, x, y, float(alpha), float(beta))


def _face_dets(mesh, values):
    p0, p1, p2 = (values[mesh.faces[:, i]] for i in range(3))
    return (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])


def run(output_dir: Path, n_intervals: int = 64, steps: int = 600, integration_steps: int = 32) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260919)
    device = torch.device("cpu")
    dim = 2 * n_intervals + 2
    model = torch.nn.Sequential(
        torch.nn.Linear(2 * dim + 1, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        batch_size = 128
        condition = _target_latent(n_intervals, batch_size, device)
        base = torch.randn_like(condition)
        t = torch.rand(batch_size, 1, device=device)
        state = (1.0 - t) * base + t * condition
        velocity = condition - base
        prediction = model(torch.cat((state, condition, t), dim=1))
        loss = torch.mean((prediction - velocity) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    condition = _target_latent(n_intervals, 1, device)
    state = torch.zeros_like(condition)
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(integration_steps):
            t = torch.full((1, 1), step / integration_steps, device=device)
            velocity = model(torch.cat((state, condition, t), dim=1))
            state = state + velocity / integration_steps
    integration_seconds = time.perf_counter() - t0
    latent_error = float(torch.mean((state - condition) ** 2).sqrt())

    mesh = structured_rectangle(512, 512)
    mapped = _decode(state, mesh.vertices)
    det = _face_dets(mesh, mapped)
    result = {
        "latent_intervals": n_intervals,
        "training_steps": steps,
        "integration_steps": integration_steps,
        "initial_loss": losses[0],
        "final_training_loss": losses[-1],
        "held_out_latent_rmse": latent_error,
        "integration_seconds": integration_seconds,
        "decoder_grid": "512x512",
        "decoder_faces": int(mesh.n_faces),
        "minimum_determinant": float(np.min(det)),
        "flipped_faces": int(np.sum(det <= 0.0)),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(det))),
        "scope": "conditional flow-matching latent integration with positive-increment/shear hard decoder",
        "limitation": "toy conditional distribution, affine coupling only, and no arbitrary-Beltrami or diffusion-model claim",
    }
    (output_dir / "flow_matching_hard_decoder_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--integration-steps", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.intervals, args.steps, args.integration_steps), indent=2))


if __name__ == "__main__":
    main()
