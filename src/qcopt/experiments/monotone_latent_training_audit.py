"""Toy latent training audit for the positive-increment hard decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.monotone_torch import positive_increment_knots


def run(output_dir: Path, n_intervals: int = 128, steps: int = 600) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(91)
    target_increments = torch.exp(0.5 * torch.sin(torch.linspace(0.0, 4.0 * np.pi, n_intervals, dtype=torch.float64)))
    target = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(target_increments, dim=0)))
    target = target / target[-1]
    logits = torch.zeros(n_intervals, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.05)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        decoded = positive_increment_knots(logits)
        loss = torch.mean((decoded - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    decoded = positive_increment_knots(logits).detach()
    result = {
        "intervals": n_intervals,
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_decoded_increment": float(torch.min(torch.diff(decoded))),
        "max_knot_error": float(torch.max(torch.abs(decoded - target))),
        "finite_gradients": bool(torch.isfinite(logits.grad).all()),
    }
    (output_dir / "monotone_latent_training_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
