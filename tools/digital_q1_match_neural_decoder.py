"""Learn a spatially adaptive match-to-Q1 decoder without landmark labels.

This is an image-feature-to-map module: SuperGlue correspondences and the
saved DHR-derived initial affine are upstream, discrete, non-differentiable
inputs. The trainable part predicts multiscale spatial interpolation weights;
the existing F1 current-edge layer makes each map update topology-safe in
exact arithmetic. Image-level end-to-end differentiation is not claimed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, q1_dyadic_refine,
)
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_p1_layer import FactorizedP1Map, evaluate_factorized_p1


SIGMAS = (.04, .12, .28)


def gaussian_features(
    side: int, source: torch.Tensor, target: torch.Tensor,
    *, sigmas: tuple[float, ...] = SIGMAS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 3-by-2 motion candidates and their log support, B=1."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("matching Kx2 coordinates required")
    reference = identity_vertices(side, device=source.device).to(source.dtype)
    x = reference[0, 0, :, 0]
    y = reference[0, :, 0, 1]
    motion = target - source
    fields = []
    supports = []
    for sigma in sigmas:
        xw = torch.exp(-((x[:, None] - source[None, :, 0]).square()) /
                       (2 * sigma * sigma))
        yw = torch.exp(-((y[:, None] - source[None, :, 1]).square()) /
                       (2 * sigma * sigma))
        mass = yw @ xw.T
        numerator = torch.stack((
            (yw * motion[None, :, 0]) @ xw.T,
            (yw * motion[None, :, 1]) @ xw.T,
        ), dim=-1)
        fields.append(numerator / (mass[..., None] + 1e-6))
        supports.append(torch.log1p(mass))
    return torch.stack(fields, dim=0)[None], torch.stack(supports, dim=0)[None]


class MatchSpatialDecoder(nn.Module):
    """Spatial basis blender followed by several safe current-edge F1 levels."""

    def __init__(self, *, update_sides: tuple[int, ...] = (17, 33, 65),
                 final_side: int = 257, width: int = 32) -> None:
        super().__init__()
        if not update_sides or update_sides[0] != 17 or any(
            b != 2 * a - 1 for a, b in zip(update_sides, update_sides[1:])
        ):
            raise ValueError("update_sides must be a dyadic prefix from 17")
        side = update_sides[-1]
        while side < final_side:
            side = 2 * side - 1
        if side != final_side:
            raise ValueError("final_side must be dyadic after update levels")
        self.update_sides = update_sides
        self.final_side = final_side
        # 3*(motion x/y + log support) + reference xy + current displacement xy + h.
        self.net = nn.Sequential(
            nn.Conv2d(14, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 4, 1),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor([-2., 2., -2., 0.]))
        self.layers = nn.ModuleList(
            AdaptiveSoftRadialQ1Relaxation(
                side, raw_span=8., safety_fraction=.75, minimum_jacobian=.05,
            ) for side in update_sides
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("matching Kx2 coordinates required")
        current = identity_vertices(17, device=source.device).to(source.dtype)
        for index, (side, safe_layer) in enumerate(zip(self.update_sides, self.layers)):
            if index:
                current = q1_dyadic_refine(current)
            reference = identity_vertices(side, device=source.device).to(source.dtype)
            fields, supports = gaussian_features(side, source, target)
            # Include all three individual motion fields and support masses.
            motion_channels = fields.permute(0, 1, 4, 2, 3).reshape(1, 6, side, side)
            features = torch.cat((motion_channels, supports,
                                  reference.permute(0, 3, 1, 2),
                                  (current - reference).permute(0, 3, 1, 2),
                                  torch.full_like(supports[:, :1], 1 / (side - 1))), dim=1)
            control = self.net(features).permute(0, 2, 3, 1)
            weights = torch.softmax(control[..., :3], dim=-1)
            amplitude = 2 * torch.sigmoid(control[..., 3:4])
            proposal = amplitude * (weights[..., None] * fields.permute(0, 2, 3, 1, 4)).sum(-2)
            desired = reference + proposal
            displacement = (desired - current)[:, 1:-1, 1:-1]
            horizontal = (current[:, 1:-1, 2:] - current[:, 1:-1, :-2]) / 2
            vertical = (current[:, 2:, 1:-1] - current[:, :-2, 1:-1]) / 2
            determinant = horizontal[..., 0] * vertical[..., 1] - horizontal[..., 1] * vertical[..., 0]
            det_safe = determinant.clamp_min(1e-10)
            w_x = (vertical[..., 1] * displacement[..., 0] -
                   vertical[..., 0] * displacement[..., 1]) / (8 * det_safe)
            w_y = (-horizontal[..., 1] * displacement[..., 0] +
                   horizontal[..., 0] * displacement[..., 1]) / (8 * det_safe)
            latent = torch.atanh(torch.stack((w_x, w_y), dim=-1).clamp(-.95, .95))
            current = safe_layer(current, latent)
        while current.shape[1] < self.final_side:
            current = q1_dyadic_refine(current)
        return current


def split_cases(rows: list[dict], seed: int = 20260929) -> tuple[list[dict], list[dict]]:
    valid = [row for row in rows if row["status"] == "ok" and row["ransac_inliers"] >= 12]
    order = np.random.default_rng(seed).permutation(len(valid))
    count = int(.8 * len(order))
    return [valid[i] for i in order[:count]], [valid[i] for i in order[count:]]


def tensors(row: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(row["source_points_unit"], device=device, dtype=torch.float32),
            torch.tensor(row["target_points_unit"], device=device, dtype=torch.float32))


def split_matches(source: torch.Tensor, target: torch.Tensor, *, seed: int,
                  input_fraction: float = .7) -> tuple[torch.Tensor, ...]:
    if not 0 < input_fraction < 1 or len(source) < 3:
        raise ValueError("need >=3 points and nonempty input/target split")
    indices = torch.randperm(len(source), generator=torch.Generator().manual_seed(seed))
    count = min(len(source) - 1, max(2, int(len(source) * input_fraction)))
    used, held = indices[:count].to(source.device), indices[count:].to(source.device)
    return source[used], target[used], source[held], target[held]


def match_rmse(mapped: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(2, device=mapped.device, dtype=mapped.dtype)[None]
    offset = torch.zeros((1, 2), device=mapped.device, dtype=mapped.dtype)
    predicted = evaluate_factorized_p1(
        FactorizedP1Map(mapped, identity, offset), source[None],
    )[0]
    # A tiny positive offset avoids the singular sqrt derivative at exact fit;
    # it changes the reported nonzero RMSE below float32 measurement precision.
    return ((predicted - target).square().sum(-1).mean() + 1e-16).sqrt()


def train(report_path: Path, output_path: Path, *, steps: int, device_name: str,
          update_sides: tuple[int, ...], final_side: int, learning_rate: float) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train_rows, validation_rows = split_cases(report["rows"])
    device = torch.device(device_name)
    torch.manual_seed(20260929)
    rng = np.random.default_rng(20260930)
    model = MatchSpatialDecoder(update_sides=update_sides,
                                final_side=final_side).to(device)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=learning_rate, weight_decay=1e-4)
    cases = {row["case"]: tensors(row, device) for row in train_rows + validation_rows}
    losses = []
    began = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        row = train_rows[int(rng.integers(len(train_rows)))]
        source, target = cases[row["case"]]
        input_source, input_target, held_source, held_target = split_matches(
            source, target, seed=step + 290929,
            input_fraction=float(rng.uniform(.55, .85)),
        )
        optimizer.zero_grad(set_to_none=True)
        mapped = model(input_source, input_target)
        fit = match_rmse(mapped, held_source, held_target)
        reference = identity_vertices(final_side, device=device)
        prior = (mapped - reference).square().mean()
        loss = fit + .02 * prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            losses.append({"step": step, "holdout_match_rmse": float(fit.detach()),
                           "regularized_loss": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - began
    model.eval()
    validation = []
    with torch.no_grad():
        for row in validation_rows:
            source, target = cases[row["case"]]
            input_source, input_target, held_source, held_target = split_matches(
                source, target, seed=row["case"] + 470000, input_fraction=.7,
            )
            mapped = model(input_source, input_target)
            baseline = identity_vertices(final_side, device=device)
            validation.append({"case": row["case"],
                               "input_matches": len(input_source),
                               "held_matches": len(held_source),
                               "identity_rmse": float(match_rmse(baseline, held_source, held_target)),
                               "predicted_rmse": float(match_rmse(mapped, held_source, held_target))})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.net.state_dict(), "update_sides": update_sides,
                "final_side": final_side, "train_case_ids": [row["case"] for row in train_rows],
                "validation_case_ids": [row["case"] for row in validation_rows]}, output_path)
    result = {"method": "spatial multiband Gaussian blend + current-edge safe F1",
              "report_input": str(report_path), "checkpoint": str(output_path),
              "steps": steps, "train_cases": len(train_rows),
              "validation_cases": len(validation_rows), "update_sides": update_sides,
              "final_side": final_side, "train_seconds": train_seconds,
              "peak_torch_cuda_allocated_bytes": (None if device.type != "cuda" else
                                                   int(torch.cuda.max_memory_allocated(device))),
              "loss_trace": losses, "validation": validation}
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=.0005)
    parser.add_argument("--update-sides", type=int, nargs="+", default=[17, 33, 65])
    parser.add_argument("--final-side", type=int, default=257)
    args = parser.parse_args()
    result = train(args.matches, args.output, steps=args.steps,
                   device_name=args.device, update_sides=tuple(args.update_sides),
                   final_side=args.final_side, learning_rate=args.learning_rate)
    print(json.dumps({key: result[key] for key in ("steps", "train_cases", "validation_cases",
                                               "train_seconds", "peak_torch_cuda_allocated_bytes")}))


if __name__ == "__main__":
    main()
