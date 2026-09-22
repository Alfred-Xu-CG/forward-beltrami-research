"""Small end-to-end image-to-dense-latent training for the monotone layer.

The only optimized objective is image registration. The analytic target map is
used to create fixed images and later evaluate map error, never as a decoder
input or gradient target. The target has a horizontal displacement dependent
on y, so a single vertical monotone layer cannot represent it exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import torch
import torch.nn.functional as torch_functional

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import MultiscaleMonotoneGridLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


class SmallImageEncoder(torch.nn.Module):
    """Predict fine spacing logits from a fixed/moving image pair."""

    def __init__(self, side: int, width: int = 8) -> None:
        super().__init__()
        self.side = side
        self.coarse_side = min(side, max(3, (side - 1) // 8 + 1))
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, width, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.GELU(),
        )
        self.line_head = torch.nn.Conv2d(width, 1, 1)
        self.global_head = torch.nn.Conv1d(width, 1, 1)

    def _heads(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        side = feature.shape[-1]
        line = self.line_head(feature)[:, 0, :-1, :].transpose(1, 2).contiguous()
        global_logits = self.global_head(feature.mean(dim=2))[:, 0, : side - 1]
        return global_logits, line

    def forward(
        self, pair: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        reduced = torch_functional.interpolate(
            pair, size=(self.side, self.side), mode="bilinear", align_corners=True
        )
        feature = self.body(reduced)
        coarse_feature = torch_functional.interpolate(
            feature,
            size=(self.coarse_side, self.coarse_side),
            mode="bilinear",
            align_corners=True,
        )
        # Image axes are (row=y, column=x); decoder line axes are (column,interval-y).
        return self._heads(coarse_feature), self._heads(feature)


def _synthetic_pair(
    image_side: int,
    batch: int,
    device: torch.device,
    target_kind: str = "smooth",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_kind not in ("smooth", "high_frequency"):
        raise ValueError("target_kind must be smooth or high_frequency")
    line = torch.linspace(0.0, 1.0, image_side, device=device)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    moving_scalar = (
        0.30 * torch.sin(8 * math.pi * xx + 3 * math.pi * yy)
        + 0.25 * torch.cos(13 * math.pi * yy - 2 * math.pi * xx)
        + 0.35 * torch.exp(-((xx - 0.36) ** 2 + (yy - 0.43) ** 2) / 0.012)
        + 0.30 * torch.exp(-((xx - 0.73) ** 2 + (yy - 0.68) ** 2) / 0.008)
    )
    if target_kind == "high_frequency":
        moving_scalar = moving_scalar + 0.16 * torch.sin(22 * math.pi * xx + 4 * math.pi * yy) * torch.cos(18 * math.pi * yy - 3 * math.pi * xx)
    moving = moving_scalar[None, None].expand(batch, 1, -1, -1).contiguous()
    strengths = torch.linspace(0.7, 1.0, batch, device=device)[:, None, None]
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    fine_bump = (
        0.003 * torch.sin(16 * math.pi * xx) * torch.sin(16 * math.pi * yy)
        if target_kind == "high_frequency"
        else 0.0
    )
    true_map = torch.stack(
        (xx[None] + strengths * (0.03 * bump + fine_bump), yy[None] + strengths * (0.05 * bump + fine_bump)),
        dim=-1,
    )
    fixed = torch_functional.grid_sample(
        moving,
        2.0 * true_map - 1.0,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).detach()
    return fixed, moving, true_map


def _minimum_area_ratio(control: torch.Tensor) -> float:
    a = control[:, :-1, :-1]
    b = control[:, :-1, 1:]
    c = control[:, 1:, 1:]
    d = control[:, 1:, :-1]
    lower = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])
    upper = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (
        c[..., 1] - a[..., 1]
    ) * (d[..., 0] - a[..., 0])
    return min(lower.min().item(), upper.min().item()) * (control.shape[1] - 1) ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-kind", choices=("smooth", "high_frequency"), default="smooth")
    args = parser.parse_args()
    if args.side < 2 or args.image_side < 2 or args.batch < 1 or args.steps < 1:
        raise ValueError("side, image-side, batch, and steps must be positive")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    fixed, moving, true_map = _synthetic_pair(args.image_side, args.batch, device, args.target_kind)
    pair = torch.cat((fixed, moving), dim=1)
    encoder = SmallImageEncoder(args.side).to(device)
    decoder = MultiscaleMonotoneGridLayer(args.side)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side,
        width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)

    def evaluate() -> tuple[torch.Tensor, torch.Tensor]:
        control = decoder(encoder(pair))
        queries = table.interpolate(control.reshape(args.batch, -1, 2))
        warped = torch_functional.grid_sample(
            moving,
            2.0 * queries - 1.0,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return (warped - fixed).square().mean(), control

    with torch.no_grad():
        initial_loss = evaluate()[0].item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    steps = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        begin = time.perf_counter()
        loss, _ = evaluate()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        middle = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_backward = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        steps.append(
            {
                "loss_before_update": loss.item(),
                "forward_seconds": middle - begin,
                "backward_seconds": after_backward - middle,
                "optimizer_seconds": end - after_backward,
            }
        )
    with torch.no_grad():
        final_loss, final_control = evaluate()
        control_line = torch.linspace(0.0, 1.0, args.side, device=device)
        cy, cx = torch.meshgrid(control_line, control_line, indexing="ij")
        strengths = torch.linspace(0.7, 1.0, args.batch, device=device)[:, None, None]
        bump = torch.sin(2 * math.pi * cx) * torch.sin(2 * math.pi * cy)
        fine_bump = (
            0.003 * torch.sin(16 * math.pi * cx) * torch.sin(16 * math.pi * cy)
            if args.target_kind == "high_frequency"
            else 0.0
        )
        true_control = torch.stack(
            (cx[None] + strengths * (0.03 * bump + fine_bump), cy[None] + strengths * (0.05 * bump + fine_bump)),
            dim=-1,
        )
        map_rmse = (final_control - true_control).square().mean().sqrt().item()
        minimum_ratio = _minimum_area_ratio(final_control)
        final_query = table.interpolate(final_control.reshape(args.batch, -1, 2))
        query_map_rmse = (final_query - true_map).square().mean().sqrt().item()
    payload = {
        "route": "A",
        "target_kind": args.target_kind,
        "latent_sides": [encoder.coarse_side, args.side],
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "initial_image_mse": initial_loss,
        "final_image_mse": final_loss.item(),
        "final_map_rmse": map_rmse,
        "final_query_map_rmse": query_map_rmse,
        "minimum_signed_area_ratio": minimum_ratio,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "step_records": steps,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
