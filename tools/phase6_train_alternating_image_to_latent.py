"""Image-only training through exact alternating full-grid PL composition."""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch
import torch.nn.functional as F

from phase6_train_image_to_latent import _minimum_area_ratio, _synthetic_pair
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactAlternatingMonotoneComposition
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


class AlternatingImageEncoder(torch.nn.Module):
    """Shared image features with independent global/line logits per PL layer."""

    def __init__(self, side: int, axes: tuple[str, ...], width: int = 8) -> None:
        super().__init__()
        self.side = side
        self.coarse_side = min(side, max(3, (side - 1) // 8 + 1))
        self.axes = axes
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, width, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.GELU(),
        )
        self.line_heads = torch.nn.ModuleList(torch.nn.Conv2d(width, 1, 1) for _ in axes)
        self.global_heads = torch.nn.ModuleList(torch.nn.Conv1d(width, 1, 1) for _ in axes)

    def _heads(self, feature: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        side = feature.shape[-1]
        line = self.line_heads[index](feature)[:, 0]
        if self.axes[index] == "vertical":
            line_logits = line[:, :-1, :].transpose(1, 2).contiguous()
            global_logits = self.global_heads[index](feature.mean(dim=2))[:, 0, :side - 1]
        else:
            line_logits = line[:, :, :-1].contiguous()
            global_logits = self.global_heads[index](feature.mean(dim=3))[:, 0, :side - 1]
        return global_logits, line_logits

    def forward(self, pair: torch.Tensor) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], ...]:
        reduced = F.interpolate(pair, size=(self.side, self.side), mode="bilinear", align_corners=True)
        fine_feature = self.body(reduced)
        coarse_feature = F.interpolate(fine_feature, size=(self.coarse_side, self.coarse_side), mode="bilinear", align_corners=True)
        return tuple((self._heads(coarse_feature, i), self._heads(fine_feature, i)) for i in range(len(self.axes)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--layers", type=int, choices=(2, 4), default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.side < 3 or args.image_side < 2 or args.batch < 1 or args.steps < 1:
        raise ValueError("invalid side, image side, batch, or steps")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    fixed, moving, true_map = _synthetic_pair(args.image_side, args.batch, device)
    pair = torch.cat((fixed, moving), dim=1)
    axes = tuple("vertical" if layer_id % 2 == 0 else "horizontal" for layer_id in range(args.layers))
    encoder = AlternatingImageEncoder(args.side, axes).to(device)
    table = StructuredDenseQueryTable.from_mesh(structured_rectangle(args.side - 1, args.side - 1), height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    decoder = ExactAlternatingMonotoneComposition(args.side, table, axes)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)

    def evaluate() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        result = decoder(encoder(pair))
        warped = F.grid_sample(moving, 2.0 * result.dense - 1.0, mode="bilinear", padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), result.dense, result.controls

    with torch.no_grad():
        initial_loss = evaluate()[0].item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        loss, _, _ = evaluate()
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
        ended = time.perf_counter()
        records.append({"loss_before_update": loss.item(), "forward_seconds": middle - began, "backward_seconds": after_backward - middle, "optimizer_seconds": ended - after_backward})
    with torch.no_grad():
        final_loss, final_map, controls = evaluate()
        map_rmse = (final_map - true_map).square().mean().sqrt().item()
        area_ratios = [_minimum_area_ratio(control) for control in controls]
    print(json.dumps({
        "route": "A/B",
        "method": "exact_alternating_full_grid_monotone_composition",
        "representation": "exact_PL_composition_not_original_grid_P1",
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces_per_layer": 2 * (args.side - 1) ** 2,
        "latent_sides": [encoder.coarse_side, args.side],
        "layers": args.layers,
        "axes": axes,
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
        "minimum_layer_signed_area_ratios": area_ratios,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "step_records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
