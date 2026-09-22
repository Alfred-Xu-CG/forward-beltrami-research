"""Train an image encoder through exact local-patch PL composition.

The analytic map generates images and is used only for evaluation. Training
uses image mean-square error, with no displacement or landmark supervision.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch
import torch.nn.functional as F

from phase6_train_image_to_latent import _minimum_area_ratio, _synthetic_pair
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import LocalPatchComposition, LocalPatchMonotoneLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


class PatchImageEncoder(torch.nn.Module):
    """Map image pairs to one local monotone-spacing latent per layer."""

    def __init__(self, side: int, layers: list[LocalPatchMonotoneLayer], width: int = 8) -> None:
        super().__init__()
        self.side = side
        self.layers = layers
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, width, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.GELU(),
        )
        self.heads = torch.nn.ModuleList(torch.nn.Conv2d(width, 1, 1) for _ in layers)

    def forward(self, pair: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature = self.body(F.interpolate(pair, size=(self.side, self.side), mode="bilinear", align_corners=True))
        result = []
        for layer, head in zip(self.layers, self.heads):
            p = layer.patch_cells
            ox = int(layer._origins[0, 0].item())
            oy = int(layer._origins[0, 1].item())
            logits = head(feature)[:, :, oy : self.side - 1, ox : self.side - 1]
            patches = F.unfold(logits, kernel_size=p, stride=p)
            patches = patches.transpose(1, 2).reshape(pair.shape[0], layer.patch_count, p, p)
            if layer.axis == "vertical":
                result.append(patches[:, :, :, 1:].transpose(-1, -2).contiguous())
            else:
                result.append(patches[:, :, 1:, :].contiguous())
        return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--layers", type=int, choices=(2, 4), default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.side < args.patch_cells + 1 or args.patch_cells % 2 or args.steps < 1:
        raise ValueError("invalid side, patch side, or steps")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    fixed, moving, true_map = _synthetic_pair(args.image_side, args.batch, device)
    pair = torch.cat((fixed, moving), dim=1)
    layers = []
    for layer_id in range(args.layers):
        offset = args.patch_cells // 2 if layer_id % 2 else 0
        axis = "horizontal" if layer_id % 2 else "vertical"
        layers.append(LocalPatchMonotoneLayer(args.side, args.patch_cells, offset_x=offset, offset_y=offset, axis=axis).to(device))
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side,
        width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    decoder = LocalPatchComposition(layers, table)
    encoder = PatchImageEncoder(args.side, layers).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)

    def evaluate() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        decoded = decoder(encoder(pair))
        warped = F.grid_sample(moving, 2.0 * decoded.dense - 1.0, mode="bilinear", padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), decoded.dense, decoded.controls

    with torch.no_grad():
        initial_loss = evaluate()[0].item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        begin = time.perf_counter()
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
        end = time.perf_counter()
        records.append({"loss_before_update": loss.item(), "forward_seconds": middle - begin, "backward_seconds": after_backward - middle, "optimizer_seconds": end - after_backward})
    with torch.no_grad():
        final_loss, final_map, controls = evaluate()
        map_rmse = (final_map - true_map).square().mean().sqrt().item()
        min_ratios = [_minimum_area_ratio(control) for control in controls]
    print(json.dumps({
        "route": "B",
        "representation": "exact_PL_composition_not_P1_on_original_mesh",
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "patch_cells": args.patch_cells,
        "patch_counts": [layer.patch_count for layer in layers],
        "layers": args.layers,
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
        "minimum_layer_signed_area_ratios": min_ratios,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "step_records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
