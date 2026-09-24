"""Train a spatial image-to-fine-latent residual encoder on unknown P1 packets.

The decoder is the exact same certified 1025-control-vertex layer as in Phase VII.
The carrier is supplied, while packet position, envelope and vector are not.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_unknown_fine_packet_hint import make_dataset


def carrier(side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)


def demod_amplitude(fixed: torch.Tensor, moving: torch.Tensor,
                    image_carrier: torch.Tensor, window: int = 17,
                    ridge: float = 1.0) -> torch.Tensor:
    gradient = physical_image_gradient(moving)
    gx = image_carrier * gradient[:, 0:1]
    gy = image_carrier * gradient[:, 1:2]
    residual = fixed - moving

    def smooth(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, window, stride=1, padding=window // 2,
                            count_include_pad=False)

    a = smooth(gx * gx) + ridge
    b = smooth(gx * gy)
    d = smooth(gy * gy) + ridge
    rhs_x = smooth(gx * residual)
    rhs_y = smooth(gy * residual)
    determinant = a * d - b.square()
    return torch.cat(((d * rhs_x - b * rhs_y) / determinant,
                      (a * rhs_y - b * rhs_x) / determinant), dim=1)


class SpatialResidualEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, amplitude_in_units: torch.Tensor) -> torch.Tensor:
        return amplitude_in_units + self.net(amplitude_in_units)


def logits_from_amplitude(amplitude_128: torch.Tensor,
                          fine_carrier: torch.Tensor) -> torch.Tensor:
    amplitude = F.interpolate(amplitude_128, size=(1025, 1025),
                              mode="bilinear", align_corners=True)
    displacement = (1e-4 * amplitude * fine_carrier[None, None])
    ratio = (displacement / (2 / 1024)).clamp(-0.95, 0.95)
    return torch.atanh(ratio[:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()


def build_data(count: int, seed: int, batch: int, device: torch.device,
               image_carrier: torch.Tensor,
               p1_image_truth: bool) -> tuple[torch.Tensor, ...]:
    fixed, moving, target, support = make_dataset(
        count, 512, seed, device, p1_image_truth=p1_image_truth)
    with torch.no_grad():
        amplitudes = []
        for start in range(0, count, batch):
            amp = demod_amplitude(fixed[start:start + batch],
                                  moving[start:start + batch], image_carrier)
            amplitudes.append(F.interpolate(amp / 1e-4, size=(128, 128),
                                            mode="bilinear", align_corners=True))
    return fixed, moving, target, support, torch.cat(amplitudes)


def evaluate(model: nn.Module, data: tuple[torch.Tensor, ...],
             decoder: CoarsePatchFineVertexP1Layer,
             table: StructuredDenseQueryTable, fine_carrier: torch.Tensor,
             batch_size: int) -> dict[str, float | int]:
    fixed, moving, target, support, features = data
    identity = decoder.fine_identity
    total_map = total_support = total_image = 0.0
    support_n = 0
    minimum = math.inf
    fallback = 0
    times = []
    model.eval()
    for start in range(0, len(features), batch_size):
        fixed_b, moving_b, target_b, support_b, feature_b = (
            tensor[start:start + batch_size] for tensor in data)
        batch = feature_b.shape[0]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.no_grad():
            predicted_amp = model(feature_b)
            fine = logits_from_amplitude(predicted_amp, fine_carrier)
            coarse = torch.zeros(batch, 255, 255, 2, dtype=torch.float32,
                                 device=features.device)
            output = decoder(coarse, fine)
            query = table.interpolate(output.reshape(batch, -1, 2))
            warped = F.grid_sample(moving_b, 2 * query.float() - 1,
                                   align_corners=True, mode="bilinear",
                                   padding_mode="border")
            square = (output - target_b).square().sum(dim=-1)
            total_map += float(square.mean(dim=(1, 2)).sum())
            mask = support_b > 0
            total_support += float(square[mask].sum())
            support_n += int(mask.sum())
            total_image += float((warped - fixed_b).square().mean(
                dim=(1, 2, 3)).sum())
            minimum = min(minimum, minimum_jacobian(output))
            fallback += int(torch.all(output == identity,
                                      dim=(1, 2, 3)).sum())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
    return {
        "map_vector_rmse": math.sqrt(total_map / len(features)),
        "support_vector_rmse": math.sqrt(total_support / support_n),
        "image_mse": total_image / len(features),
        "minimum_jacobian": minimum,
        "identity_outputs": fallback,
        "median_eval_batch_seconds": sorted(times)[len(times) // 2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objective", choices=["map", "image"], default="map")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=31719)
    parser.add_argument("--test-seed", type=int, default=99113)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--p1-image-truth", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(21071)
    image_carrier = carrier(512, device, torch.float32)[None, None]
    fine_carrier = carrier(1025, device, torch.float32)
    train = build_data(args.train_count, args.train_seed, args.batch, device,
                       image_carrier, args.p1_image_truth)
    test = build_data(args.test_count, args.test_seed, args.batch, device,
                      image_carrier, args.p1_image_truth)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=0.05, certify_output=True,
    ).to(device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512,
    )
    table.prepare(device=device, dtype=torch.float64)
    model = SpatialResidualEncoder(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline = evaluate(model, test, decoder, table, fine_carrier, args.batch)
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    for step in range(args.steps):
        model.train()
        indices = torch.randint(args.train_count, (args.batch,), device=device)
        fixed, moving, target, _, features = (tensor[indices] for tensor in train)
        predicted_amp = model(features)
        fine = logits_from_amplitude(predicted_amp, fine_carrier)
        coarse = torch.zeros(args.batch, 255, 255, 2, dtype=torch.float32,
                             device=device)
        output = decoder(coarse, fine)
        if args.objective == "map":
            loss = 1e8 * (output - target).square().sum(dim=-1).mean()
        else:
            query = table.interpolate(output.reshape(args.batch, -1, 2))
            warped = F.grid_sample(moving, 2 * query.float() - 1,
                                   align_corners=True, mode="bilinear",
                                   padding_mode="border")
            loss = 1e6 * (warped - fixed).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "loss_last_100_mean": sum(losses[-100:]) / 100}),
                  flush=True)
    torch.cuda.synchronize()
    training_time = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated(device)
    final = evaluate(model, test, decoder, table, fine_carrier, args.batch)
    print(json.dumps({
        "experiment": "phase7_spatial_packet_encoder",
        "objective": args.objective,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": 512,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "batch": args.batch,
        "steps": args.steps,
        "width": args.width,
        "parameters": sum(p.numel() for p in model.parameters()),
        "lr": args.lr,
        "p1_image_truth": args.p1_image_truth,
        "baseline": baseline,
        "final": final,
        "mean_training_step_seconds": training_time / args.steps,
        "peak_cuda_allocated_bytes": peak,
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
