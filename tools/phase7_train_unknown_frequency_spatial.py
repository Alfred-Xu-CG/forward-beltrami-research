"""Train a spatial encoder with unknown integer carrier and certified 1025² P1 output.

A fixed FFT/top-k preprocessor chooses the carrier; the trainable CNN predicts
the spatial amplitude. Gradients to the CNN pass through the full P1 decoder.
The hard carrier choice itself is not differentiable.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_fft_carrier_inference import carrier_field, infer_cycles
from phase7_test_unknown_carrier_bank import dataset
from phase7_train_spatial_packet_images import (
    SpatialResidualEncoder, demod_amplitude,
)


def choose_frequency(fixed: torch.Tensor, moving: torch.Tensor,
                     low: int, high: int, topk: int,
                     ) -> torch.Tensor:
    _, fft_scores = infer_cycles(fixed, moving, low, high, radius=2)
    candidates = fft_scores.topk(min(topk, high - low + 1),
                                 dim=1).indices + low
    gradient = physical_image_gradient(moving)
    residual = fixed - moving
    scores = []
    for frequency in candidates.unbind(dim=1):
        h = carrier_field(frequency, 512, fixed.device)[:, None]
        amplitude = demod_amplitude(fixed, moving, h)
        predicted = h * (
            gradient[:, :1] * amplitude[:, :1] +
            gradient[:, 1:2] * amplitude[:, 1:2])
        scores.append((residual * predicted - .5 * predicted.square()
                       ).mean(dim=(1, 2, 3)))
    winner = torch.stack(scores, dim=1).argmax(dim=1)
    return candidates.gather(1, winner[:, None]).squeeze(1)


def build_data(count: int, seed: int, batch: int, device: torch.device,
               table: StructuredDenseQueryTable,
               low: int, high: int, topk: int) -> tuple[torch.Tensor, ...]:
    fixed, moving, target, support, _, truth = dataset(
        count, seed, device, batch, table, tuple(range(low, high + 1)))
    feature_parts, carrier_parts, estimate_parts = [], [], []
    with torch.no_grad():
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            fixed_b, moving_b = fixed[start:stop], moving[start:stop]
            estimate = choose_frequency(
                fixed_b, moving_b, low, high, topk)
            h = carrier_field(estimate, 512, device)[:, None]
            amplitude = demod_amplitude(fixed_b, moving_b, h)
            feature_parts.append(F.interpolate(
                amplitude / 1e-4, size=(128, 128),
                mode="bilinear", align_corners=True))
            carrier_parts.append(carrier_field(estimate, 1025, device))
            estimate_parts.append(estimate)
    return (fixed, moving, target, support,
            torch.cat(feature_parts), torch.cat(carrier_parts),
            torch.cat(estimate_parts), truth)


def decode(model: SpatialResidualEncoder,
           decoder: CoarsePatchFineVertexP1Layer,
           features: torch.Tensor,
           carriers: torch.Tensor) -> torch.Tensor:
    amplitude = model(features)
    amplitude = F.interpolate(amplitude, size=(1025, 1025),
                              mode="bilinear", align_corners=True)
    displacement = 1e-4 * amplitude * carriers[:, None]
    latent = torch.atanh(
        (displacement / (2 / 1024)).clamp(-.95, .95)
        [:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()
    coarse = torch.zeros(len(features), 255, 255, 2,
                         device=features.device, dtype=torch.float32)
    return decoder(coarse, latent)


def evaluate(model: SpatialResidualEncoder, data: tuple[torch.Tensor, ...],
             decoder: CoarsePatchFineVertexP1Layer,
             table: StructuredDenseQueryTable, batch_size: int):
    fixed, moving, target, support, features, carriers, estimates, truth = data
    total_map = total_support = total_image = 0.
    support_count = 0
    minimum = math.inf
    fallback = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            output = decode(
                model, decoder, features[start:stop],
                carriers[start:stop])
            query = table.interpolate(output.reshape(stop - start, -1, 2))
            warped = F.grid_sample(
                moving[start:stop], 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            square = (output - target[start:stop]).square().sum(dim=-1)
            total_map += float(square.mean(dim=(1, 2)).sum())
            mask = support[start:stop] > 0
            total_support += float(square[mask].sum())
            support_count += int(mask.sum())
            total_image += float((warped - fixed[start:stop]).square().mean(
                dim=(1, 2, 3)).sum())
            minimum = min(minimum, minimum_jacobian(output))
            fallback += int(torch.all(
                output == decoder.fine_identity, dim=(1, 2, 3)).sum())
    return {
        "map_vector_rmse": math.sqrt(total_map / len(features)),
        "support_vector_rmse": math.sqrt(total_support / support_count),
        "image_mse": total_image / len(features),
        "minimum_jacobian": minimum,
        "identity_outputs": fallback,
        "frequency_correct": int((estimates == truth).sum()),
        "frequency_total": len(estimates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objective", choices=["map", "image"], default="map")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=31719)
    parser.add_argument("--test-seed", type=int, default=99113)
    parser.add_argument("--low", type=int, default=80)
    parser.add_argument("--high", type=int, default=128)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    torch.manual_seed(21071)
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    train = build_data(
        args.train_count, args.train_seed, args.batch, device,
        table, args.low, args.high, args.topk)
    test = build_data(
        args.test_count, args.test_seed, args.batch, device,
        table, args.low, args.high, args.topk)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True,
    ).to(device)
    model = SpatialResidualEncoder(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline = evaluate(model, test, decoder, table, args.batch)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    losses = []
    for step in range(args.steps):
        model.train()
        indices = torch.randint(
            args.train_count, (args.batch,), device=device)
        fixed, moving, target, _, features, carriers, _, _ = (
            tensor[indices] for tensor in train)
        output = decode(model, decoder, features, carriers)
        if args.objective == "map":
            loss = 1e8 * (output - target).square().sum(dim=-1).mean()
        else:
            query = table.interpolate(output.reshape(args.batch, -1, 2))
            warped = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            loss = 1e6 * (warped - fixed).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({
                "step": step + 1,
                "loss_last_100_mean": sum(losses[-100:]) / 100,
            }), flush=True)
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated(device)
    final = evaluate(model, test, decoder, table, args.batch)
    print(json.dumps({
        "experiment": "phase7_unknown_frequency_spatial_training",
        "objective": args.objective,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": 512,
        "frequency_range": [args.low, args.high],
        "topk": args.topk,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "batch": args.batch,
        "steps": args.steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "lr": args.lr,
        "baseline": baseline,
        "final": final,
        "mean_training_step_seconds": duration / args.steps,
        "peak_cuda_allocated_bytes": peak,
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
