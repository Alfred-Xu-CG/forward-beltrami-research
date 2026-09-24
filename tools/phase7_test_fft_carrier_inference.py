"""Infer an unknown integer packet carrier using image residual FFT, then decode P1."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_unknown_carrier_bank import dataset
from phase7_train_spatial_packet_images import demod_amplitude


def infer_cycles(fixed: torch.Tensor, moving: torch.Tensor,
                 low: int, high: int, radius: int,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Score diagonal carrier bins in FFT of residual times both gradients."""
    gradient = physical_image_gradient(moving)
    residual = fixed - moving
    features = residual * gradient
    spectrum = torch.fft.rfft2(features, norm="ortho")
    power = spectrum.abs().square().sum(dim=1)
    side = fixed.shape[-1]
    scores = []
    for cycle in range(low, high + 1):
        positive = power[:, cycle - radius:cycle + radius + 1,
                         cycle - radius:cycle + radius + 1].sum(dim=(1, 2))
        negative = power[:, side - cycle - radius:side - cycle + radius + 1,
                         cycle - radius:cycle + radius + 1].sum(dim=(1, 2))
        scores.append(positive + negative)
    scores = torch.stack(scores, dim=1)
    return scores.argmax(dim=1) + low, scores


def carrier_field(cycles: torch.Tensor, side: int,
                  device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    frequency = cycles.float()[:, None, None]
    return (torch.sin(2 * math.pi * frequency * xx) *
            torch.sin(2 * math.pi * frequency * yy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=70123)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--low", type=int, default=80)
    parser.add_argument("--high", type=int, default=128)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--ridge", type=float, default=1.)
    parser.add_argument("--refine-topk", type=int, default=5)
    parser.add_argument("--fixed-noise-sigma", type=float, default=0.)
    parser.add_argument("--amplitude-threshold", type=float, default=0.)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.low <= args.high < 256 - args.radius:
        raise ValueError("FFT range must fit unique positive frequencies")
    if not math.isfinite(args.fixed_noise_sigma) or args.fixed_noise_sigma < 0:
        raise ValueError("fixed-noise-sigma must be nonnegative and finite")
    if not math.isfinite(args.amplitude_threshold) or args.amplitude_threshold < 0:
        raise ValueError("amplitude-threshold must be nonnegative and finite")
    if args.refine_topk < 1:
        raise ValueError("refine-topk must be positive")
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    target_cycles = tuple(range(args.low, args.high + 1))
    fixed_all, moving_all, target_all, support_all, _, cycles_all = dataset(
        args.count, args.seed, device, args.batch, table, target_cycles)
    if args.fixed_noise_sigma:
        noise_generator = torch.Generator(device=device).manual_seed(
            args.seed + 900001)
        fixed_all = fixed_all + args.fixed_noise_sigma * torch.randn(
            fixed_all.shape, device=device, dtype=fixed_all.dtype,
            generator=noise_generator)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True).to(device)
    identity = decoder.fine_identity
    count_correct = 0
    frequency_abs_error = 0.
    frequency_sq_error = 0.
    hist = {}
    refined_correct = 0
    refined_error_hist = {}
    true_rank_hist = {}
    fft_times = []
    refinement_times = []
    method_times = {name: [] for name in (
        "identity", "teacher", "known_frequency", "fft_frequency",
        "refined_frequency")}
    results = {
        name: {"map": 0., "support": 0., "image": 0., "support_n": 0,
               "minimum": math.inf, "fallback": 0}
        for name in ("identity", "teacher", "known_frequency",
                     "fft_frequency", "refined_frequency")
    }
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            fixed, moving, target, support, true_cycles = (
                tensor[start:start + args.batch] for tensor in
                (fixed_all, moving_all, target_all, support_all, cycles_all))
            batch = len(fixed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fft_start = time.perf_counter()
            estimate, fft_scores = infer_cycles(
                fixed, moving, args.low, args.high, args.radius)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fft_times.append(time.perf_counter() - fft_start)
            refine_start = time.perf_counter()
            true_score = fft_scores.gather(
                1, (true_cycles - args.low)[:, None])
            ranks = (fft_scores > true_score).sum(dim=1) + 1
            for rank in ranks.cpu().tolist():
                true_rank_hist[str(rank)] = true_rank_hist.get(str(rank), 0) + 1
            top = fft_scores.topk(min(args.refine_topk,
                                       args.high - args.low + 1), dim=1).indices
            candidate_cycles = top + args.low
            gradient = physical_image_gradient(moving)
            residual = fixed - moving
            refined_scores = []
            for candidate in candidate_cycles.unbind(dim=1):
                img_h = carrier_field(candidate, 512, device)[:, None]
                amp = demod_amplitude(
                    fixed, moving, img_h, window=args.window,
                    ridge=args.ridge)
                prediction = img_h * (
                    gradient[:, :1] * amp[:, :1] +
                    gradient[:, 1:2] * amp[:, 1:2])
                score = (residual * prediction -
                         .5 * prediction.square()).mean(dim=(1, 2, 3))
                refined_scores.append(score)
            refined_index = torch.stack(refined_scores, dim=1).argmax(dim=1)
            refined = candidate_cycles.gather(
                1, refined_index[:, None]).squeeze(1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            refinement_times.append(time.perf_counter() - refine_start)
            errors = (estimate - true_cycles).cpu().tolist()
            for error in errors:
                count_correct += int(error == 0)
                frequency_abs_error += abs(error)
                frequency_sq_error += error * error
                hist[str(error)] = hist.get(str(error), 0) + 1
            for error in (refined - true_cycles).cpu().tolist():
                refined_correct += int(error == 0)
                refined_error_hist[str(error)] = (
                    refined_error_hist.get(str(error), 0) + 1)
            coarse = torch.zeros(batch, 255, 255, 2,
                                 device=device, dtype=torch.float32)
            for name, row in results.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                method_start = time.perf_counter()
                if name == "identity":
                    latent = torch.zeros(batch, 1023, 1023, 2,
                                         device=device, dtype=torch.float32)
                elif name == "teacher":
                    latent = torch.atanh(
                        ((target - identity)[:, 1:-1, 1:-1] /
                         (2 / 1024)).clamp(-.95, .95)).float()
                else:
                    cycles = (true_cycles if name == "known_frequency" else
                              refined if name == "refined_frequency" else
                              estimate)
                    image_carrier = carrier_field(cycles, 512, device)
                    amplitude = demod_amplitude(
                        fixed, moving, image_carrier[:, None],
                        window=args.window, ridge=args.ridge)
                    if args.amplitude_threshold:
                        norm = torch.linalg.vector_norm(
                            amplitude, dim=1, keepdim=True)
                        shrink = (1 - args.amplitude_threshold /
                                  norm.clamp_min(1e-12)).clamp_min(0)
                        amplitude = amplitude * shrink
                    amplitude = F.interpolate(
                        amplitude, size=(1025, 1025),
                        mode="bilinear", align_corners=True)
                    fine_carrier = carrier_field(cycles, 1025, device)
                    proposal = amplitude * fine_carrier[:, None]
                    latent = torch.atanh(
                        (proposal / (2 / 1024)).clamp(-.95, .95)
                        [:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()
                output = decoder(coarse, latent)
                query = table.interpolate(output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1,
                    align_corners=True, mode="bilinear",
                    padding_mode="border")
                square = (output - target).square().sum(dim=-1)
                row["map"] += float(square.mean(dim=(1, 2)).sum())
                mask = support > 0
                row["support"] += float(square[mask].sum())
                row["support_n"] += int(mask.sum())
                row["image"] += float((warped - fixed).square().mean(
                    dim=(1, 2, 3)).sum())
                row["minimum"] = min(row["minimum"],
                                     minimum_jacobian(output))
                row["fallback"] += int(torch.all(
                    output == identity, dim=(1, 2, 3)).sum())
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                method_times[name].append(time.perf_counter() - method_start)
    rows = []
    for name, row in results.items():
        rows.append({
            "method": name,
            "map_vector_rmse": math.sqrt(row["map"] / args.count),
            "support_vector_rmse": math.sqrt(
                row["support"] / row["support_n"]),
            "image_mse": row["image"] / args.count,
            "minimum_jacobian": row["minimum"],
            "identity_outputs": row["fallback"],
        })
    print(json.dumps({
        "experiment": "phase7_fft_unknown_packet_carrier",
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": 512,
        "count": args.count,
        "seed": args.seed,
        "batch": args.batch,
        "frequency_range": [args.low, args.high],
        "frequency_patch_radius": args.radius,
        "window": args.window,
        "ridge": args.ridge,
        "fixed_noise_sigma": args.fixed_noise_sigma,
        "amplitude_threshold": args.amplitude_threshold,
        "exact_frequency_accuracy": count_correct / args.count,
        "frequency_mae": frequency_abs_error / args.count,
        "frequency_rmse": math.sqrt(frequency_sq_error / args.count),
        "frequency_error_histogram": hist,
        "true_fft_rank_histogram": true_rank_hist,
        "refine_topk": args.refine_topk,
        "refined_frequency_accuracy": refined_correct / args.count,
        "refined_frequency_error_histogram": refined_error_hist,
        "rows": rows,
        "median_fft_batch_seconds": statistics.median(fft_times),
        "median_topk_refinement_batch_seconds": statistics.median(
            refinement_times),
        "median_method_batch_seconds": {
            name: statistics.median(times)
            for name, times in method_times.items()},
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
