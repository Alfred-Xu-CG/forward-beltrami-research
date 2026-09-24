"""Local photometric refinement of a continuous packet frequency."""
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
from phase7_test_continuous_frequency import decode
from phase7_test_fft_carrier_inference import carrier_field, infer_cycles
from phase7_test_unknown_carrier_bank import dataset


def local_amplitude_and_score(
    fixed: torch.Tensor, moving: torch.Tensor,
    gradient: torch.Tensor, carrier: torch.Tensor,
    window: int = 17, ridge: float = 1.,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = fixed - moving
    gx = carrier[:, None] * gradient[:, :1]
    gy = carrier[:, None] * gradient[:, 1:2]

    def smooth(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x, window, stride=1, padding=window // 2,
            count_include_pad=False)

    a = smooth(gx.square()) + ridge
    b = smooth(gx * gy)
    d = smooth(gy.square()) + ridge
    bx = smooth(gx * residual)
    by = smooth(gy * residual)
    determinant = a * d - b.square()
    amp = torch.cat(((d * bx - b * by) / determinant,
                     (a * by - b * bx) / determinant), dim=1)
    prediction = gx * amp[:, :1] + gy * amp[:, 1:2]
    score = (residual * prediction -
             .5 * prediction.square()).mean(dim=(1, 2, 3))
    return amp, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=59473)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-cycles", type=float, nargs="+",
                        default=[80.5, 96.5, 112.5, 127.5])
    parser.add_argument("--radius", type=float, default=.5)
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[.05, .1, .2, .4])
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    fixed_all, moving_all, target_all, support_all, _, truth_all = dataset(
        args.count, args.seed, device, args.batch, table,
        tuple(args.target_cycles))
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True).to(device)
    offsets = torch.arange(
        -args.radius, args.radius + .5 * args.step,
        args.step, device=device, dtype=torch.float32)
    methods = ["fft_mean", "photo_hard", "photo_parabola"] + [
        f"photo_soft_{t}" for t in args.temperatures]
    accum = {
        name: dict(freq_abs=0., freq_sq=0., map=0., support=0.,
                   support_n=0, image=0., minimum=math.inf, fallback=0)
        for name in methods
    }
    score_peak_deltas = []
    times = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            stop = min(start + args.batch, args.count)
            fixed, moving, target, support, truth = (
                x[start:stop] for x in
                (fixed_all, moving_all, target_all, support_all, truth_all))
            batch = len(fixed)
            _, fft_scores = infer_cycles(fixed, moving, 80, 128, radius=2)
            p = torch.softmax(
                torch.log(fft_scores.clamp_min(1e-30)) / .1, dim=1)
            center = (
                p * torch.arange(80, 129, device=device)[None]).sum(dim=1)
            gradient = physical_image_gradient(moving)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            scores = []
            for offset in offsets:
                trial = center + offset
                h = carrier_field(trial, 512, device)
                _, score = local_amplitude_and_score(
                    fixed, moving, gradient, h)
                scores.append(score)
            scores = torch.stack(scores, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
            peak = scores.argmax(dim=1)
            chosen = center + offsets[peak]
            left = scores.gather(
                1, (peak - 1).clamp_min(0)[:, None]).squeeze(1)
            center_score = scores.gather(
                1, peak[:, None]).squeeze(1)
            right = scores.gather(
                1, (peak + 1).clamp_max(len(offsets) - 1)[:, None]
            ).squeeze(1)
            curvature = left - 2 * center_score + right
            substep = torch.where(
                curvature.abs() > 1e-20,
                .5 * (left - right) / curvature,
                torch.zeros_like(curvature)).clamp(-.5, .5)
            parabola = chosen + substep * args.step
            score_peak_deltas.extend(
                (chosen - truth.float()).abs().cpu().tolist())
            normalized = (
                scores - scores.amax(dim=1, keepdim=True)) / (
                    scores.std(dim=1, keepdim=True) + 1e-30)
            candidates = {
                "fft_mean": center,
                "photo_hard": chosen,
                "photo_parabola": parabola,
            }
            for temperature in args.temperatures:
                weights = torch.softmax(
                    normalized / temperature, dim=1)
                candidates[f"photo_soft_{temperature}"] = (
                    center + (weights * offsets[None]).sum(dim=1))
            for name, estimated in candidates.items():
                row = accum[name]
                error = estimated - truth.float()
                row["freq_abs"] += float(error.abs().sum())
                row["freq_sq"] += float(error.square().sum())
                h_image = carrier_field(estimated, 512, device)
                h_fine = carrier_field(estimated, 1025, device)
                output = decode(
                    h_image, h_fine, fixed, moving, decoder)
                square = (output - target).square().sum(dim=-1)
                row["map"] += float(square.mean(dim=(1, 2)).sum())
                mask = support > 0
                row["support"] += float(square[mask].sum())
                row["support_n"] += int(mask.sum())
                query = table.interpolate(
                    output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1,
                    mode="bilinear", padding_mode="border",
                    align_corners=True)
                row["image"] += float(
                    (warped - fixed).square().mean(dim=(1, 2, 3)).sum())
                row["minimum"] = min(
                    row["minimum"], minimum_jacobian(output))
                row["fallback"] += int(torch.all(
                    output == decoder.fine_identity,
                    dim=(1, 2, 3)).sum())
    rows = []
    for name, row in accum.items():
        rows.append(dict(
            method=name,
            frequency_mae=row["freq_abs"] / args.count,
            frequency_rmse=math.sqrt(row["freq_sq"] / args.count),
            map_vector_rmse=math.sqrt(row["map"] / args.count),
            support_vector_rmse=math.sqrt(
                row["support"] / row["support_n"]),
            image_mse=row["image"] / args.count,
            minimum_jacobian=row["minimum"],
            identity_outputs=row["fallback"]))
    print(json.dumps(dict(
        experiment="phase7_continuous_frequency_photometric_refinement",
        control_vertices=1025 ** 2,
        count=args.count,
        seed=args.seed,
        batch=args.batch,
        target_cycles=args.target_cycles,
        offsets=[float(offsets[0]), float(offsets[-1])],
        step=args.step,
        candidates_per_sample=len(offsets),
        mean_refinement_seconds_per_batch=sum(times) / len(times),
        peak_frequency_error_max=max(score_peak_deltas),
        rows=rows,
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
