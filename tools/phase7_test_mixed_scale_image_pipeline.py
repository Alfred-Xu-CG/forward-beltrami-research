"""Mixed low-amplitude-scale and high-frequency P1 image inference diagnostic.

An independent low local patch and high continuous carrier packet are added.
The decoder first performs two image-conditioned 257² safe patch updates,
exactly refines that P1 map to 1025², then performs one safe fine vertex pass.
Oracle-low and image-estimated-low branches isolate inference bottlenecks.
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
    ResidualStaggeredPatchP1Layer, SafeColoredVertexRelaxation,
    certify_p1_or_identity, exact_dyadic_p1_refine,
    local_photometric_logits, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_refine_continuous_frequency import local_amplitude_and_score
from phase7_test_fft_carrier_inference import carrier_field
from phase7_matched_low_patch import (
    low_map_from_params, matched_low_patch,
    multistart_refine_low_params,
)
from phase7_test_unknown_carrier_bank import dataset as high_dataset


def mixed_dataset(
    count: int, seed: int, device: torch.device,
    batch: int, table: StructuredDenseQueryTable,
    frequencies: tuple[float, ...],
    *, return_params: bool = False,
    low_width: float = 1 / 16,
) -> tuple[torch.Tensor, ...]:
    if not (0 < low_width < .25):
        raise ValueError("low_width must lie in (0, .25)")
    _, moving, high_target, high_support, _, true_k = high_dataset(
        count, seed, device, batch, table, frequencies)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100000)
    origin_span = .78 if low_width == 1 / 16 else .84 - low_width
    origin = .08 + origin_span * torch.rand(
        count, 2, generator=generator)
    magnitude = .003 + .003 * torch.rand(
        count, generator=generator)
    angle = 2 * math.pi * torch.rand(
        count, generator=generator)
    vector = torch.stack((
        magnitude * torch.cos(angle),
        magnitude * torch.sin(angle)), dim=-1).to(
            device=device, dtype=torch.float64)

    def low_map(side: int) -> tuple[torch.Tensor, torch.Tensor]:
        axis = torch.arange(
            side, device=device, dtype=torch.float64) / (side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        ox = origin[:, 0].to(device=device, dtype=torch.float64)[:, None, None]
        oy = origin[:, 1].to(device=device, dtype=torch.float64)[:, None, None]
        tx, ty = (xx - ox) / low_width, (yy - oy) / low_width
        wx = torch.where(
            (tx >= 0) & (tx <= 1),
            torch.sin(math.pi * tx).square(), 0)
        wy = torch.where(
            (ty >= 0) & (ty <= 1),
            torch.sin(math.pi * ty).square(), 0)
        window = wx * wy
        identity = torch.stack((xx, yy), dim=-1)[None]
        return identity + window[..., None] * vector[:, None, None], window

    low_fine, low_support = low_map(1025)
    low_coarse, _ = low_map(257)
    fine_axis = torch.arange(
        1025, device=device, dtype=torch.float64) / 1024
    yy, xx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    fine_identity = torch.stack((xx, yy), dim=-1)[None]
    total = low_fine + high_target - fine_identity
    fixed_parts = []
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        query = table.interpolate(
            total[start:stop].reshape(stop - start, -1, 2))
        fixed_parts.append(F.grid_sample(
            moving[start:stop], 2 * query.float() - 1,
            mode="bilinear", padding_mode="border",
            align_corners=True))
    fixed = torch.cat(fixed_parts, dim=0)
    result = (fixed, moving, total, low_coarse,
              low_fine, low_support, high_support, true_k)
    return (result + (origin.to(device), vector)
            if return_params else result)


def fft_scores(
    residual: torch.Tensor, gradient: torch.Tensor,
    low: int = 80, high: int = 128,
) -> torch.Tensor:
    spectrum = torch.fft.rfft2(
        residual * gradient, norm="ortho")
    power = spectrum.abs().square().sum(dim=1)
    side = residual.shape[-1]
    radius = 2
    scores = []
    for cycle in range(low, high + 1):
        positive = power[
            :, cycle-radius:cycle+radius+1,
            cycle-radius:cycle+radius+1].sum(dim=(1, 2))
        negative = power[
            :, side-cycle-radius:side-cycle+radius+1,
            cycle-radius:cycle+radius+1].sum(dim=(1, 2))
        scores.append(positive + negative)
    return torch.stack(scores, dim=1)


def photo_high(
    fixed: torch.Tensor, moving: torch.Tensor,
    coarse_fine: torch.Tensor,
    table: StructuredDenseQueryTable,
    offsets: torch.Tensor,
    *, return_amplitude: bool = False,
) -> tuple[torch.Tensor, ...]:
    batch = len(fixed)
    base_query = table.interpolate(
        coarse_fine.reshape(batch, -1, 2))
    grid = 2 * base_query.float() - 1
    moved = F.grid_sample(
        moving, grid, mode="bilinear",
        padding_mode="border", align_corners=True)
    gradient = F.grid_sample(
        physical_image_gradient(moving), grid,
        mode="bilinear", padding_mode="border",
        align_corners=True)
    residual = fixed - moved
    scores = fft_scores(residual, gradient)
    posterior = torch.softmax(
        torch.log(scores.clamp_min(1e-30)) / .1, dim=1)
    frequency_axis = torch.arange(
        80, 129, device=fixed.device, dtype=torch.float32)
    center = (posterior * frequency_axis[None]).sum(dim=1)
    photo_scores = []
    for offset in offsets:
        carrier = carrier_field(
            center + offset, 512, fixed.device)
        _, score = local_amplitude_and_score(
            fixed, moved, gradient, carrier)
        photo_scores.append(score)
    photo_scores = torch.stack(photo_scores, dim=1)
    normalized = (
        photo_scores - photo_scores.amax(
            dim=1, keepdim=True)) / (
                photo_scores.std(
                    dim=1, keepdim=True) + 1e-30)
    weights = torch.softmax(normalized / .05, dim=1)
    estimated_k = center + (
        weights * offsets[None]).sum(dim=1)
    h_image = carrier_field(
        estimated_k, 512, fixed.device)
    amplitude_image, _ = local_amplitude_and_score(
        fixed, moved, gradient, h_image)
    amplitude = F.interpolate(
        amplitude_image, size=(1025, 1025),
        mode="bilinear", align_corners=True)
    high = amplitude * carrier_field(
        estimated_k, 1025, fixed.device)[:, None]
    return ((high, estimated_k, amplitude_image)
            if return_amplitude else (high, estimated_k))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=59473)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--low-directions", type=int, default=4)
    parser.add_argument("--low-steps", type=int, default=4)
    parser.add_argument("--high-cycles", type=float, nargs="+",
                        default=[80.5, 96.5, 112.5, 127.5])
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    data = mixed_dataset(
        args.count, args.seed, device, args.batch,
        table, tuple(args.high_cycles))
    coarse_layer = ResidualStaggeredPatchP1Layer(
        257, patch_cells=16, cycles=2,
        minimum_jacobian=.05).to(device)
    fine_layer = SafeColoredVertexRelaxation(
        1025, safety_fraction=.85,
        motion_mode="radial", raw_span=2.,
        floor_fraction=0.).to(device)
    coarse_axis = torch.arange(
        257, device=device, dtype=torch.float64) / 256
    cy, cx = torch.meshgrid(
        coarse_axis, coarse_axis, indexing="ij")
    coarse_identity = torch.stack((cx, cy), dim=-1)[None]
    fine_axis = torch.arange(
        1025, device=device, dtype=torch.float64) / 1024
    fy, fx = torch.meshgrid(
        fine_axis, fine_axis, indexing="ij")
    fine_identity = torch.stack((fx, fy), dim=-1)[None]
    offsets = torch.arange(
        -.5, .525, .05, device=device,
        dtype=torch.float32)
    names = (
        "identity", "oracle_low_only", "estimated_low_only",
        "oracle_low_photo_high", "estimated_low_photo_high",
        "estimated_low_oracle_residual", "matched_low_only",
        "matched_multi_low_only", "matched_multi_low_photo_high",
        "matched_multi_low_oracle_residual")
    metrics = {
        name: dict(map=0., low_support=0.,
                   high_support=0., low_n=0,
                   high_n=0, image=0.,
                   min_j=math.inf, accepted=0,
                   freq_abs=0.)
        for name in names
    }
    target_min = math.inf
    oracle_low_error = 0.
    times = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            stop = min(start + args.batch, args.count)
            (fixed, moving, target, low_coarse,
             low_fine, low_support, high_support, truth) = (
                 item[start:stop] for item in data)
            batch = len(fixed)
            target_min = min(
                target_min, minimum_jacobian(target))
            oracle_base = exact_dyadic_p1_refine(
                exact_dyadic_p1_refine(low_coarse))
            oracle_low_error += float(
                (oracle_base - low_fine).square().sum(
                    dim=-1).mean(dim=(1, 2)).sum())
            current = coarse_identity.expand(
                batch, -1, -1, -1)
            for _ in range(2):
                hint = local_photometric_logits(
                    fixed, moving, current.float(),
                    window=7, ridge=1.,
                    raw_span=8.)
                current = coarse_layer(
                    current, .75 * hint.double())
                current, _ = certify_p1_or_identity(
                    current, coarse_identity)
            estimated_base = exact_dyadic_p1_refine(
                exact_dyadic_p1_refine(current))
            matched_origin, matched_vector, _ = matched_low_patch(
                fixed, moving)
            gn_origin, gn_vector, _ = multistart_refine_low_params(
                fixed, moving, matched_origin,
                matched_vector, steps=args.low_steps,
                directions=args.low_directions)

            def safe_template_base(
                origin: torch.Tensor, vector: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                proposed = low_map_from_params(
                    origin, vector, 257, torch.float64)
                latent = torch.atanh((
                    (proposed - coarse_identity)[:, 1:-1, 1:-1] /
                    (8 / 256)).clamp(-.95, .95))
                coarse = coarse_layer(
                    coarse_identity.expand(batch, -1, -1, -1),
                    latent)
                coarse, valid = certify_p1_or_identity(
                    coarse, coarse_identity)
                return exact_dyadic_p1_refine(
                    exact_dyadic_p1_refine(coarse)), valid

            matched_base, valid_matched = safe_template_base(
                matched_origin, matched_vector)
            gn_base, valid_gn = safe_template_base(
                gn_origin, gn_vector)
            floor = estimated_base.new_full(
                (batch,), .05 / 1024 ** 2)

            def fine_update(
                base: torch.Tensor,
                displacement: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                latent = torch.atanh(
                    (displacement / (2 / 1024)).clamp(
                        -.95, .95)[:, :, 1:-1, 1:-1]
                ).permute(0, 2, 3, 1).contiguous()
                output = fine_layer(
                    base, latent.to(torch.float64),
                    area_floor=floor)
                return certify_p1_or_identity(
                    output, fine_identity)

            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            high_oracle_low, oracle_k = photo_high(
                fixed, moving, oracle_base,
                table, offsets)
            high_estimated_low, estimated_k = photo_high(
                fixed, moving, estimated_base,
                table, offsets)
            high_gn_low, gn_k = photo_high(
                fixed, moving, gn_base,
                table, offsets)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
            oracle_photo, valid_oracle = fine_update(
                oracle_base, high_oracle_low)
            estimated_photo, valid_estimated = fine_update(
                estimated_base, high_estimated_low)
            estimated_oracle, valid_estimated_oracle = fine_update(
                estimated_base,
                (target - estimated_base).permute(
                    0, 3, 1, 2))
            gn_photo, valid_gn_photo = fine_update(
                gn_base, high_gn_low)
            gn_oracle, valid_gn_oracle = fine_update(
                gn_base, (target - gn_base).permute(
                    0, 3, 1, 2))
            outputs = {
                "identity": (
                    fine_identity.expand(batch, -1, -1, -1),
                    torch.ones(batch, dtype=torch.bool, device=device),
                    None),
                "oracle_low_only": (
                    oracle_base,
                    torch.ones(batch, dtype=torch.bool, device=device),
                    None),
                "estimated_low_only": (
                    estimated_base,
                    torch.ones(batch, dtype=torch.bool, device=device),
                    None),
                "oracle_low_photo_high": (
                    oracle_photo, valid_oracle, oracle_k),
                "estimated_low_photo_high": (
                    estimated_photo, valid_estimated,
                    estimated_k),
                "estimated_low_oracle_residual": (
                    estimated_oracle, valid_estimated_oracle,
                    None),
                "matched_low_only": (
                    matched_base, valid_matched, None),
                "matched_multi_low_only": (
                    gn_base, valid_gn, None),
                "matched_multi_low_photo_high": (
                    gn_photo, valid_gn_photo, gn_k),
                "matched_multi_low_oracle_residual": (
                    gn_oracle, valid_gn_oracle, None),
            }
            for name, (output, valid, frequency) in outputs.items():
                row = metrics[name]
                square = (output - target).square().sum(dim=-1)
                row["map"] += float(square.mean(
                    dim=(1, 2)).sum())
                low_mask = low_support > 0
                high_mask = high_support > 0
                row["low_support"] += float(
                    square[low_mask].sum())
                row["high_support"] += float(
                    square[high_mask].sum())
                row["low_n"] += int(low_mask.sum())
                row["high_n"] += int(high_mask.sum())
                query = table.interpolate(
                    output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1,
                    mode="bilinear", padding_mode="border",
                    align_corners=True)
                row["image"] += float(
                    (warped - fixed).square().mean(
                        dim=(1, 2, 3)).sum())
                row["min_j"] = min(
                    row["min_j"], minimum_jacobian(output))
                row["accepted"] += int(valid.sum())
                if frequency is not None:
                    row["freq_abs"] += float(
                        (frequency - truth).abs().sum())
    rows = []
    for name, row in metrics.items():
        rows.append(dict(
            method=name,
            map_vector_rmse=math.sqrt(
                row["map"] / args.count),
            low_support_vector_rmse=math.sqrt(
                row["low_support"] / row["low_n"]),
            high_support_vector_rmse=math.sqrt(
                row["high_support"] / row["high_n"]),
            image_mse=row["image"] / args.count,
            frequency_mae=(
                row["freq_abs"] / args.count
                if name.endswith("photo_high") else None),
            minimum_jacobian=row["min_j"],
            certificate_passed=row["accepted"],
        ))
    print(json.dumps(dict(
        experiment="phase7_mixed_scale_forward_image_pipeline",
        control_vertices=1025 ** 2,
        control_faces=2 * 1024 ** 2,
        coarse_vertices=257 ** 2,
        count=args.count,
        batch=args.batch,
        seed=args.seed,
        high_cycles=args.high_cycles,
        low_directions=args.low_directions,
        low_steps=args.low_steps,
        target_minimum_jacobian=target_min,
        oracle_coarse_interpolation_rmse=math.sqrt(
            oracle_low_error / args.count),
        mean_three_frequency_inferences_seconds_per_batch=(
            sum(times) / len(times)),
        rows=rows,
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
