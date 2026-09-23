"""Image-to-scalar spectral latent followed by a safe fixed-grid P1 refiner."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, ForwardP1Pyramid, ForwardPatchP1Pyramid,
    PatchPyramidImageEncoder, SafeColoredVertexRelaxation, SineModeP1Refiner,
    exact_dyadic_p1_refine, local_photometric_logits, spectralize_bounded_logits,
)
from qcopt.neural_bijection.dense.photometric_hint import physical_image_gradient
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]

    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    return float(torch.minimum(
        cross(b - a, c - a).amin(),
        cross(c - a, d - a).amin(),
    ) * (mapped.shape[1] - 1) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-kind", choices=("colored", "patch"), required=True)
    parser.add_argument("--target-family", choices=("high128", "high128_tri", "high128_tiles"),
                        default="high128")
    parser.add_argument("--amplitude-source", choices=("estimated", "true", "zero"),
                        default="estimated")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99317)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--check-vjp", action="store_true")
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--checkpoint-updates", action="store_true")
    parser.add_argument("--readout-iterations", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.readout_iterations < 1:
        raise ValueError("readout-iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    data = make_dataset(
        args.count, args.image_side, args.seed,
        target_family=args.target_family, return_coefficients=True,
    )
    if args.target_family == "high128":
        mode_cycles = ((128, 128),)
        mode_windows = None
    elif args.target_family == "high128_tri":
        mode_cycles = ((128, 128), (128, 64), (64, 128))
        mode_windows = None
    else:
        mode_cycles = ((128, 128),) * 16
        mode_windows = tuple(
            (column / 4, (column + 1) / 4, row / 4, (row + 1) / 4)
            for row in range(4) for column in range(4)
        )
    if args.decoder_kind == "colored":
        decoder = ForwardP1Pyramid(5, 257, seed_passes=2).to(device)
        encoder = ForwardP1ImageEncoder(
            5, decoder.level_sides, seed_passes=2, feature_side=257, width=16,
        ).to(device)
    else:
        decoder = ForwardPatchP1Pyramid(17, 257, patch_cells=8).to(device)
        encoder = PatchPyramidImageEncoder(
            17, decoder.level_sides, feature_side=257, width=16,
        ).to(device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    safe = SafeColoredVertexRelaxation(257, motion_mode="radial", raw_span=2).to(device)
    spectral = SineModeP1Refiner(
        257, 1025, cycles=mode_cycles, windows=mode_windows,
        mechanism=args.decoder_kind,
        sweeps=args.sweeps, checkpoint_updates=args.checkpoint_updates,
    ).to(device)
    table_coarse = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(256, 256), height=args.image_side, width=args.image_side,
    )
    table_fine = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=args.image_side, width=args.image_side,
    )
    for table in (table_coarse, table_fine):
        table.prepare(device=device, dtype=torch.float32)
    axis = torch.linspace(0, 1, args.image_side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)
    def mode_values(x: torch.Tensor, y: torch.Tensor,
                    kx: int, ky: int,
                    window: tuple[float, float, float, float] | None) -> torch.Tensor:
        value = torch.sin(2 * math.pi * kx * x) * torch.sin(2 * math.pi * ky * y)
        if window is not None:
            xlo, xhi, ylo, yhi = window
            tx = (x - xlo) / (xhi - xlo)
            ty = (y - ylo) / (yhi - ylo)
            value = value * torch.where((tx >= 0) & (tx <= 1),
                                        torch.sin(math.pi * tx).square(), 0.0)
            value = value * torch.where((ty >= 0) & (ty <= 1),
                                        torch.sin(math.pi * ty).square(), 0.0)
        return value

    image_modes = torch.stack(tuple(
        mode_values(xx, yy, kx, ky,
                    None if mode_windows is None else mode_windows[k])
        for k, (kx, ky) in enumerate(mode_cycles)
    ))
    image_mode_norm = image_modes.square().mean(dim=(1, 2))
    fine_axis = torch.arange(1025, device=device, dtype=torch.float64) / 1024
    fine_y, fine_x = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    fine_identity = torch.stack((fine_x, fine_y), dim=-1).to(torch.float32)
    low_vertex_mode = (
        torch.sin(2 * math.pi * fine_x) * torch.sin(2 * math.pi * fine_y)
    ).to(torch.float32)
    fine_modes = torch.stack(tuple(
        mode_values(fine_x, fine_y, kx, ky,
                    None if mode_windows is None else mode_windows[k])
        for k, (kx, ky) in enumerate(mode_cycles)
    )).to(torch.float32)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward_batch(fixed: torch.Tensor, moving: torch.Tensor,
                      coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seed_latent, levels = encoder(fixed, moving)
        control = decoder(seed_latent, levels)
        for _ in range(2):
            hint = local_photometric_logits(
                fixed, moving, control, window=3, ridge=1.0, raw_span=2.0,
            )
            hint = spectralize_bounded_logits(
                hint, side=257, raw_span=2.0, count=16,
            )
            floor = control.new_full((control.shape[0],), 0.05 / 256**2)
            control = safe(control, hint, area_floor=floor)
        coarse_query = table_coarse.interpolate(control.reshape(control.shape[0], -1, 2))
        coarse_query = coarse_query.reshape(-1, args.image_side, args.image_side, 2)
        moving_gradient = physical_image_gradient(moving)

        def estimate_at(query_points: torch.Tensor) -> torch.Tensor:
            grid = 2 * query_points - 1
            warped = F.grid_sample(moving, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            gradient = F.grid_sample(
                moving_gradient, grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            design = image_modes[None] * (gradient[:, None, 0] + gradient[:, None, 1])
            residual = (fixed - warped)[:, 0]
            pixels = args.image_side ** 2
            rhs = torch.einsum("bkhw,bhw->bk", design, residual) / pixels
            if args.target_family == "high128_tiles":
                # Compact interiors have disjoint support: diagonal Gram.
                return rhs / (design.square().mean(dim=(2, 3)) + 1e-12)
            gram = torch.einsum("bkhw,blhw->bkl", design, design) / pixels
            gram = gram + 1e-12 * torch.eye(
                len(mode_cycles), device=device, dtype=gram.dtype,
            )[None]
            return torch.linalg.solve(gram, rhs[..., None])[..., 0]

        estimated = estimate_at(coarse_query)
        if args.amplitude_source == "estimated":
            amplitude = estimated
            for _ in range(args.readout_iterations - 1):
                interim = spectral(control, amplitude)
                interim_query = table_fine.interpolate(
                    interim.reshape(interim.shape[0], -1, 2)
                ).reshape(-1, args.image_side, args.image_side, 2)
                amplitude = amplitude + estimate_at(interim_query)
            estimated = amplitude
        elif args.amplitude_source == "true":
            amplitude = coefficients[:, 2:]
        else:
            amplitude = estimated * 0
        mapped = spectral(control, amplitude)
        query = table_fine.interpolate(mapped.reshape(mapped.shape[0], -1, 2))
        query = query.reshape(-1, args.image_side, args.image_side, 2)
        predicted = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                                  padding_mode="border", align_corners=True)
        return mapped, query, predicted, estimated

    totals = dict(image=0.0, map=0.0, mode_error=0.0, mode_predicted=0.0,
                  mode_target=0.0, coefficient_error=0.0,
                  mode_p1_error=0.0, mode_p1_target=0.0)
    minimum = float("inf")
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            fixed, moving, target, coefficients = (
                x[start:start + args.batch].to(device) for x in data
            )
            sync()
            tick = time.perf_counter()
            mapped, query, predicted, estimated = forward_batch(
                fixed, moving, coefficients,
            )
            sync()
            times.append(time.perf_counter() - tick)
            n = fixed.shape[0]
            totals["image"] += float((predicted - fixed).square().mean()) * n
            totals["map"] += float((query - target).square().sum(dim=-1).mean()) * n
            predicted_mode = torch.einsum(
                "bhwd,khw->bkd", query - identity, image_modes,
            ) / (args.image_side ** 2 * image_mode_norm[None, :, None])
            target_mode = torch.einsum(
                "bhwd,khw->bkd", target - identity, image_modes,
            ) / (args.image_side ** 2 * image_mode_norm[None, :, None])
            target_vertices = fine_identity[None] + (
                coefficients[:, 0, None, None, None]
                * low_vertex_mode[None, :, :, None]
                * torch.tensor([1.0, 0.0], device=device)
                + coefficients[:, 1, None, None, None]
                * low_vertex_mode[None, :, :, None]
                * torch.tensor([0.0, 1.0], device=device)
                + torch.einsum(
                    "bk,khw,d->bhwd", coefficients[:, 2:], fine_modes,
                    fine_identity.new_ones((2,)),
                )
            )
            target_p1_query = table_fine.interpolate(
                target_vertices.reshape(n, -1, 2)
            ).reshape(n, args.image_side, args.image_side, 2)
            target_p1_mode = torch.einsum(
                "bhwd,khw->bkd", target_p1_query - identity, image_modes,
            ) / (args.image_side ** 2 * image_mode_norm[None, :, None])
            totals["mode_error"] += float(
                (predicted_mode - target_mode).square().sum()
            )
            totals["mode_p1_error"] += float(
                (predicted_mode - target_p1_mode).square().sum()
            )
            totals["mode_p1_target"] += float(
                target_p1_mode.square().sum()
            )
            totals["mode_predicted"] += float(predicted_mode.square().sum())
            totals["mode_target"] += float(target_mode.square().sum())
            totals["coefficient_error"] += float(
                (estimated - coefficients[:, 2:]).square().sum()
            )
            minimum = min(minimum, minimum_jacobian(mapped))
    forward_peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    forward_peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    vjp_finite = None
    vjp_max = None
    vjp_seconds = None
    vjp_peak_allocated = None
    if args.check_vjp:
        fixed, moving, _, coefficients = (x[:1].to(device) for x in data)
        encoder.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        vjp_tick = time.perf_counter()
        mapped, query, predicted, estimated = forward_batch(
            fixed, moving, coefficients,
        )
        loss = (predicted - fixed).square().mean()
        loss.backward()
        sync()
        vjp_seconds = time.perf_counter() - vjp_tick
        vjp_peak_allocated = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        vjp_finite = bool(grads and all(torch.isfinite(g).all() for g in grads))
        vjp_max = max(float(g.abs().amax()) for g in grads) if grads else None
    print(json.dumps({
        "method": "phase7_image_spectral_modes_to_safe_p1",
        "decoder_kind": args.decoder_kind,
        "target_family": args.target_family,
        "mode_cycles": mode_cycles,
        "compact_support_windows": mode_windows,
        "mode_count": len(mode_cycles),
        "amplitude_source": args.amplitude_source,
        "sweeps": args.sweeps,
        "checkpoint_updates": args.checkpoint_updates,
        "readout_iterations": args.readout_iterations,
        "checkpoint": args.checkpoint,
        "count": args.count,
        "seed": args.seed,
        "image_side": args.image_side,
        "control_side": 1025,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "batch": args.batch,
        "device": str(device),
        "image_mse": totals["image"] / args.count,
        "query_map_vector_rmse": math.sqrt(totals["map"] / args.count),
        "mode_coefficient_vector_rmse": math.sqrt(totals["mode_error"] / args.count),
        "mode_predicted_vector_rms": math.sqrt(totals["mode_predicted"] / args.count),
        "mode_target_vector_rms": math.sqrt(totals["mode_target"] / args.count),
        "mode_p1_target_vector_rms": math.sqrt(totals["mode_p1_target"] / args.count),
        "mode_p1_coefficient_vector_rmse": math.sqrt(totals["mode_p1_error"] / args.count),
        "estimated_coefficient_vector_rmse":
            math.sqrt(totals["coefficient_error"] / args.count),
        "minimum_jacobian": minimum,
        "median_forward_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": forward_peak_allocated,
        "peak_cuda_reserved_bytes": forward_peak_reserved,
        "encoder_vjp_finite": vjp_finite,
        "encoder_vjp_max_abs": vjp_max,
        "forward_and_vjp_seconds": vjp_seconds,
        "forward_and_vjp_peak_cuda_allocated_bytes": vjp_peak_allocated,
        "note": (
            "The sine basis is supplied to the architecture; this tests a targeted fine-mode latent,"
            " not general high-frequency inference. True amplitude is used only when amplitude-source=true."
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
