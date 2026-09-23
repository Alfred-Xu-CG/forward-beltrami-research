"""Image-only training of a tiny 16-mode gain through the 1025^2 safe P1 layer.

The coarse encoder is frozen; this diagnostic asks whether photometric
fine-tuning of the local coefficient readout improves true latent recovery.
No target map or true coefficient enters the optimization objective.
"""

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
    ForwardPatchP1Pyramid, PatchPyramidImageEncoder,
    SafeColoredVertexRelaxation, SineModeP1Refiner,
    local_photometric_logits, spectralize_bounded_logits,
)
from qcopt.neural_bijection.dense.photometric_hint import physical_image_gradient
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(
        cross(b - a, c - a).amin(), cross(c - a, d - a).amin(),
    ).detach() * 1024**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=77557)
    parser.add_argument("--test-seed", type=int, default=939031)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-12)
    parser.add_argument("--save-state")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    image_side = 512
    train = make_dataset(args.train_count, image_side, args.train_seed,
                         target_family="high128_tiles", return_coefficients=True)
    test = make_dataset(args.test_count, image_side, args.test_seed,
                        target_family="high128_tiles", return_coefficients=True)
    decoder = ForwardPatchP1Pyramid(17, 257, patch_cells=8).to(device)
    encoder = PatchPyramidImageEncoder(
        17, decoder.level_sides, feature_side=257, width=16,
    ).to(device)
    saved = torch.load(args.coarse_checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    encoder.requires_grad_(False)
    feedback = SafeColoredVertexRelaxation(257, motion_mode="radial", raw_span=2).to(device)
    windows = tuple(
        (col / 4, (col + 1) / 4, row / 4, (row + 1) / 4)
        for row in range(4) for col in range(4)
    )
    refiner = SineModeP1Refiner(
        257, 1025, cycles=((128, 128),) * 16, windows=windows,
        mechanism="patch", checkpoint_updates=True,
    ).to(device)
    table_coarse = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(256, 256), height=image_side, width=image_side,
    )
    table_fine = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=image_side, width=image_side,
    )
    for table in (table_coarse, table_fine):
        table.prepare(device=device, dtype=torch.float32)
    axis = torch.arange(image_side, device=device, dtype=torch.float32) / (image_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    high = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    modes = []
    for xlo, xhi, ylo, yhi in windows:
        tx = (xx - xlo) / (xhi - xlo)
        ty = (yy - ylo) / (yhi - ylo)
        wx = torch.where((tx >= 0) & (tx <= 1),
                         torch.sin(math.pi * tx).square(), 0.0)
        wy = torch.where((ty >= 0) & (ty <= 1),
                         torch.sin(math.pi * ty).square(), 0.0)
        modes.append(high * wx * wy)
    modes = torch.stack(modes)
    gain_logit = torch.nn.Parameter(torch.zeros(16, device=device))
    optimizer = torch.optim.Adam((gain_logit,), lr=args.learning_rate,
                                 eps=args.adam_eps)
    batch_rng = torch.Generator(device="cpu").manual_seed(41728)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(fixed: torch.Tensor, moving: torch.Tensor):
        with torch.no_grad():
            seed, levels = encoder(fixed, moving)
            coarse = decoder(seed, levels)
            for _ in range(2):
                hint = local_photometric_logits(
                    fixed, moving, coarse, window=3, ridge=1.0, raw_span=2.0,
                )
                hint = spectralize_bounded_logits(
                    hint, side=257, raw_span=2.0, count=16,
                )
                floor = coarse.new_full((coarse.shape[0],), 0.05 / 256**2)
                coarse = feedback(coarse, hint, area_floor=floor)
            query = table_coarse.interpolate(coarse.reshape(coarse.shape[0], -1, 2))
            query = query.reshape(-1, image_side, image_side, 2)
            grid = 2 * query - 1
            warped = F.grid_sample(moving, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            gradient = F.grid_sample(
                physical_image_gradient(moving), grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )
            design = modes[None] * (gradient[:, None, 0] + gradient[:, None, 1])
            residual = (fixed - warped)[:, 0]
            rhs = (design * residual[:, None]).mean(dim=(2, 3))
            estimate = rhs / (design.square().mean(dim=(2, 3)) + 1e-12)
        gain = 2 * torch.sigmoid(gain_logit)
        corrected = estimate * gain[None]
        mapped = refiner(coarse, corrected)
        query = table_fine.interpolate(mapped.reshape(mapped.shape[0], -1, 2))
        query = query.reshape(-1, image_side, image_side, 2)
        predicted = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                                  padding_mode="border", align_corners=True)
        return mapped, query, predicted, corrected

    def evaluate(data: tuple[torch.Tensor, ...]) -> dict:
        sums = {"image": 0.0, "map": 0.0, "coeff": 0.0}
        smallest = float("inf")
        with torch.no_grad():
            for start in range(0, data[0].shape[0], args.batch):
                fixed, moving, target, coeff = (
                    x[start:start + args.batch].to(device) for x in data
                )
                mapped, query, predicted, estimated = forward(fixed, moving)
                n = fixed.shape[0]
                sums["image"] += float((predicted - fixed).square().mean()) * n
                sums["map"] += float((query - target).square().sum(dim=-1).mean()) * n
                sums["coeff"] += float((estimated - coeff[:, 2:]).square().sum())
                smallest = min(smallest, minimum_jacobian(mapped))
        count = data[0].shape[0]
        return {
            "image_mse": sums["image"] / count,
            "query_map_vector_rmse": math.sqrt(sums["map"] / count),
            "coefficient_vector_rmse": math.sqrt(sums["coeff"] / count),
            "minimum_jacobian": smallest,
        }

    initial = evaluate(test)
    times = []
    first_step_gradient_max = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        ids = torch.randint(args.train_count, (args.batch,), generator=batch_rng)
        fixed, moving = (x[ids].to(device) for x in train[:2])
        optimizer.zero_grad(set_to_none=True)
        sync()
        start = time.perf_counter()
        _, _, predicted, _ = forward(fixed, moving)
        loss = (predicted - fixed).square().mean()
        loss.backward()
        if first_step_gradient_max is None:
            first_step_gradient_max = float(gain_logit.grad.abs().amax())
        optimizer.step()
        sync()
        times.append(time.perf_counter() - start)
    final = evaluate(test)
    if args.save_state:
        torch.save({
            "gain_logit": gain_logit.detach().cpu(),
            "coarse_checkpoint": args.coarse_checkpoint,
            "mode_cycles": ((128, 128),) * 16,
            "windows": windows,
            "training_family": "high128_tiles",
        }, args.save_state)
    print(json.dumps({
        "method": "phase7_image_only_tile_readout_gain_train",
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "training_objective": "image_only_pixel_MSE",
        "control_side": 1025,
        "image_side": image_side,
        "batch": args.batch,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "adam_eps": args.adam_eps,
        "first_step_gradient_max": first_step_gradient_max,
        "coarse_checkpoint": args.coarse_checkpoint,
        "save_state": args.save_state,
        "initial_test": initial,
        "final_test": final,
        "gain_min": float((2 * torch.sigmoid(gain_logit)).amin()),
        "gain_max": float((2 * torch.sigmoid(gain_logit)).amax()),
        "gain_mean": float((2 * torch.sigmoid(gain_logit)).mean()),
        "median_step_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
