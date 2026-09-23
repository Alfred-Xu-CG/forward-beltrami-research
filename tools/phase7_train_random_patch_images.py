"""Image-to-latent training on independently sampled local 2D P1 motions."""

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
    ResidualStaggeredPatchP1Layer, local_photometric_logits,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian


class PairToPatchLatent(torch.nn.Module):
    def __init__(self, side: int, width: int, cost_radius: int = 0,
                 hint_channels: bool = False) -> None:
        super().__init__()
        self.side = side
        self.cost_radius = cost_radius
        self.hint_channels = hint_channels
        channels = 3 + ((2 * cost_radius + 1) ** 2 if cost_radius else 0)
        channels += 2 if hint_channels else 0
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(channels, width, 5, padding=2), torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=2, dilation=2), torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=4, dilation=4), torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1), torch.nn.GELU(),
        )
        self.head = torch.nn.Conv2d(width, 2, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor,
                hint: torch.Tensor | None = None) -> torch.Tensor:
        pair = torch.cat((fixed, moving, fixed - moving), dim=1)
        pair = F.interpolate(pair, size=(self.side, self.side),
                             mode="bilinear", align_corners=True)
        if self.cost_radius:
            kernel = 2 * self.cost_radius + 1
            neighbor = F.unfold(pair[:, 1:2], kernel_size=kernel,
                                padding=self.cost_radius)
            neighbor = neighbor.reshape(pair.shape[0], kernel * kernel,
                                        self.side, self.side)
            cost = (neighbor - pair[:, :1]).square()
            pair = torch.cat((pair, cost), dim=1)
        if self.hint_channels:
            if hint is None:
                raise ValueError("hint is required when hint channels are enabled")
            padded = F.pad(hint.permute(0, 3, 1, 2), (1, 1, 1, 1))
            pair = torch.cat((pair, padded), dim=1)
        field = self.head(self.body(pair))
        return field[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1).contiguous()


def make_dataset(count: int, image_side: int, control_side: int, seed: int,
                 *, amplitude_min: float, amplitude_max: float,
                 patch_width: float) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    origins = 0.08 + 0.78 * torch.rand(count, 2, generator=generator)
    magnitude = amplitude_min + (amplitude_max - amplitude_min) * torch.rand(
        count, generator=generator,
    )
    angle = 2 * math.pi * torch.rand(count, generator=generator)
    vector = torch.stack((magnitude * torch.cos(angle),
                          magnitude * torch.sin(angle)), dim=-1)
    keys = torch.randint(-22, 23, (count, 8, 2), generator=generator)
    phases = 2 * math.pi * torch.rand(count, 8, generator=generator)
    weights = torch.randn(count, 8, generator=generator)

    def map_at(side: int) -> tuple[torch.Tensor, torch.Tensor]:
        axis = torch.arange(side, dtype=torch.float32) / (side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        tx = (xx[None] - origins[:, 0, None, None]) / patch_width
        ty = (yy[None] - origins[:, 1, None, None]) / patch_width
        wx = torch.where((tx >= 0) & (tx <= 1),
                         torch.sin(math.pi * tx).square(), 0)
        wy = torch.where((ty >= 0) & (ty <= 1),
                         torch.sin(math.pi * ty).square(), 0)
        envelope = wx * wy
        identity = torch.stack((xx, yy), dim=-1)[None]
        target = identity + envelope[..., None] * vector[:, None, None, :]
        return target, envelope

    target_image, _ = map_at(image_side)
    target_control, support = map_at(control_side)
    axis = torch.arange(image_side, dtype=torch.float32) / (image_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    texture_parts = []
    for start in range(0, count, 8):
        stop = min(start + 8, count)
        kx = keys[start:stop, :, 0, None, None]
        ky = keys[start:stop, :, 1, None, None]
        wave = torch.sin(2 * math.pi * (kx * xx + ky * yy)
                         + phases[start:stop, :, None, None])
        texture_parts.append((weights[start:stop, :, None, None] * wave).sum(dim=1))
    texture = torch.cat(texture_parts, dim=0)
    texture = (texture - texture.mean(dim=(-1, -2), keepdim=True)) / (
        texture.std(dim=(-1, -2), keepdim=True) + 1e-6
    )
    moving = texture[:, None].contiguous()
    fixed = F.grid_sample(
        moving, 2 * target_image - 1, mode="bilinear",
        padding_mode="border", align_corners=True,
    ).detach()
    return fixed, moving, target_control, support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--cost-radius", type=int, default=0)
    parser.add_argument("--photometric-hint", action="store_true")
    parser.add_argument("--hint-window", type=int, default=7)
    parser.add_argument("--hint-ridge", type=float, default=1.0)
    parser.add_argument("--hint-gain", type=float, default=0.75)
    parser.add_argument("--initial-hint-iterations", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--test-seed", type=int, default=194381)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--amplitude-min", type=float, default=0.003)
    parser.add_argument("--amplitude-max", type=float, default=0.006)
    parser.add_argument("--objective", choices=("image", "map"), default="image")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-state", default=None)
    args = parser.parse_args()
    if (args.side - 1) % args.patch_cells:
        raise ValueError("patch size must divide side-1")
    if not 0 <= args.cost_radius <= 4:
        raise ValueError("cost radius must lie in [0,4]")
    if args.initial_hint_iterations < 0 or (
        args.initial_hint_iterations and not args.photometric_hint
    ):
        raise ValueError("initial hint iterations require photometric-hint")
    torch.manual_seed(20260925)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    patch_width = args.patch_cells / (args.side - 1)
    train = tuple(t.to(device) for t in make_dataset(
        args.train_count, args.image_side, args.side, 82617,
        amplitude_min=args.amplitude_min,
        amplitude_max=args.amplitude_max, patch_width=patch_width,
    ))
    test = tuple(t.to(device) for t in make_dataset(
        args.test_count, args.image_side, args.side, args.test_seed,
        amplitude_min=args.amplitude_min,
        amplitude_max=args.amplitude_max, patch_width=patch_width,
    ))
    decoder = ResidualStaggeredPatchP1Layer(
        args.side, args.patch_cells, cycles=args.cycles,
        minimum_jacobian=0.05,
    ).to(device)
    encoder = PairToPatchLatent(args.side, args.width, args.cost_radius,
                                hint_channels=args.photometric_hint).to(device)
    if args.load_state:
        saved = torch.load(args.load_state, map_location=device, weights_only=True)
        encoder.load_state_dict(saved["encoder"])
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    axis = torch.arange(args.side, device=device, dtype=torch.float32) / (args.side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, eps=1e-8)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.05,
    ) if args.steps else None)
    sample_generator = torch.Generator(device="cpu").manual_seed(9162)

    def forward(fixed: torch.Tensor, moving: torch.Tensor):
        base = identity.expand(fixed.shape[0], -1, -1, -1)
        for _ in range(args.initial_hint_iterations):
            preliminary = local_photometric_logits(
                fixed, moving, base, window=args.hint_window,
                ridge=args.hint_ridge, raw_span=0.5 * args.patch_cells,
            )
            base = decoder(base, args.hint_gain * preliminary)
        hint = (local_photometric_logits(
            fixed, moving, base,
            window=args.hint_window, ridge=args.hint_ridge,
            raw_span=0.5 * args.patch_cells,
        ) if args.photometric_hint else None)
        correction = encoder(fixed, moving, hint)
        latent = (correction + args.hint_gain * hint
                  if hint is not None and not args.initial_hint_iterations
                  else correction)
        control = decoder(base, latent)
        query = table.interpolate(control.reshape(fixed.shape[0], -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        image_loss = (warped - fixed).square().mean()
        return control, image_loss

    @torch.no_grad()
    def evaluate(dataset: tuple[torch.Tensor, ...]) -> dict[str, float]:
        total_image = total_map = total_local = total_local_n = 0.0
        margins = []
        for start in range(0, dataset[0].shape[0], args.batch):
            fixed, moving, target, support = (
                t[start:start + args.batch] for t in dataset
            )
            control, image_loss = forward(fixed, moving)
            difference = (control - target).square().sum(dim=-1)
            total_image += float(image_loss) * fixed.shape[0]
            total_map += float(difference.mean(dim=(-1, -2)).sum())
            local_mask = support > 0
            total_local += float(difference[local_mask].sum())
            total_local_n += int(local_mask.sum())
            margins.append(minimum_jacobian(control))
        count = dataset[0].shape[0]
        return {
            "image_mse": total_image / count,
            "control_vertex_vector_rmse": math.sqrt(total_map / count),
            "support_vertex_vector_rmse": math.sqrt(total_local / total_local_n),
            "minimum_jacobian": min(margins),
        }

    encoder.eval()
    initial = evaluate(test)
    encoder.train()
    history, times = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids = torch.randint(args.train_count, (args.batch,), generator=sample_generator).to(device)
        fixed, moving, target = (t[ids] for t in train[:3])
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        control, image_loss = forward(fixed, moving)
        map_loss = (control - target).square().sum(dim=-1).mean()
        loss = (1e3 * image_loss if args.objective == "image" else 1e6 * map_loss)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
        if step in (1, args.steps) or step % max(1, args.steps // 5) == 0:
            encoder.eval()
            history.append({"step": step, "test": evaluate(test)})
            encoder.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - began
    encoder.eval()
    final_train, final_test = evaluate(train), evaluate(test)
    if args.save_state:
        torch.save({"encoder": encoder.state_dict(), "args": vars(args)}, args.save_state)

    # Independent representability check: invert the bounded endpoint field
    # from target vertices, not from a network prediction.
    teacher_errors = []
    teacher_margins = []
    with torch.no_grad():
        span = 0.5 * patch_width
        for start in range(0, args.test_count, args.batch):
            target = test[2][start:start + args.batch]
            desired = (target - identity)[:, 1:-1, 1:-1] / span
            teacher = torch.atanh(desired.clamp(-0.999999, 0.999999))
            output = decoder(identity.expand(target.shape[0], -1, -1, -1), teacher)
            teacher_errors.append(float((output - target).square().sum(dim=-1).mean()))
            teacher_margins.append(minimum_jacobian(output))

    print(json.dumps({
        "method": "phase7_random_patch_image_to_residual_f2_latent",
        "objective": args.objective,
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "patch_cells": args.patch_cells,
        "patch_width": patch_width,
        "cycles": args.cycles,
        "amplitude_range": [args.amplitude_min, args.amplitude_max],
        "train_count": args.train_count,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "batch": args.batch,
        "steps": args.steps,
        "learning_rate": args.lr,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "cost_radius": args.cost_radius,
        "photometric_hint": args.photometric_hint,
        "hint_window": args.hint_window if args.photometric_hint else None,
        "hint_gain": args.hint_gain if args.photometric_hint else None,
        "initial_hint_iterations": args.initial_hint_iterations,
        "latent_elements_per_sample": (args.side - 2) ** 2 * 2,
        "initial_test": initial,
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "teacher_test_mean_vertex_vector_rmse": math.sqrt(
            sum(teacher_errors) / len(teacher_errors)
        ),
        "teacher_test_minimum_jacobian": min(teacher_margins),
        "median_training_step_seconds": (statistics.median(
            times[10:] if len(times) > 10 else times
        ) if times else None),
        "total_training_seconds": training_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": "float32",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
