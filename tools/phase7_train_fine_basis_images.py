"""Train a tiny image-to-fine-latent encoder through the 1025^2 certified P1 layer.

This controlled task isolates sub-coarse-grid information.  The learned
encoder has only two scalar parameters; it is not a general registration net.
"""

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


class GlobalPhotometricBasisEncoder(torch.nn.Module):
    """Fit one fixed high-frequency deformation basis from an image pair."""

    def __init__(self, image_side: int, maximum_coefficient: float) -> None:
        super().__init__()
        axis = torch.arange(image_side, dtype=torch.float32) / (image_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        basis = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
        self.register_buffer("basis", basis[None, None], persistent=False)
        self.maximum_coefficient = maximum_coefficient
        self.gain = torch.nn.Parameter(torch.zeros(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def initial_least_squares(self, fixed: torch.Tensor,
                              moving: torch.Tensor) -> torch.Tensor:
        gradient = physical_image_gradient(moving)
        directional = self.basis * (gradient[:, 0:1] + gradient[:, 1:2])
        residual = fixed - moving
        numerator = (directional * residual).mean(dim=(1, 2, 3))
        denominator = directional.square().mean(dim=(1, 2, 3)) + 1e-6
        return numerator / denominator

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        ls = self.initial_least_squares(fixed, moving)
        normalized = (ls / self.maximum_coefficient).clamp(-0.999, 0.999)
        return self.maximum_coefficient * torch.tanh(
            self.gain * torch.atanh(normalized) + self.bias
        )


def make_dataset(count: int, image_side: int, seed: int, device: torch.device,
                 coefficient_limit: float) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficient = (2 * torch.rand(count, generator=generator) - 1) * coefficient_limit
    keys = torch.randint(-22, 23, (count, 8, 2), generator=generator)
    phases = 2 * math.pi * torch.rand(count, 8, generator=generator)
    weights = torch.randn(count, 8, generator=generator)
    axis = torch.arange(image_side, dtype=torch.float32, device=device) / (image_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    high = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    identity = torch.stack((xx, yy), dim=-1)
    moving_parts = []
    fixed_parts = []
    for start in range(0, count, 8):
        stop = min(start + 8, count)
        kx = keys[start:stop, :, 0, None, None].to(device)
        ky = keys[start:stop, :, 1, None, None].to(device)
        phase = phases[start:stop, :, None, None].to(device)
        weight = weights[start:stop, :, None, None].to(device)
        wave = torch.sin(2 * math.pi * (kx * xx + ky * yy) + phase)
        texture = (weight * wave).sum(dim=1)
        texture = (texture - texture.mean(dim=(-1, -2), keepdim=True)) / (
            texture.std(dim=(-1, -2), keepdim=True) + 1e-6
        )
        moving = texture[:, None].contiguous()
        target = identity[None] + (coefficient[start:stop].to(device)
                                   [:, None, None, None] * high[None, :, :, None])
        fixed = F.grid_sample(
            moving, 2 * target - 1, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).detach()
        moving_parts.append(moving)
        fixed_parts.append(fixed)
    return (torch.cat(fixed_parts), torch.cat(moving_parts),
            coefficient.to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--coefficient-limit", type=float, default=0.00015)
    parser.add_argument("--objective", choices=("map", "image"), default="map")
    parser.add_argument("--initial-gain", type=float, default=0.0)
    args = parser.parse_args()
    torch.manual_seed(20260927)
    device = torch.device(args.device)
    train = make_dataset(args.train_count, args.image_side, 31719, device,
                         args.coefficient_limit)
    test = make_dataset(args.test_count, args.image_side, 41817, device,
                        args.coefficient_limit)
    encoder = GlobalPhotometricBasisEncoder(
        args.image_side, args.coefficient_limit,
    ).to(device)
    with torch.no_grad():
        encoder.gain.fill_(args.initial_gain)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, coarse_cycles=2,
        minimum_jacobian=0.05, compute_dtype=torch.float64,
        certify_output=True,
    ).to(device)
    fine_axis = torch.arange(1025, dtype=torch.float64, device=device) / 1024
    fy, fx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    fine_basis = (torch.sin(256 * math.pi * fx)
                  * torch.sin(256 * math.pi * fy))[None, :, :, None]
    identity = decoder.fine_identity
    coarse_latent = torch.zeros(args.batch, 255, 255, 2,
                                 dtype=torch.float32, device=device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    sample_generator = torch.Generator(device="cpu").manual_seed(19917)

    def decode(coefficient: torch.Tensor) -> torch.Tensor:
        batch = coefficient.shape[0]
        displacement = coefficient.double()[:, None, None, None] * fine_basis
        ratio = (displacement / (2 / 1024))[:, 1:-1, 1:-1]
        fine_latent = torch.atanh(ratio).expand(-1, -1, -1, 2).contiguous().float()
        return decoder(coarse_latent[:batch], fine_latent)

    def sample_image(control: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        query = table.interpolate(control.reshape(control.shape[0], -1, 2))
        return F.grid_sample(
            moving, 2 * query.float() - 1, mode="bilinear",
            padding_mode="border", align_corners=True,
        )

    @torch.no_grad()
    def evaluate(dataset: tuple[torch.Tensor, ...]) -> dict[str, float]:
        squared_map = squared_coefficient = squared_image = 0.0
        min_j = math.inf
        returned_identity = 0
        for start in range(0, len(dataset[0]), args.batch):
            fixed, moving, truth = (x[start:start+args.batch] for x in dataset)
            predicted = encoder(fixed, moving)
            control = decode(predicted)
            target = identity + truth.double()[:, None, None, None] * fine_basis
            squared_map += float((control - target).square().sum(dim=-1).mean(
                dim=(1, 2)).sum())
            squared_coefficient += float((predicted - truth).square().sum())
            warped = sample_image(control, moving)
            squared_image += float((warped - fixed).square().mean(
                dim=(1, 2, 3)).sum())
            min_j = min(min_j, minimum_jacobian(control))
            returned_identity += int(torch.all(
                control == identity, dim=(1, 2, 3)).sum())
        count = len(dataset[0])
        return {
            "map_vector_rmse": math.sqrt(squared_map / count),
            "coefficient_rmse": math.sqrt(squared_coefficient / count),
            "image_mse": squared_image / count,
            "minimum_jacobian": min_j,
            "returned_identity": returned_identity,
        }

    initial = evaluate(test)
    analytic_ls_error = []
    with torch.no_grad():
        for start in range(0, len(test[0]), args.batch):
            fixed, moving, truth = (x[start:start+args.batch] for x in test)
            ls = encoder.initial_least_squares(fixed, moving)
            analytic_ls_error.append((ls - truth).square())
    analytic_coefficient_rmse = float(
        torch.cat(analytic_ls_error).mean().sqrt()
    )
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        batch_index = torch.randint(args.train_count, (args.batch,),
                                    generator=sample_generator).to(device)
        fixed, moving, truth = (x[batch_index] for x in train)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        predicted = encoder(fixed, moving)
        control = decode(predicted)
        if args.objective == "map":
            target = identity + truth.double()[:, None, None, None] * fine_basis
            loss = 1e8 * (control - target).square().sum(dim=-1).mean()
        else:
            warped = sample_image(control, moving)
            loss = 1e6 * (warped - fixed).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
    final = evaluate(test)
    teacher_squared_map = 0.0
    teacher_min_j = math.inf
    with torch.no_grad():
        for start in range(0, len(test[0]), args.batch):
            truth = test[2][start:start + args.batch]
            teacher = decode(truth)
            target = identity + truth.double()[:, None, None, None] * fine_basis
            teacher_squared_map += float((teacher - target).square().sum(dim=-1).mean(
                dim=(1, 2)).sum())
            teacher_min_j = min(teacher_min_j, minimum_jacobian(teacher))
    print(json.dumps({
        "method": "phase7_fine_basis_image_to_latent",
        "objective": args.objective,
        "control_side": 1025,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": args.image_side,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "steps": args.steps,
        "initial_gain": args.initial_gain,
        "batch": args.batch,
        "coefficient_limit": args.coefficient_limit,
        "encoder_trainable_parameters": sum(p.numel() for p in encoder.parameters()),
        "baseline": initial,
        "analytic_ls_coefficient_rmse": analytic_coefficient_rmse,
        "final": final,
        "teacher_map_vector_rmse": math.sqrt(teacher_squared_map / args.test_count),
        "teacher_minimum_jacobian": teacher_min_j,
        "gain": float(encoder.gain.detach()),
        "bias": float(encoder.bias.detach()),
        "median_training_step_seconds": statistics.median(times) if times else None,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
