"""Held-out image-to-latent experiment on a deterministic diffeomorphic family.

The analytic map creates fixed images and evaluates results; it is never a
training target or an encoder input. Train/test items differ in texture and
deformation coefficients but use the same bounded family.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_train_alternating_image_to_latent import AlternatingImageEncoder
from phase6_train_image_to_latent import SmallImageEncoder, _minimum_area_ratio
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactAlternatingMonotoneComposition, MultiscaleMonotoneGridLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def make_dataset(count: int, image_side: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make independent textures with uniformly bounded, boundary-fixed maps."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    line = torch.linspace(0.0, 1.0, image_side)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    xx = xx[None]
    yy = yy[None]
    phases = 2.0 * math.pi * torch.rand(count, 4, 1, 1, generator=generator)
    centers = 0.22 + 0.56 * torch.rand(count, 4, 1, 1, generator=generator)
    amps = 0.8 + 0.4 * torch.rand(count, 4, 1, 1, generator=generator)
    texture = (
        0.30 * amps[:, 0] * torch.sin(8 * math.pi * xx + 3 * math.pi * yy + phases[:, 0])
        + 0.25 * amps[:, 1] * torch.cos(13 * math.pi * yy - 2 * math.pi * xx + phases[:, 1])
        + 0.35 * amps[:, 2] * torch.exp(-((xx - centers[:, 0]) ** 2 + (yy - centers[:, 1]) ** 2) / 0.012)
        + 0.30 * amps[:, 3] * torch.exp(-((xx - centers[:, 2]) ** 2 + (yy - centers[:, 3]) ** 2) / 0.008)
        + 0.16 * torch.sin(22 * math.pi * xx + 4 * math.pi * yy + phases[:, 2])
        * torch.cos(18 * math.pi * yy - 3 * math.pi * xx + phases[:, 3])
    )
    ax = 0.012 + 0.023 * torch.rand(count, 1, 1, generator=generator)
    ay = 0.025 + 0.030 * torch.rand(count, 1, 1, generator=generator)
    af = -0.002 + 0.005 * torch.rand(count, 1, 1, generator=generator)
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    fine = torch.sin(16 * math.pi * xx) * torch.sin(16 * math.pi * yy)
    disp_x = ax * bump + af * fine
    disp_y = ay * bump + af * fine
    true_map = torch.stack((xx + disp_x, yy + disp_y), dim=-1)
    moving = texture[:, None].contiguous()
    fixed = F.grid_sample(moving, 2 * true_map - 1, mode="bilinear", padding_mode="border", align_corners=True).detach()
    return fixed, moving, true_map


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("A", "AB2"), required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    if min(args.side, args.image_side, args.train_count, args.test_count, args.batch, args.steps) < 1:
        raise ValueError("all dimensions, counts and steps must be positive")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    train = tuple(t.to(device) for t in make_dataset(args.train_count, args.image_side, 55101))
    test = tuple(t.to(device) for t in make_dataset(args.test_count, args.image_side, 99317))
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1), height=args.image_side, width=args.image_side
    )
    table.prepare(device=device, dtype=torch.float32)
    if args.method == "A":
        encoder = SmallImageEncoder(args.side).to(device)
        decoder = MultiscaleMonotoneGridLayer(args.side)
    else:
        axes = ("vertical", "horizontal")
        encoder = AlternatingImageEncoder(args.side, axes).to(device)
        decoder = ExactAlternatingMonotoneComposition(args.side, table, axes)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)

    def forward(
        indices: torch.Tensor,
        dataset: tuple[torch.Tensor, ...],
        measure_map_error: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...]]:
        fixed, moving = (part[indices] for part in dataset[:2])
        latent = encoder(torch.cat((fixed, moving), dim=1))
        if args.method == "A":
            control = decoder(latent)
            predicted = table.interpolate(control.reshape(len(indices), -1, 2))
            controls = (control,)
        else:
            result = decoder(latent)
            predicted = result.dense
            controls = result.controls
        warped = F.grid_sample(moving, 2 * predicted - 1, mode="bilinear", padding_mode="border", align_corners=True)
        image_mse = (warped - fixed).square().mean()
        map_mse = (predicted - dataset[2][indices]).square().mean() if measure_map_error else None
        return image_mse, map_mse, controls

    @torch.no_grad()
    def evaluate(dataset: tuple[torch.Tensor, ...]) -> dict[str, object]:
        image_sum = 0.0
        map_sum = 0.0
        minimum = [float("inf")] * (1 if args.method == "A" else 2)
        count = dataset[0].shape[0]
        for start in range(0, count, args.batch):
            stop = min(start + args.batch, count)
            indices = torch.arange(start, stop, device=device)
            image_mse, map_mse, controls = forward(indices, dataset)
            assert map_mse is not None
            image_sum += image_mse.item() * (stop - start)
            map_sum += map_mse.item() * (stop - start)
            for index, control in enumerate(controls):
                minimum[index] = min(minimum[index], _minimum_area_ratio(control))
        return {
            "image_mse": image_sum / count,
            "map_rmse": math.sqrt(map_sum / count),
            "minimum_layer_signed_area_ratios": minimum,
        }

    initial_train = evaluate(train)
    initial_test = evaluate(test)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    train_generator = torch.Generator(device="cpu").manual_seed(38819)
    forward_times: list[float] = []
    backward_times: list[float] = []
    records = []
    peak_rss = _rss_bytes()
    began_all = time.perf_counter()
    for step in range(args.steps):
        draw = torch.randint(args.train_count, (args.batch,), generator=train_generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        image_mse, _, _ = forward(draw, train, measure_map_error=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        middle = time.perf_counter()
        image_mse.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_backward = time.perf_counter()
        optimizer.step()
        forward_times.append(middle - began)
        backward_times.append(after_backward - middle)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            current_rss = _rss_bytes()
            if current_rss is not None:
                peak_rss = max(peak_rss or 0, current_rss)
            records.append({"step": step + 1, "sampled_train_image_mse_before_update": image_mse.item(), "cumulative_seconds": time.perf_counter() - began_all})
    total_seconds = time.perf_counter() - began_all
    final_train = evaluate(train)
    final_test = evaluate(test)
    if args.save_state:
        torch.save({"encoder": encoder.state_dict(), "args": vars(args)}, args.save_state)
    print(json.dumps({
        "method": args.method,
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces_per_layer": 2 * (args.side - 1)**2,
        "layers": 1 if args.method == "A" else 2,
        "representation": "original_grid_P1" if args.method == "A" else "exact_PL_composition",
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": 55101,
        "test_seed": 99317,
        "batch": args.batch,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "initial_train": initial_train,
        "initial_test": initial_test,
        "final_train": final_train,
        "final_test": final_test,
        "median_forward_seconds": statistics.median(forward_times[1:] or forward_times),
        "median_backward_seconds": statistics.median(backward_times[1:] or backward_times),
        "total_train_seconds": total_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "sampled_peak_process_rss_bytes": peak_rss,
        "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
