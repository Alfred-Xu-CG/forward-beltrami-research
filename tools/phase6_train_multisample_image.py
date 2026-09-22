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
from qcopt.neural_bijection.dense import (
    CoarseFineConvexQuadComposition,
    ExactAlternatingMonotoneComposition,
    HierarchicalConvexQuadFreeCenterLayer,
    MultiscaleMonotoneGridLayer,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def make_dataset(
    count: int, image_side: int, seed: int, *, return_coefficients: bool = False, target_family: str = "base"
) -> tuple[torch.Tensor, ...]:
    """Make independent textures with uniformly bounded, boundary-fixed maps."""
    if target_family not in ("base", "high32"):
        raise ValueError("target_family must be base or high32")
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
    if target_family == "base":
        ax = 0.012 + 0.023 * torch.rand(count, 1, 1, generator=generator)
        ay = 0.025 + 0.030 * torch.rand(count, 1, 1, generator=generator)
        af = -0.002 + 0.005 * torch.rand(count, 1, 1, generator=generator)
        fine_cycles = 8
    else:
        ax = 0.005 + 0.010 * torch.rand(count, 1, 1, generator=generator)
        ay = 0.010 + 0.015 * torch.rand(count, 1, 1, generator=generator)
        af = -0.002 + 0.0045 * torch.rand(count, 1, 1, generator=generator)
        fine_cycles = 32
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    fine = torch.sin(2 * math.pi * fine_cycles * xx) * torch.sin(2 * math.pi * fine_cycles * yy)
    disp_x = ax * bump + af * fine
    disp_y = ay * bump + af * fine
    true_map = torch.stack((xx + disp_x, yy + disp_y), dim=-1)
    moving = texture[:, None].contiguous()
    fixed = F.grid_sample(moving, 2 * true_map - 1, mode="bilinear", padding_mode="border", align_corners=True).detach()
    if return_coefficients:
        coefficients = torch.cat((ax, ay, af), dim=-1).reshape(count, 3)
        return fixed, moving, true_map, coefficients
    return fixed, moving, true_map


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def _edge_strain(control: torch.Tensor) -> torch.Tensor:
    """Mean squared fine-edge derivative of F-id; no target map is consulted."""
    scale = control.shape[1] - 1
    horizontal = scale * (control[:, :, 1:] - control[:, :, :-1])
    vertical = scale * (control[:, 1:] - control[:, :-1])
    identity_horizontal = horizontal.new_tensor((1.0, 0.0))
    identity_vertical = vertical.new_tensor((0.0, 1.0))
    return 0.5 * (
        (horizontal - identity_horizontal).square().sum(dim=-1).mean()
        + (vertical - identity_vertical).square().sum(dim=-1).mean()
    )


class ContextualA2Body(torch.nn.Module):
    """Three pooled scales supply context while retaining fine-grid features."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(2, width, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1), torch.nn.GELU(),
        )
        self.context = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Conv2d(width, width, 3, padding=1), torch.nn.GELU())
            for _ in range(3)
        )
        self.fuse = torch.nn.Sequential(torch.nn.Conv2d(width, width, 3, padding=1), torch.nn.GELU())

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        local = self.local(pair)
        current = local
        fused = local
        for block in self.context:
            current = block(F.avg_pool2d(current, kernel_size=2, stride=2))
            fused = fused + F.interpolate(current, size=local.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(fused)


class ConvexQuadImageEncoder(torch.nn.Module):
    """Image pyramid to shared-edge and free-center logits at every level."""

    def __init__(self, side: int, width: int = 8, *, head_mode: str = "multilevel", body_mode: str = "local") -> None:
        super().__init__()
        if head_mode not in ("multilevel", "shared"):
            raise ValueError("head_mode must be multilevel or shared")
        if body_mode not in ("local", "context"):
            raise ValueError("body_mode must be local or context")
        self.side = side
        self.head_mode = head_mode
        self.body_mode = body_mode
        self.latent_sides = HierarchicalConvexQuadFreeCenterLayer(side).latent_sides
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, width, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.GELU(),
        ) if body_mode == "local" else ContextualA2Body(width)
        self.root_head = torch.nn.Linear(width, 2)
        head_count = len(self.latent_sides) if head_mode == "multilevel" else 1
        self.horizontal_heads = torch.nn.ModuleList(torch.nn.Conv2d(width, 1, 1) for _ in range(head_count))
        self.vertical_heads = torch.nn.ModuleList(torch.nn.Conv2d(width, 1, 1) for _ in range(head_count))
        self.center_heads = torch.nn.ModuleList(torch.nn.Conv2d(width, 2, 1) for _ in range(head_count))
        for head in (self.root_head, *self.horizontal_heads, *self.vertical_heads, *self.center_heads):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(
        self, pair: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]]:
        reduced = F.interpolate(pair, size=(self.side, self.side), mode="bilinear", align_corners=True)
        fine = self.body(reduced)
        root = self.root_head(fine.mean(dim=(2, 3)))[:, None, None, :]
        levels = []
        for index, current in enumerate(self.latent_sides):
            head_index = index if self.head_mode == "multilevel" else 0
            feature = F.interpolate(fine, size=(current, current), mode="bilinear", align_corners=True)
            horizontal_vertices = self.horizontal_heads[head_index](feature)[:, 0]
            vertical_vertices = self.vertical_heads[head_index](feature)[:, 0]
            center_vertices = self.center_heads[head_index](feature).permute(0, 2, 3, 1)
            horizontal = 0.5 * (horizontal_vertices[:, :, :-1] + horizontal_vertices[:, :, 1:])
            vertical = 0.5 * (vertical_vertices[:, :-1, :] + vertical_vertices[:, 1:, :])
            center = 0.25 * (
                center_vertices[:, :-1, :-1] + center_vertices[:, :-1, 1:]
                + center_vertices[:, 1:, :-1] + center_vertices[:, 1:, 1:]
            )
            levels.append((horizontal, vertical, center))
        return root, tuple(levels)


class CoarseFineConvexQuadImageEncoder(torch.nn.Module):
    """Separate image heads for the coarse and fine exact-composition factors."""

    def __init__(self, coarse_side: int, fine_side: int, *, width: int = 8, head_mode: str, body_mode: str) -> None:
        super().__init__()
        self.coarse = ConvexQuadImageEncoder(coarse_side, width=width, head_mode=head_mode, body_mode=body_mode)
        self.fine = ConvexQuadImageEncoder(fine_side, width=width, head_mode=head_mode, body_mode=body_mode)

    def forward(self, pair: torch.Tensor):
        return self.coarse(pair), self.fine(pair)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("A", "AB2", "A2", "CF2"), required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--coarse-side", type=int, default=17)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--strain-weight", type=float, default=0.0)
    parser.add_argument("--a2-head-mode", choices=("multilevel", "shared"), default="multilevel")
    parser.add_argument("--a2-body-mode", choices=("local", "context"), default="local")
    parser.add_argument("--a2-width", type=int, default=8)
    parser.add_argument("--oracle-map-loss", action="store_true", help="diagnostic target-map supervision; not image-only training")
    parser.add_argument("--target-family", choices=("base", "high32"), default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-state", default=None, help="warm-start encoder weights; Adam state is restarted")
    args = parser.parse_args()
    if min(args.side, args.image_side, args.train_count, args.test_count, args.batch, args.steps, args.a2_width) < 1 or args.strain_weight < 0:
        raise ValueError("all dimensions, counts and steps must be positive")
    if args.strain_weight and args.method != "A2":
        raise ValueError("the current strain ablation is defined only for A2")
    if args.a2_head_mode != "multilevel" and args.method not in ("A2", "CF2"):
        raise ValueError("a2-head-mode applies only to A2 or CF2")
    if args.a2_body_mode != "local" and args.method not in ("A2", "CF2"):
        raise ValueError("a2-body-mode applies only to A2 or CF2")
    if args.a2_width != 8 and args.method not in ("A2", "CF2"):
        raise ValueError("a2-width applies only to A2 or CF2")
    if args.oracle_map_loss and args.method != "A2":
        raise ValueError("the oracle-map-loss diagnostic is defined only for A2")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    train = tuple(t.to(device) for t in make_dataset(args.train_count, args.image_side, 55101, target_family=args.target_family))
    test = tuple(t.to(device) for t in make_dataset(args.test_count, args.image_side, 99317, target_family=args.target_family))
    table = None
    if args.method != "CF2":
        table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(args.side - 1, args.side - 1), height=args.image_side, width=args.image_side
        )
        table.prepare(device=device, dtype=torch.float32)
    if args.method == "A":
        encoder = SmallImageEncoder(args.side).to(device)
        decoder = MultiscaleMonotoneGridLayer(args.side)
    elif args.method == "A2":
        encoder = ConvexQuadImageEncoder(args.side, width=args.a2_width, head_mode=args.a2_head_mode, body_mode=args.a2_body_mode).to(device)
        decoder = HierarchicalConvexQuadFreeCenterLayer(args.side)
    elif args.method == "CF2":
        encoder = CoarseFineConvexQuadImageEncoder(
            args.coarse_side, args.side, width=args.a2_width, head_mode=args.a2_head_mode, body_mode=args.a2_body_mode
        ).to(device)
        decoder = CoarseFineConvexQuadComposition(args.coarse_side, args.side, args.image_side)
        decoder.prepare(device=device, dtype=torch.float32)
    else:
        axes = ("vertical", "horizontal")
        encoder = AlternatingImageEncoder(args.side, axes).to(device)
        decoder = ExactAlternatingMonotoneComposition(args.side, table, axes)
    if args.load_state:
        previous = torch.load(args.load_state, map_location=device, weights_only=False)
        previous_args = previous["args"]
        defaults = {"target_family": "base", "coarse_side": 17, "a2_head_mode": "multilevel", "a2_body_mode": "local", "a2_width": 8}
        for key in ("method", "side", "coarse_side", "image_side", "target_family", "a2_head_mode", "a2_body_mode", "a2_width"):
            if previous_args.get(key, defaults.get(key)) != getattr(args, key):
                raise ValueError(f"loaded checkpoint does not match {key}")
        encoder.load_state_dict(previous["encoder"])
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
        elif args.method == "A2":
            control = decoder(*latent)
            predicted = table.interpolate(control.reshape(len(indices), -1, 2))
            controls = (control,)
        elif args.method == "CF2":
            result = decoder(*latent[0], *latent[1])
            predicted = result.dense
            controls = result.controls
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
        strain_sum = 0.0
        minimum = [float("inf")] * (2 if args.method in ("AB2", "CF2") else 1)
        count = dataset[0].shape[0]
        for start in range(0, count, args.batch):
            stop = min(start + args.batch, count)
            indices = torch.arange(start, stop, device=device)
            image_mse, map_mse, controls = forward(indices, dataset)
            assert map_mse is not None
            image_sum += image_mse.item() * (stop - start)
            map_sum += map_mse.item() * (stop - start)
            if args.method == "A2":
                strain_sum += _edge_strain(controls[0]).item() * (stop - start)
            for index, control in enumerate(controls):
                minimum[index] = min(minimum[index], _minimum_area_ratio(control))
        result = {
            "image_mse": image_sum / count,
            "map_rmse": math.sqrt(map_sum / count),
            "minimum_layer_signed_area_ratios": minimum,
        }
        if args.method == "A2":
            result["edge_strain_energy"] = strain_sum / count
        return result

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
        image_mse, map_mse, controls = forward(draw, train, measure_map_error=args.oracle_map_loss)
        total_loss = map_mse if args.oracle_map_loss else image_mse
        if args.strain_weight:
            total_loss = total_loss + args.strain_weight * _edge_strain(controls[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        middle = time.perf_counter()
        total_loss.backward()
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
            records.append({
                "step": step + 1,
                "sampled_train_image_mse_before_update": image_mse.item(),
                "sampled_train_total_objective_before_update": total_loss.item(),
                "cumulative_seconds": time.perf_counter() - began_all,
            })
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
        "layers": 2 if args.method in ("AB2", "CF2") else 1,
        "representation": "exact_PL_composition" if args.method in ("AB2", "CF2") else "original_grid_P1",
        "coarse_control_side": args.coarse_side if args.method == "CF2" else None,
        "coarse_control_vertices": args.coarse_side**2 if args.method == "CF2" else None,
        "coarse_control_faces": 2 * (args.coarse_side - 1)**2 if args.method == "CF2" else None,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": 55101,
        "test_seed": 99317,
        "batch": args.batch,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "strain_weight": args.strain_weight,
        "a2_head_mode": args.a2_head_mode if args.method in ("A2", "CF2") else None,
        "a2_body_mode": args.a2_body_mode if args.method in ("A2", "CF2") else None,
        "a2_width": args.a2_width if args.method in ("A2", "CF2") else None,
        "oracle_map_loss": args.oracle_map_loss,
        "target_family": args.target_family,
        "loaded_state": args.load_state,
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
