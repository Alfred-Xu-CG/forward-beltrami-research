"""Evaluate an image-only local hint on unknown 1025-grid high-frequency packets."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, local_photometric_logits,
    physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian


def packet(side: int, origin: torch.Tensor, vector: torch.Tensor,
           width: float, dtype: torch.dtype, device: torch.device):
    axis = torch.arange(side, dtype=dtype, device=device) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ox = origin[:, 0].to(device=device, dtype=dtype)[:, None, None]
    oy = origin[:, 1].to(device=device, dtype=dtype)[:, None, None]
    tx, ty = (xx - ox) / width, (yy - oy) / width
    wx = torch.where((tx >= 0) & (tx <= 1),
                     torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1),
                     torch.sin(math.pi * ty).square(), 0)
    envelope = wx * wy
    high = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    identity = torch.stack((xx, yy), dim=-1)[None]
    target = identity + (envelope * high)[..., None] * vector.to(
        device=device, dtype=dtype)[:, None, None, :]
    return target, envelope


def make_dataset(count: int, image_side: int, seed: int,
                 device: torch.device, p1_image_truth: bool = False,
                 ) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    origin = 0.08 + 0.67 * torch.rand(count, 2, generator=generator)
    magnitude = 0.00008 + 0.00007 * torch.rand(count, generator=generator)
    angle = 2 * math.pi * torch.rand(count, generator=generator)
    vector = torch.stack((magnitude * torch.cos(angle),
                          magnitude * torch.sin(angle)), dim=-1)
    keys = torch.randint(-22, 23, (count, 8, 2), generator=generator)
    phases = 2 * math.pi * torch.rand(count, 8, generator=generator)
    weights = torch.randn(count, 8, generator=generator)
    image_target, _ = packet(image_side, origin, vector, 0.25,
                             torch.float32, device)
    control_target, support = packet(1025, origin, vector, 0.25,
                                     torch.float64, device)
    if p1_image_truth:
        table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(1024, 1024),
            height=image_side, width=image_side,
        )
        table.prepare(device=device, dtype=torch.float64)
    axis = torch.arange(image_side, device=device, dtype=torch.float32) / (image_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    moving_parts = []
    fixed_parts = []
    for start in range(0, count, 8):
        stop = min(start + 8, count)
        kx = keys[start:stop, :, 0, None, None].to(device)
        ky = keys[start:stop, :, 1, None, None].to(device)
        phase = phases[start:stop, :, None, None].to(device)
        weight = weights[start:stop, :, None, None].to(device)
        texture = (weight * torch.sin(2 * math.pi * (kx * xx + ky * yy) + phase)).sum(dim=1)
        texture = (texture - texture.mean(dim=(-1, -2), keepdim=True)) / (
            texture.std(dim=(-1, -2), keepdim=True) + 1e-6)
        moving = texture[:, None].contiguous()
        image_query = (
            table.interpolate(control_target[start:stop].reshape(
                stop - start, -1, 2)).float()
            if p1_image_truth else image_target[start:stop]
        )
        fixed = F.grid_sample(
            moving, 2 * image_query - 1, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).detach()
        moving_parts.append(moving)
        fixed_parts.append(fixed)
    return torch.cat(fixed_parts), torch.cat(moving_parts), control_target, support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=58211)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--demod-windows", type=int, nargs="*", default=[])
    parser.add_argument("--p1-image-truth", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    fixed_all, moving_all, target_all, support_all = make_dataset(
        args.count, args.image_side, args.seed, device,
        p1_image_truth=args.p1_image_truth)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=0.05, certify_output=True,
    ).to(device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float64)
    identity = decoder.fine_identity
    fine_x, fine_y = identity[0, ..., 0], identity[0, ..., 1]
    fine_carrier = (torch.sin(256 * math.pi * fine_x)
                    * torch.sin(256 * math.pi * fine_y)).float()
    image_axis = torch.arange(args.image_side, dtype=torch.float32,
                               device=device) / (args.image_side - 1)
    image_y, image_x = torch.meshgrid(image_axis, image_axis, indexing="ij")
    image_carrier = (torch.sin(256 * math.pi * image_x)
                     * torch.sin(256 * math.pi * image_y))[None, None]
    coarse = torch.zeros(args.batch, 255, 255, 2, dtype=torch.float32, device=device)

    def demodulated_logits(fixed: torch.Tensor, moving: torch.Tensor,
                           window: int) -> torch.Tensor:
        gradient = physical_image_gradient(moving)
        gx = image_carrier * gradient[:, 0:1]
        gy = image_carrier * gradient[:, 1:2]
        residual = fixed - moving
        def smooth(value: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(value, window, stride=1, padding=window // 2,
                                count_include_pad=False)
        a = smooth(gx * gx) + args.ridge
        b = smooth(gx * gy)
        d = smooth(gy * gy) + args.ridge
        rhs_x = smooth(gx * residual)
        rhs_y = smooth(gy * residual)
        determinant = a * d - b.square()
        amplitude = torch.cat(((d * rhs_x - b * rhs_y) / determinant,
                               (a * rhs_y - b * rhs_x) / determinant), dim=1)
        amplitude = F.interpolate(amplitude, size=(1025, 1025),
                                  mode="bilinear", align_corners=True)
        proposed = args.gain * amplitude * fine_carrier[None, None]
        ratio = (proposed / (2 / 1024)).clamp(-0.95, 0.95)
        return torch.atanh(ratio[:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()

    def evaluate(method: str, window: int | None) -> dict[str, object]:
        map_sq = local_sq = image_sq = 0.0
        local_n = 0
        minimum = math.inf
        fallback = 0
        times = []
        for start in range(0, args.count, args.batch):
            fixed, moving, target, support = (
                tensor[start:start + args.batch]
                for tensor in (fixed_all, moving_all, target_all, support_all)
            )
            batch = fixed.shape[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.no_grad():
                if method == "zero":
                    latent = torch.zeros(batch, 1023, 1023, 2,
                                         device=device, dtype=torch.float32)
                elif method == "teacher":
                    ratio = ((target - identity)[:, 1:-1, 1:-1] / (2 / 1024))
                    latent = torch.atanh(ratio).float()
                elif method == "demod":
                    latent = demodulated_logits(fixed, moving, window)
                else:
                    latent = args.gain * local_photometric_logits(
                        fixed, moving, identity.float().expand(batch, -1, -1, -1),
                        window=window, ridge=args.ridge, raw_span=2.0,
                    )
                output = decoder(coarse[:batch], latent)
                query = table.interpolate(output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1, mode="bilinear",
                    padding_mode="border", align_corners=True,
                )
                difference = (output - target).square().sum(dim=-1)
                map_sq += float(difference.mean(dim=(1, 2)).sum())
                mask = support > 0
                local_sq += float(difference[mask].sum())
                local_n += int(mask.sum())
                image_sq += float((warped - fixed).square().mean(
                    dim=(1, 2, 3)).sum())
                minimum = min(minimum, minimum_jacobian(output))
                fallback += int(torch.all(output == identity, dim=(1, 2, 3)).sum())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
        return {
            "method": method,
            "window": window,
            "map_vector_rmse": math.sqrt(map_sq / args.count),
            "support_vector_rmse": math.sqrt(local_sq / local_n),
            "image_mse": image_sq / args.count,
            "minimum_jacobian": minimum,
            "identity_outputs": fallback,
            "median_eval_batch_seconds": sorted(times)[len(times) // 2],
        }

    rows = [evaluate("zero", None), evaluate("teacher", None)]
    rows.extend(evaluate("hint", w) for w in args.windows)
    rows.extend(evaluate("demod", w) for w in args.demod_windows)
    print(json.dumps({
        "experiment": "phase7_unknown_fine_packet_hint",
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": args.image_side,
        "count": args.count,
        "batch": args.batch,
        "seed": args.seed,
        "ridge": args.ridge,
        "gain": args.gain,
        "p1_image_truth": args.p1_image_truth,
        "device": str(device),
        "torch_version": torch.__version__,
        "rows": rows,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
