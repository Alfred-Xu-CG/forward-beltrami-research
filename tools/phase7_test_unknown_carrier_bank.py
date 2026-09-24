"""Evaluate an unknown-carrier image hint with a finite carrier dictionary."""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_train_spatial_packet_images import carrier, demod_amplitude


CYCLES = (96, 112, 128)


def multicarrier_packet(side: int, origin: torch.Tensor,
                        vector: torch.Tensor, cycles: torch.Tensor,
                        device: torch.device, dtype: torch.dtype):
    axis = torch.arange(side, device=device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ox = origin[:, 0].to(device=device, dtype=dtype)[:, None, None]
    oy = origin[:, 1].to(device=device, dtype=dtype)[:, None, None]
    tx, ty = (xx - ox) / .25, (yy - oy) / .25
    wx = torch.where((tx >= 0) & (tx <= 1),
                     torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1),
                     torch.sin(math.pi * ty).square(), 0)
    envelope = wx * wy
    frequency = cycles.to(device=device, dtype=dtype)[:, None, None]
    high = torch.sin(2 * math.pi * frequency * xx) * torch.sin(
        2 * math.pi * frequency * yy)
    identity = torch.stack((xx, yy), dim=-1)[None]
    target = identity + (envelope * high)[..., None] * vector.to(
        device=device, dtype=dtype)[:, None, None, :]
    return target, envelope


def dataset(count: int, seed: int, device: torch.device, batch: int,
            table: StructuredDenseQueryTable,
            target_cycles: tuple[int, ...]):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    origin = .08 + .67 * torch.rand(count, 2, generator=generator)
    magnitude = .00008 + .00007 * torch.rand(count, generator=generator)
    angle = 2 * math.pi * torch.rand(count, generator=generator)
    vector = torch.stack((magnitude * torch.cos(angle),
                          magnitude * torch.sin(angle)), dim=-1)
    carrier_id = torch.randint(len(target_cycles), (count,), generator=generator)
    cycles = torch.tensor(target_cycles)[carrier_id]
    keys = torch.randint(-22, 23, (count, 8, 2), generator=generator)
    phases = 2 * math.pi * torch.rand(count, 8, generator=generator)
    weights = torch.randn(count, 8, generator=generator)
    target, support = multicarrier_packet(
        1025, origin, vector, cycles, device, torch.float64)
    axis = torch.arange(512, device=device, dtype=torch.float32) / 511
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    fixed_parts, moving_parts = [], []
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        kx = keys[start:stop, :, 0, None, None].to(device)
        ky = keys[start:stop, :, 1, None, None].to(device)
        phase = phases[start:stop, :, None, None].to(device)
        weight = weights[start:stop, :, None, None].to(device)
        texture = (weight * torch.sin(2 * math.pi * (kx * xx + ky * yy)
                                      + phase)).sum(dim=1)
        texture = (texture - texture.mean(dim=(-1, -2), keepdim=True)) / (
            texture.std(dim=(-1, -2), keepdim=True) + 1e-6)
        moving = texture[:, None].contiguous()
        query = table.interpolate(
            target[start:stop].reshape(stop - start, -1, 2)).float()
        fixed = F.grid_sample(
            moving, 2 * query - 1, align_corners=True,
            mode="bilinear", padding_mode="border")
        moving_parts.append(moving)
        fixed_parts.append(fixed)
    return (torch.cat(fixed_parts), torch.cat(moving_parts),
            target, support, carrier_id.to(device), cycles.to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=89217)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--ridge", type=float, default=1.)
    parser.add_argument("--target-cycles", type=int, nargs="+",
                        default=list(CYCLES))
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    target_cycles = tuple(args.target_cycles)
    fixed_all, moving_all, target_all, support_all, ids_all, cycles_all = dataset(
        args.count, args.seed, device, args.batch, table,
        target_cycles)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True).to(device)
    carriers_img = torch.stack(
        [carrier(512, device, torch.float32) if k == 128 else
         torch.sin(2 * math.pi * k *
                   (torch.arange(512, device=device, dtype=torch.float32) / 511)[:, None])
         * torch.sin(2 * math.pi * k *
                     (torch.arange(512, device=device, dtype=torch.float32) / 511)[None, :])
         for k in CYCLES], dim=0)
    fine_axis = torch.arange(1025, device=device, dtype=torch.float32) / 1024
    fy, fx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    carriers_fine = torch.stack(
        [torch.sin(2 * math.pi * k * fx) *
         torch.sin(2 * math.pi * k * fy) for k in CYCLES], dim=0)
    identity = decoder.fine_identity
    square_by_method = {
        name: {"map": 0., "support": 0., "image": 0.,
               "support_n": 0, "minimum": math.inf, "fallback": 0}
        for name in ("identity", "teacher", "known_carrier", "selected_carrier")
    }
    confusion = torch.zeros(len(target_cycles), len(CYCLES), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            fixed, moving, target, support, true_id, true_cycles = (
                tensor[start:start + args.batch] for tensor in
                (fixed_all, moving_all, target_all, support_all, ids_all,
                 cycles_all))
            count = len(fixed)
            gradient = physical_image_gradient(moving)
            residual = fixed - moving
            amps, scores = [], []
            for k, img_h in enumerate(carriers_img):
                amp = demod_amplitude(
                    fixed, moving, img_h[None, None],
                    window=args.window, ridge=args.ridge)
                prediction = img_h[None, None] * (
                    gradient[:, :1] * amp[:, :1] +
                    gradient[:, 1:2] * amp[:, 1:2])
                score = (residual * prediction -
                         .5 * prediction.square()).mean(dim=(1, 2, 3))
                amps.append(amp)
                scores.append(score)
            amp_stack = torch.stack(amps, dim=1)
            scores = torch.stack(scores, dim=1)
            selected = scores.argmax(dim=1)
            img_axis = torch.arange(512, device=device,
                                    dtype=torch.float32) / 511
            img_y, img_x = torch.meshgrid(img_axis, img_axis, indexing="ij")
            true_frequency = true_cycles.float()[:, None, None]
            true_img_carrier = (
                torch.sin(2 * math.pi * true_frequency * img_x) *
                torch.sin(2 * math.pi * true_frequency * img_y))
            known_amp = demod_amplitude(
                fixed, moving, true_img_carrier[:, None],
                window=args.window, ridge=args.ridge)
            fine_frequency = true_cycles.float()[:, None, None]
            true_fine_carrier = (
                torch.sin(2 * math.pi * fine_frequency * fx) *
                torch.sin(2 * math.pi * fine_frequency * fy))
            for truth, guess in zip(true_id.cpu().tolist(),
                                    selected.cpu().tolist()):
                confusion[truth, guess] += 1
            batch_index = torch.arange(count, device=device)
            coarse = torch.zeros(count, 255, 255, 2,
                                 device=device, dtype=torch.float32)
            for name in square_by_method:
                if name == "identity":
                    latent = torch.zeros(count, 1023, 1023, 2,
                                         device=device, dtype=torch.float32)
                elif name == "teacher":
                    latent = torch.atanh(((target - identity)[:, 1:-1, 1:-1] /
                                          (2 / 1024)).clamp(-.95, .95)).float()
                else:
                    ids = selected
                    amp = (known_amp if name == "known_carrier" else
                           amp_stack[batch_index, ids])
                    amp = F.interpolate(
                        amp, size=(1025, 1025), mode="bilinear",
                        align_corners=True)
                    carrier_fine = (true_fine_carrier[:, None] if
                                    name == "known_carrier" else
                                    carriers_fine[ids, None])
                    proposal = amp * carrier_fine
                    latent = torch.atanh(
                        (proposal / (2 / 1024)).clamp(-.95, .95)
                        [:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()
                output = decoder(coarse, latent)
                query = table.interpolate(output.reshape(count, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1,
                    align_corners=True, mode="bilinear", padding_mode="border")
                square = (output - target).square().sum(dim=-1)
                row = square_by_method[name]
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
    rows = []
    for name, raw in square_by_method.items():
        rows.append({
            "method": name,
            "map_vector_rmse": math.sqrt(raw["map"] / args.count),
            "support_vector_rmse": math.sqrt(raw["support"] / raw["support_n"]),
            "image_mse": raw["image"] / args.count,
            "minimum_jacobian": raw["minimum"],
            "identity_outputs": raw["fallback"],
        })
    print(json.dumps({
        "experiment": "phase7_unknown_carrier_bank",
        "candidate_cycles": list(CYCLES),
        "target_cycles": list(target_cycles),
        "count": args.count,
        "seed": args.seed,
        "batch": args.batch,
        "window": args.window,
        "ridge": args.ridge,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": 512,
        "confusion": confusion.tolist(),
        "exact_frequency_accuracy": float(sum(
            confusion[i, j] for i, target in enumerate(target_cycles)
            for j, candidate in enumerate(CYCLES) if target == candidate
        ) / confusion.sum()),
        "rows": rows,
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
