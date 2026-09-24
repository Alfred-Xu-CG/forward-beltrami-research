"""Train a 1025² P1 image-to-latent layer on independent local affine maps."""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_local_affine_inference import (
    affine_window_map, infer_affine_two_start,
)
from phase7_test_local_affine_packet import target_parameters
from phase7_test_unknown_carrier_bank import (
    dataset as high_dataset, multicarrier_packet,
)
from phase7_train_mixed_scale_residual import (
    MixedScaleResidualEncoder, evaluate,
)


def affine_dataset(
    count: int, seed: int, device: torch.device,
    batch: int, table: StructuredDenseQueryTable,
) -> tuple[torch.Tensor, ...]:
    _, moving, high_target, high_support, _, frequency = high_dataset(
        count, seed, device, batch, table,
        (80.5, 96.5, 112.5, 127.5))
    origins, vectors, matrices = target_parameters(
        count, seed, device)
    identity = affine_window_map(
        origins[:1], torch.zeros_like(vectors[:1]),
        torch.zeros_like(matrices[:1]),
        1025, torch.float64)
    targets, fixed_parts, low_supports = [], [], []
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        low = affine_window_map(
            origins[start:stop], vectors[start:stop],
            matrices[start:stop], 1025, torch.float64)
        total = low + high_target[start:stop] - identity
        query = table.interpolate(
            total.reshape(stop - start, -1, 2))
        fixed = F.grid_sample(
            moving[start:stop], 2 * query.float() - 1,
            mode="bilinear", padding_mode="border",
            align_corners=True)
        targets.append(total)
        fixed_parts.append(fixed)
        low_supports.append((low - identity).square().sum(dim=-1) > 0)
    return (torch.cat(fixed_parts), moving, torch.cat(targets),
            None, None, torch.cat(low_supports),
            high_support, frequency)


def affine_dataset_cpu_stream(
    count: int, seed: int, device: torch.device,
    batch: int, table: StructuredDenseQueryTable,
    *, with_masks: bool,
) -> tuple[torch.Tensor, ...]:
    """Reproduce affine_dataset's random draws, keeping only a batch on GPU."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    high_origin = .08 + .67 * torch.rand(count, 2, generator=rng)
    high_magnitude = .00008 + .00007 * torch.rand(
        count, generator=rng)
    high_angle = 2 * torch.pi * torch.rand(
        count, generator=rng)
    high_vector = torch.stack((
        high_magnitude * torch.cos(high_angle),
        high_magnitude * torch.sin(high_angle)), dim=-1)
    target_cycles = (80.5, 96.5, 112.5, 127.5)
    carrier_id = torch.randint(
        len(target_cycles), (count,), generator=rng)
    cycles = torch.tensor(target_cycles)[carrier_id]
    keys = torch.randint(
        -22, 23, (count, 8, 2), generator=rng)
    phases = 2 * torch.pi * torch.rand(
        count, 8, generator=rng)
    weights = torch.randn(count, 8, generator=rng)
    low_origin, low_vector, low_matrix = target_parameters(
        count, seed, device)
    identity = affine_window_map(
        low_origin[:1], torch.zeros_like(low_vector[:1]),
        torch.zeros_like(low_matrix[:1]),
        1025, torch.float64)
    axis = torch.arange(
        512, device=device, dtype=torch.float32) / 511
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    fixed_parts, moving_parts, target_parts = [], [], []
    low_masks, high_masks = [], []
    with torch.no_grad():
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            high_target, high_support = multicarrier_packet(
                1025, high_origin[start:stop],
                high_vector[start:stop], cycles[start:stop],
                device, torch.float64)
            kx = keys[start:stop, :, 0, None, None].to(device)
            ky = keys[start:stop, :, 1, None, None].to(device)
            phase = phases[start:stop, :, None, None].to(device)
            weight = weights[start:stop, :, None, None].to(device)
            texture = (weight * torch.sin(
                2 * torch.pi * (kx * xx + ky * yy) + phase)
                       ).sum(dim=1)
            texture = (
                texture - texture.mean(dim=(-1, -2), keepdim=True)) / (
                    texture.std(dim=(-1, -2), keepdim=True) + 1e-6)
            moving = texture[:, None].contiguous()
            low = affine_window_map(
                low_origin[start:stop], low_vector[start:stop],
                low_matrix[start:stop], 1025, torch.float64)
            target = low + high_target - identity
            query = table.interpolate(
                target.reshape(stop - start, -1, 2))
            fixed = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            fixed_parts.append(fixed.cpu())
            moving_parts.append(moving.cpu())
            target_parts.append(target.cpu())
            if with_masks:
                low_masks.append((
                    (low - identity).square().sum(dim=-1) > 0).cpu())
                high_masks.append(high_support.cpu())
    core = (torch.cat(fixed_parts), torch.cat(moving_parts),
            torch.cat(target_parts))
    if not with_masks:
        return core
    return (core + (None, None, torch.cat(low_masks),
                    torch.cat(high_masks), cycles))


def preprocess(
    data: tuple[torch.Tensor, ...], batch: int,
    *, device: torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    fixed, moving = data[:2]
    origins, vectors, proposals = [], [], []
    with torch.no_grad():
        for start in range(0, len(fixed), batch):
            stop = min(start + batch, len(fixed))
            input_fixed = fixed[start:stop]
            input_moving = moving[start:stop]
            if device is not None:
                input_fixed = input_fixed.to(device)
                input_moving = input_moving.to(device)
            origin, vector, proposal = infer_affine_two_start(
                input_fixed, input_moving,
                steps=8, damping=.001)
            origins.append(origin.cpu() if device is not None else origin)
            vectors.append(vector.cpu() if device is not None else vector)
            proposals.append(proposal.cpu() if device is not None else proposal)
    return (torch.cat(origins), torch.cat(vectors),
            None, torch.cat(proposals))


def evaluate_cpu_stream(
    model: MixedScaleResidualEncoder,
    data: tuple[torch.Tensor, ...],
    parameters: tuple[torch.Tensor, ...],
    batch_size: int, device: torch.device,
) -> dict[str, float | int]:
    totals = dict(map=0., low=0., high=0.,
                  low_n=0, high_n=0,
                  image=0., freq=0.,
                  minimum=float("inf"), fallback=0)
    for start in range(0, len(data[0]), batch_size):
        stop = min(start + batch_size, len(data[0]))
        count = stop - start
        batch_data = tuple(
            item[start:stop].to(device, non_blocking=True)
            if item is not None else None for item in data)
        batch_parameters = tuple(
            item[start:stop].to(device, non_blocking=True)
            if item is not None else None for item in parameters)
        row = evaluate(
            model, batch_data, batch_parameters, count)
        low_n = int((batch_data[5] > 0).sum())
        high_n = int((batch_data[6] > 0).sum())
        totals["map"] += count * row["map_vector_rmse"] ** 2
        totals["low"] += low_n * row["low_support_vector_rmse"] ** 2
        totals["high"] += high_n * row["high_support_vector_rmse"] ** 2
        totals["low_n"] += low_n
        totals["high_n"] += high_n
        totals["image"] += count * row["image_mse"]
        totals["freq"] += count * row["frequency_mae"]
        totals["minimum"] = min(
            totals["minimum"], row["minimum_jacobian"])
        totals["fallback"] += row["identity_outputs"]
    count = len(data[0])
    return dict(
        map_vector_rmse=(totals["map"] / count) ** .5,
        low_support_vector_rmse=(
            totals["low"] / totals["low_n"]) ** .5,
        high_support_vector_rmse=(
            totals["high"] / totals["high_n"]) ** .5,
        image_mse=totals["image"] / count,
        frequency_mae=totals["freq"] / count,
        minimum_jacobian=totals["minimum"],
        identity_outputs=totals["fallback"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objective", choices=("map", "image"),
                        default="map")
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=41917)
    parser.add_argument("--test-seed", type=int, default=93713)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--save", default=None)
    parser.add_argument("--stream-cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(76381)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    torch.cuda.reset_peak_memory_stats(device)
    if args.stream_cpu:
        train = affine_dataset_cpu_stream(
            args.train_count, args.train_seed,
            device, args.batch, table, with_masks=False)
        test = affine_dataset_cpu_stream(
            args.test_count, args.test_seed,
            device, args.batch, table, with_masks=True)
    else:
        train = affine_dataset(
            args.train_count, args.train_seed,
            device, args.batch, table)
        test = affine_dataset(
            args.test_count, args.test_seed,
            device, args.batch, table)
        # Training uses only images and map; release unused masks.
        train = train[:3]
    tick = time.perf_counter()
    train_params = preprocess(
        train, args.batch,
        device=device if args.stream_cpu else None)
    test_params = preprocess(
        test, args.batch,
        device=device if args.stream_cpu else None)
    preprocessing_seconds = time.perf_counter() - tick
    preparation_peak = torch.cuda.max_memory_allocated(device)
    model = MixedScaleResidualEncoder(
        device, table).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr)
    baseline = (evaluate_cpu_stream(
        model, test, test_params, args.batch, device)
        if args.stream_cpu else
        evaluate(model, test, test_params, args.batch))
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    coarse_grad_steps = 0
    fine_grad_steps = 0
    nonfinite_grad_steps = 0
    losses = []
    for step in range(args.steps):
        ids = torch.randint(
            args.train_count, (args.batch,),
            device="cpu" if args.stream_cpu else device)
        fixed, moving, target = (
            train[i][ids].to(device, non_blocking=True)
            for i in (0, 1, 2))
        origin = train_params[0][ids].to(device, non_blocking=True)
        vector = train_params[1][ids].to(device, non_blocking=True)
        proposed_low = train_params[3][ids].to(device, non_blocking=True)
        output, _ = model(
            fixed, moving, origin, vector,
            proposed_low=proposed_low)
        if args.objective == "map":
            loss = 1e8 * (
                output - target).square().sum(dim=-1).mean()
        else:
            query = table.interpolate(
                output.reshape(args.batch, -1, 2))
            warped = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            loss = 1e6 * (warped - fixed).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        coarse_gradient = model.coarse_net[-1].weight.grad
        fine_gradient = model.fine_net[-1].weight.grad
        coarse_grad_steps += int((coarse_gradient != 0).any())
        fine_grad_steps += int((fine_gradient != 0).any())
        nonfinite_grad_steps += int(
            not torch.isfinite(coarse_gradient).all() or
            not torch.isfinite(fine_gradient).all())
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps(dict(
                step=step + 1,
                loss_last_100_mean=sum(losses[-100:]) / 100,
            )), flush=True)
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated(device)
    final = (evaluate_cpu_stream(
        model, test, test_params, args.batch, device)
        if args.stream_cpu else
        evaluate(model, test, test_params, args.batch))
    if args.save:
        torch.save(dict(
            model=model.state_dict(), config=vars(args)), args.save)
    print(json.dumps(dict(
        experiment="phase7_local_affine_image_training",
        objective=args.objective,
        train_count=args.train_count,
        test_count=args.test_count,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        steps=args.steps, batch=args.batch,
        stream_cpu=args.stream_cpu,
        control_vertices=1025 ** 2,
        control_faces=2 * 1024 ** 2,
        image_side=512,
        parameters=sum(p.numel() for p in model.parameters()),
        baseline=baseline, final=final,
        coarse_gradient_nonzero_steps=coarse_grad_steps,
        fine_gradient_nonzero_steps=fine_grad_steps,
        gradient_nonfinite_steps=nonfinite_grad_steps,
        low_preprocessing_seconds=preprocessing_seconds,
        mean_training_step_seconds=duration / args.steps,
        peak_cuda_allocated_bytes=peak,
        peak_cuda_preparation_allocated_bytes=preparation_peak,
        torch_version=torch.__version__,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
