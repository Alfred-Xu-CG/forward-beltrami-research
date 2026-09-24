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
from phase7_test_unknown_carrier_bank import dataset as high_dataset
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


def preprocess(
    data: tuple[torch.Tensor, ...], batch: int,
) -> tuple[torch.Tensor, ...]:
    fixed, moving = data[:2]
    origins, vectors, proposals = [], [], []
    with torch.no_grad():
        for start in range(0, len(fixed), batch):
            stop = min(start + batch, len(fixed))
            origin, vector, proposal = infer_affine_two_start(
                fixed[start:stop], moving[start:stop],
                steps=8, damping=.001)
            origins.append(origin)
            vectors.append(vector)
            proposals.append(proposal)
    return (torch.cat(origins), torch.cat(vectors),
            None, torch.cat(proposals))


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
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(76381)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    train = affine_dataset(
        args.train_count, args.train_seed,
        device, args.batch, table)
    test = affine_dataset(
        args.test_count, args.test_seed,
        device, args.batch, table)
    # Training uses only images and map; release masks and high-only fields.
    train = train[:3]
    tick = time.perf_counter()
    train_params = preprocess(train, args.batch)
    test_params = preprocess(test, args.batch)
    preprocessing_seconds = time.perf_counter() - tick
    model = MixedScaleResidualEncoder(
        device, table).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr)
    baseline = evaluate(model, test, test_params, args.batch)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    coarse_grad_steps = 0
    fine_grad_steps = 0
    nonfinite_grad_steps = 0
    losses = []
    for step in range(args.steps):
        ids = torch.randint(
            args.train_count, (args.batch,), device=device)
        fixed, moving, target = (
            train[i][ids] for i in (0, 1, 2))
        origin = train_params[0][ids]
        vector = train_params[1][ids]
        proposed_low = train_params[3][ids]
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
    final = evaluate(model, test, test_params, args.batch)
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
        torch_version=torch.__version__,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
