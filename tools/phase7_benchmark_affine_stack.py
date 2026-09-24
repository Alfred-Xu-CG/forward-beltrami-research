"""Batch-4 latency and VJP cost of the local-affine 1025² P1 stack."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_local_affine_inference import (
    affine_window_map, energy_window_origin,
    fit_affine_window,
)
from phase7_matched_low_patch import (
    matched_low_patch, multistart_refine_low_params,
)
from phase7_test_local_affine_packet import target_parameters
from phase7_test_unknown_carrier_bank import dataset as high_dataset
from phase7_train_mixed_scale_residual import MixedScaleResidualEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=28371)
    parser.add_argument("--affine-steps", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    _, moving, high_target, _, _, _ = high_dataset(
        args.batch, args.seed, device, args.batch, table,
        (80.5, 96.5, 112.5, 127.5))
    origin, vector, matrix = target_parameters(
        args.batch, args.seed, device)
    low_target = affine_window_map(
        origin, vector, matrix, 1025, torch.float64)
    model = MixedScaleResidualEncoder(device, table).to(device)
    target = (low_target + high_target -
              model.decoder.fine_identity)
    query = table.interpolate(target.reshape(args.batch, -1, 2))
    fixed = F.grid_sample(
        moving, 2 * query.float() - 1,
        mode="bilinear", padding_mode="border",
        align_corners=True)

    def infer_proposal() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p, v, _ = matched_low_patch(fixed, moving)
        p, v, _ = multistart_refine_low_params(
            fixed, moving, p, v, steps=4, directions=4)
        energy_p = energy_window_origin(fixed, moving)
        proposals = []
        for initial_p in (p, energy_p):
            p6, v6, m6, _ = fit_affine_window(
                fixed, moving, initial_p, v,
                steps=args.affine_steps, moving_origin=False,
                damping=.001)
            p8, v8, m8, loss = fit_affine_window(
                fixed, moving, p6, v6, m6,
                steps=args.affine_steps, moving_origin=True,
                damping=.001)
            proposals.append((p8, v8, m8, loss))
        choose_second = proposals[1][3] < proposals[0][3]
        best = [torch.where(
            choose_second.reshape((-1,) + (1,) * (proposals[0][i].ndim - 1)),
            proposals[1][i], proposals[0][i])
                for i in range(3)]
        low = affine_window_map(
            best[0], best[1], best[2],
            257, torch.float64)
        return p, v, low

    durations = dict(preprocess=[], forward=[], backward=[])
    gradient_finite = True
    gradient_nonzero = [False, False]
    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(args.repeats + 1):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with torch.no_grad():
            p, v, proposal = infer_proposal()
        torch.cuda.synchronize(device)
        preprocess_duration = time.perf_counter() - tick
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        output, _ = model(
            fixed, moving, p, v,
            proposed_low=proposal)
        loss = 1e8 * (
            output - target).square().sum(dim=-1).mean()
        torch.cuda.synchronize(device)
        forward_duration = time.perf_counter() - tick
        tick = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_duration = time.perf_counter() - tick
        gradients = (
            model.coarse_net[-1].weight.grad,
            model.fine_net[-1].weight.grad)
        gradient_finite &= all(
            torch.isfinite(gradient).all() for gradient in gradients)
        gradient_nonzero = [
            previous or bool((gradient != 0).any())
            for previous, gradient in zip(gradient_nonzero, gradients)]
        if iteration:
            durations["preprocess"].append(preprocess_duration)
            durations["forward"].append(forward_duration)
            durations["backward"].append(backward_duration)
    print(json.dumps(dict(
        experiment="phase7_affine_stack_benchmark",
        batch=args.batch, seed=args.seed,
        repeats=args.repeats,
        affine_steps=args.affine_steps,
        control_vertices=1025 ** 2,
        image_side=512,
        median_preprocess_seconds=statistics.median(
            durations["preprocess"]),
        median_forward_seconds=statistics.median(
            durations["forward"]),
        median_backward_seconds=statistics.median(
            durations["backward"]),
        map_vector_rmse=float((
            output - target).square().sum(
                dim=-1).mean().sqrt()),
        coarse_gradient_nonzero=gradient_nonzero[0],
        fine_gradient_nonzero=gradient_nonzero[1],
        gradients_finite=gradient_finite,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        torch_version=torch.__version__,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
