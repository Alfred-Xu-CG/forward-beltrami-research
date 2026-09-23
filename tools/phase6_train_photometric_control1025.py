"""Actual multi-step image-only optimization on a million-control P1 layer."""

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
from qcopt.neural_bijection.dense import PhotometricSpectralTutteLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fit-side",type=int,default=128)
    parser.add_argument("--frequencies",default="1,2,4,8,16,32")
    parser.add_argument("--target-family",choices=("high32","high64"),default="high32")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    frequencies=tuple(int(value) for value in args.frequencies.split(","))
    if len(frequencies)!=len(set(frequencies)):
        raise ValueError("frequencies must be unique")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    image_side = 512
    train = tuple(value.to(device) for value in make_dataset(
        32, image_side, 55101, target_family=args.target_family,
        return_coefficients=True))
    heldout = tuple(value.to(device) for value in make_dataset(
        8, image_side, 99317, target_family=args.target_family,
        return_coefficients=True))
    layer = PhotometricSpectralTutteLayer(
        args.side,fit_side=args.fit_side,frequencies=frequencies).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    with torch.no_grad():
        original=(1,2,4,8,16,32)
        original_gains=state["raw_gains"].to(device)
        if original_gains.numel()!=3*len(original):
            raise ValueError("expected a six-frequency trained checkpoint")
        for axis in range(3):
            for new_index,frequency in enumerate(frequencies):
                if frequency in original:
                    layer.raw_mode_gains[axis*len(frequencies)+new_index]=\
                        original_gains[axis*len(original)+original.index(frequency)]
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    query_table = StructuredDenseQueryTable.from_mesh(
        mesh, height=image_side, width=image_side)
    query_table.prepare(device=device, dtype=torch.float32)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    prepare_start = time.perf_counter()
    response_validation = layer.prepare(device=device)
    synchronize()
    prepare_seconds = time.perf_counter() - prepare_start
    prepare_peak = (torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else None)

    def warp_loss(fixed: torch.Tensor, moving: torch.Tensor):
        mapped = layer(fixed, moving)
        query = query_table.interpolate(mapped.reshape(mapped.shape[0], -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), query

    @torch.no_grad()
    def evaluate(dataset) -> dict[str, float]:
        image_mse = []
        map_mse = []
        minimum_area = math.inf
        maximum_residual = 0.0
        for index in range(len(dataset[0])):
            fixed, moving, target = (part[index:index + 1] for part in dataset[:3])
            loss, query = warp_loss(fixed, moving)
            image_mse.append(float(loss))
            map_mse.append(float((query - target).square().mean()))
            minimum_area = min(minimum_area,
                               layer.solver.last_forward_stats["minimum_signed_area_ratio"])
            maximum_residual = max(maximum_residual,
                                   layer.solver.last_forward_stats["true_relative_residual"])
        return {
            "image_mse": statistics.mean(image_mse),
            "query_map_rmse": math.sqrt(statistics.mean(map_mse)),
            "minimum_signed_area_ratio": minimum_area,
            "maximum_true_relative_forward_residual": maximum_residual,
        }

    initial_train = evaluate(train)
    initial_heldout = evaluate(heldout)
    optimizer = torch.optim.Adam([layer.raw_mode_gains], lr=0.01)
    generator = torch.Generator(device="cpu").manual_seed(38819)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times, records = [], [], []
    max_forward_residual = 0.0
    max_adjoint_residual = 0.0
    min_training_area = math.inf
    start_all = time.perf_counter()
    for step in range(args.steps):
        indices = torch.randint(32, (args.batch,), generator=generator).to(device)
        fixed, moving = train[0][indices], train[1][indices]
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        start = time.perf_counter()
        loss, _ = warp_loss(fixed, moving)
        synchronize()
        middle = time.perf_counter()
        max_forward_residual = max(
            max_forward_residual, layer.solver.last_forward_stats["true_relative_residual"])
        min_training_area = min(
            min_training_area, layer.solver.last_forward_stats["minimum_signed_area_ratio"])
        loss.backward()
        synchronize()
        end = time.perf_counter()
        max_adjoint_residual = max(
            max_adjoint_residual, layer.solver.last_backward_stats["true_relative_residual"])
        if layer.raw_mode_gains.grad is None or not torch.isfinite(
                layer.raw_mode_gains.grad).all():
            raise RuntimeError("nonfinite or missing million-control gain VJP")
        optimizer.step()
        forward_times.append(middle - start)
        backward_times.append(end - middle)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            records.append({
                "step": step + 1,
                "sampled_image_mse_before_update": float(loss),
                "gradient_norm": float(layer.raw_mode_gains.grad.norm()),
                "gain_min": float(layer.mode_gains.min()),
                "gain_max": float(layer.mode_gains.max()),
                "elapsed_seconds": time.perf_counter() - start_all,
            })
    train_seconds = time.perf_counter() - start_all
    train_peak = (torch.cuda.max_memory_allocated(device)
                  if device.type == "cuda" else None)
    final_train = evaluate(train)
    final_heldout = evaluate(heldout)
    if args.save_state:
        torch.save({
            "raw_gains": layer.raw_mode_gains.detach().cpu(),
            "args": vars(args),
        }, args.save_state)
    print(json.dumps({
        "method": f"million_control_photometric{3*len(frequencies)}_actual_training",
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": image_side,
        "image_queries": image_side ** 2,
        "fit_side": args.fit_side,
        "frequencies":frequencies,
        "target_family":args.target_family,
        "train_count": 32,
        "heldout_count": 8,
        "batch": args.batch,
        "steps": args.steps,
        "device": str(device),
        "prepare_seconds": prepare_seconds,
        "prepare_peak_cuda_allocated_bytes": prepare_peak,
        "response_validation": response_validation,
        "initial_train": initial_train,
        "initial_heldout": initial_heldout,
        "final_train": final_train,
        "final_heldout": final_heldout,
        "median_full_forward_seconds_after_first": statistics.median(
            forward_times[1:] or forward_times),
        "median_full_vjp_seconds_after_first": statistics.median(
            backward_times[1:] or backward_times),
        "training_seconds": train_seconds,
        "training_peak_cuda_allocated_bytes": train_peak,
        "minimum_training_signed_area_ratio": min_training_area,
        "maximum_training_true_relative_forward_residual": max_forward_residual,
        "maximum_training_true_relative_adjoint_residual": max_adjoint_residual,
        "learned_mode_gains": layer.mode_gains.detach().tolist(),
        "records": records,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
