"""Target-coordinate oracle diagnosis of all-edge bounded conductance capacity.

This is NOT image-to-latent training: the analytic target enters the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from phase6_train_multisample_image import make_dataset
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer
from qcopt.neural_bijection.dense.sine_pcg_tutte import _minimum_signed_area_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--item", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--maximum-conductance", type=float, default=16.0)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    if not 0 <= args.item < 32:
        raise ValueError("item must index the fixed 32-sample training set")
    if args.gradient_weight < 0:
        raise ValueError("gradient weight must be nonnegative")
    device = torch.device(args.device)
    _, _, _, coefficients = make_dataset(32, args.image_side, 55101,
                                          target_family="high32", return_coefficients=True)
    ax, ay, af = (float(value) for value in coefficients[args.item])
    layer = SinePreconditionedTutteLayer(
        args.side, maximum_conductance=args.maximum_conductance,
        tolerance=1e-10, max_iterations=120,
    ).to(device)
    source = layer._source[None]
    xx, yy = source[..., 0], source[..., 1]
    low = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    high = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
    target = torch.stack((xx + ax * low + af * high, yy + ay * low + af * high), dim=-1)
    target_minimum_area_ratio = _minimum_signed_area_ratio(target)
    logits = [
        torch.nn.Parameter(torch.zeros(1, args.side, args.side - 1, device=device, dtype=torch.float64)),
        torch.nn.Parameter(torch.zeros(1, args.side - 1, args.side, device=device, dtype=torch.float64)),
        torch.nn.Parameter(torch.zeros(1, args.side - 1, args.side - 1, device=device, dtype=torch.float64)),
    ]
    optimizer = torch.optim.Adam(logits, lr=args.learning_rate)
    records = []
    elapsed_steps = []
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_all = time.perf_counter()
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        mapped = layer(*logits)
        error = mapped - target
        map_mse = error.square().mean()
        gradient_mse = 0.5 * (args.side - 1) ** 2 * (
            (error[:, :, 1:] - error[:, :, :-1]).square().mean()
            + (error[:, 1:] - error[:, :-1]).square().mean()
        )
        loss = map_mse + args.gradient_weight * gradient_mse
        if step < args.steps:
            loss.backward()
            optimizer.step()
        synchronize()
        if step > 0:
            elapsed_steps.append(time.perf_counter() - began)
        if step == 0 or step == args.steps or (step + 1) % 100 == 0:
            projection = ((mapped - source) * high[..., None]).sum(dim=(1, 2)) / high.square().sum()
            records.append({
                "step": step,
                "vertex_map_rmse": math.sqrt(map_mse.item()),
                "target_gradient_mse": gradient_mse.item(),
                "total_objective": loss.item(),
                "minimum_signed_area_ratio": layer.last_forward_stats["minimum_signed_area_ratio"],
                "true_relative_residual": layer.last_forward_stats["true_relative_residual"],
                "projected_fine_x_amplitude": float(projection[0, 0]),
                "projected_fine_y_amplitude": float(projection[0, 1]),
                "forward_iterations": layer.last_forward_stats["iterations"],
                "backward_iterations": layer.last_backward_stats.get("iterations") if step < args.steps else None,
                "elapsed_seconds": time.perf_counter() - start_all,
            })
    if args.save_state:
        torch.save({"logits": [value.detach().cpu() for value in logits], "args": vars(args),
                    "coefficients": (ax, ay, af)}, args.save_state)
    print(json.dumps({
        "method": "bounded_sine_pcg_direct_target_coordinate_oracle",
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "image_side_used_only_for_coefficient_generation": args.image_side,
        "target_family": "high32",
        "target_seed": 55101,
        "target_set_size": 32,
        "target_item": args.item,
        "target_coefficients": {"ax": ax, "ay": ay, "af": af},
        "target_sampled_p1_minimum_area_ratio": target_minimum_area_ratio,
        "conductance_interval": [1.0, args.maximum_conductance],
        "device": str(device),
        "dtype": "float64",
        "learning_rate": args.learning_rate,
        "gradient_weight": args.gradient_weight,
        "steps": args.steps,
        "median_full_training_step_seconds": statistics.median(elapsed_steps) if elapsed_steps else None,
        "total_seconds": time.perf_counter() - start_all,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
