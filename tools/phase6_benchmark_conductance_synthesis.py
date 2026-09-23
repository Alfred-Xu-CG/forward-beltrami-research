"""257² target-map diagnostic for explicit conductance synthesis plus PCG."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from phase6_local_conductance_ratio_bound import target_vertices
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer, synthesize_bounded_conductances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--item", type=int, required=True)
    parser.add_argument("--maximum-passes", type=int, default=3)
    parser.add_argument("--gauge", choices=("centered", "range"), default="centered")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = SinePreconditionedTutteLayer(args.side, maximum_conductance=16.0,
                                          tolerance=1e-10, max_iterations=120).to(device)
    target, coefficients = target_vertices(args.side, args.item, device)
    target = target[None]
    xx, yy = layer._source[..., 0], layer._source[..., 1]
    high = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
    results = []
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for passes in range(1, args.maximum_passes + 1):
            synchronize(); start = time.perf_counter()
            logits, stats = synthesize_bounded_conductances(target, passes=passes, gauge=args.gauge, return_stats=True)
            synchronize(); after_synthesis = time.perf_counter()
            mapped = layer(*logits)
            synchronize(); after_solve = time.perf_counter()
            basis_norm = high.square().sum()
            projected = (
                ((mapped - layer._source[None]) * high[None, :, :, None]).sum(dim=(1, 2)) / basis_norm
                if basis_norm > 1e-10 else None
            )
            results.append({
                "passes": passes,
                "synthesis_seconds": after_synthesis - start,
                "solve_and_certificate_seconds": after_solve - after_synthesis,
                "total_forward_seconds": after_solve - start,
                "vertex_map_rmse": (mapped - target).square().mean().sqrt().item(),
                "projected_fine_x_amplitude": projected[0, 0].item() if projected is not None else None,
                "projected_fine_y_amplitude": projected[0, 1].item() if projected is not None else None,
                "target_fine_amplitude": coefficients[2],
                "synthesis_stats": stats,
                "solver_stats": dict(layer.last_forward_stats),
            })
    print(json.dumps({
        "method": "target_displacement_to_bounded_conductance_synthesis_oracle",
        "target_geometry_used_for_synthesis": True,
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "target_set_size": 32,
        "target_seed": 55101,
        "target_item": args.item,
        "target_coefficients": {"ax": coefficients[0], "ay": coefficients[1], "af": coefficients[2]},
        "conductance_interval": [1, 16],
        "base_conductance": 4,
        "gauge": args.gauge,
        "device": str(device),
        "dtype": "float64",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
