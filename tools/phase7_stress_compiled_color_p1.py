"""Compare eager/compiled F1 under extreme finite latents on a dense P1 grid."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense.colored_vertex_relaxation import (
    SafeColoredVertexRelaxation,
)
from qcopt.neural_bijection.dense.multilevel_forward_p1 import certify_p1_or_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=4097)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--amplitude", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.side < 3 or args.count < 2 or args.amplitude <= 0:
        raise ValueError("invalid side/count/amplitude")
    device = torch.device(args.device)
    axis = torch.arange(args.side, dtype=torch.float32, device=device) / (args.side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    area_floor = identity.new_full((1,), .05 / (args.side - 1) ** 2)
    eager = SafeColoredVertexRelaxation(
        args.side, motion_mode="radial", raw_span=2., index_mode="generated",
    ).to(device)
    compiled = SafeColoredVertexRelaxation(
        args.side, motion_mode="radial", raw_span=2., index_mode="generated",
    ).to(device)
    from torch import _dynamo
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
    compiled._update_color = torch.compile(compiled._update_color)
    generator = torch.Generator(device=device).manual_seed(1949)
    eager_times = []
    compiled_times = []
    eager_minima = []
    compiled_minima = []
    eager_accepted = 0
    compiled_accepted = 0
    maximum_coordinate_gap = 0.
    squared_coordinate_gap = 0.
    large_gap_vertices = 0

    def minimum_jacobian(values: torch.Tensor) -> float:
        # Recompute from the *actual float32 coordinates* in float64.
        mapped = values.double()
        sw = mapped[:, :-1, :-1]
        se = mapped[:, :-1, 1:]
        ne = mapped[:, 1:, 1:]
        nw = mapped[:, 1:, :-1]
        def cross(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        return float(torch.minimum(
            cross(se - sw, ne - sw).amin(),
            cross(ne - sw, nw - sw).amin(),
        ) * (args.side - 1) ** 2)

    with torch.no_grad():
        for _ in range(args.count):
            logits = args.amplitude * torch.randn(
                1, args.side - 2, args.side - 2, 2,
                device=device, generator=generator,
            )
            outputs = []
            for layer, times, minima in (
                (eager, eager_times, eager_minima),
                (compiled, compiled_times, compiled_minima),
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                began = time.perf_counter()
                result = layer(identity, logits, area_floor=area_floor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(time.perf_counter() - began)
                minima.append(minimum_jacobian(result))
                _, accepted = certify_p1_or_identity(result, identity)
                if layer is eager:
                    eager_accepted += int(accepted.item())
                else:
                    compiled_accepted += int(accepted.item())
                outputs.append(result)
            gap = outputs[0] - outputs[1]
            maximum_coordinate_gap = max(
                maximum_coordinate_gap,
                float(gap.abs().amax()),
            )
            squared_coordinate_gap += float(gap.square().sum())
            large_gap_vertices += int((
                gap.abs().amax(dim=-1) > 0.01 / (args.side - 1)
            ).sum())
    print(json.dumps({
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "count": args.count,
        "amplitude": args.amplitude,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "eager_accepted": eager_accepted,
        "compiled_accepted": compiled_accepted,
        "eager_minimum_jacobians": eager_minima,
        "compiled_minimum_jacobians": compiled_minima,
        "maximum_coordinate_gap": maximum_coordinate_gap,
        "coordinate_rmse_gap": (
            squared_coordinate_gap / (args.count * args.side ** 2 * 2)
        ) ** .5,
        "vertices_with_gap_over_one_percent_cell": large_gap_vertices,
        "eager_hot_forward_median_seconds": statistics.median(eager_times[1:]),
        "compiled_hot_forward_median_seconds": statistics.median(compiled_times[1:]),
        "eager_first_seconds": eager_times[0],
        "compiled_first_seconds": compiled_times[0],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
