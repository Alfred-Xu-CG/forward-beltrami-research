"""Search for a folded fixed-P1 export of an exact two-factor PL homeomorphism."""

from __future__ import annotations

import argparse
import json

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import DenseMonotoneGridLayer
from qcopt.neural_bijection.tutte.composition import compose_control_maps
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def minimum_area_ratio(control: torch.Tensor) -> float:
    side = control.shape[1]
    a = control[:, :-1, :-1]
    b = control[:, :-1, 1:]
    c = control[:, 1:, 1:]
    d = control[:, 1:, :-1]
    first = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    second = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
    return min(first.min().item(), second.min().item()) * (side - 1)**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=17)
    parser.add_argument("--trials-per-scale", type=int, default=200)
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    if args.side < 3 or args.trials_per_scale < 1:
        raise ValueError("invalid side or trial count")
    torch.manual_seed(20260923)
    vertical = DenseMonotoneGridLayer(args.side, axis="vertical")
    horizontal = DenseMonotoneGridLayer(args.side, axis="horizontal")
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.side, width=args.side)
    table.prepare(device=torch.device("cpu"), dtype=torch.float64)
    scales = (0.5, 2.0, 4.0, 8.0, 16.0)
    best = {"export_minimum_area_ratio": float("inf")}
    tested = 0
    for scale in scales:
        for trial in range(args.trials_per_scale):
            logits = (
                scale * torch.randn(1, args.side - 1, dtype=torch.float64),
                scale * torch.randn(1, args.side, args.side - 1, dtype=torch.float64),
                scale * torch.randn(1, args.side - 1, dtype=torch.float64),
                scale * torch.randn(1, args.side, args.side - 1, dtype=torch.float64),
            )
            first = vertical(logits[0], logits[1])
            second = horizontal(logits[2], logits[3])
            factor_areas = (minimum_area_ratio(first), minimum_area_ratio(second))
            if min(factor_areas) <= 0:
                raise AssertionError("an exact factor lost its own positive-face contract")
            sampled = compose_control_maps(
                (table, table), (first.reshape(1, -1, 2), second.reshape(1, -1, 2))
            )
            exported_area = minimum_area_ratio(sampled)
            tested += 1
            if exported_area < best["export_minimum_area_ratio"]:
                best = {
                    "scale": scale,
                    "trial_at_scale": trial,
                    "factor_minimum_area_ratios": factor_areas,
                    "export_minimum_area_ratio": exported_area,
                }
                if exported_area < -1e-10 and args.save_state:
                    torch.save({
                        "side": args.side,
                        "scale": scale,
                        "trial_at_scale": trial,
                        "logits": tuple(value.detach().clone() for value in logits),
                        "factor_minimum_area_ratios": factor_areas,
                        "export_minimum_area_ratio": exported_area,
                    }, args.save_state)
            if exported_area < -1e-10:
                print(json.dumps({
                    "question": "can original-grid P1 resampling fold an exact composition of valid P1 factors?",
                    "control_side": args.side,
                    "control_faces": mesh.n_faces,
                    "factor_layers": 2,
                    "dtype": "float64",
                    "tested_trials": tested,
                    "found_folded_export": True,
                    "best": best,
                    "state": args.save_state,
                    "exact_composition_topology": "homeomorphism on common refinement; the sampled export is a different map",
                }, sort_keys=True))
                return
    print(json.dumps({
        "question": "can original-grid P1 resampling fold an exact composition of valid P1 factors?",
        "control_side": args.side,
        "control_faces": mesh.n_faces,
        "factor_layers": 2,
        "dtype": "float64",
        "tested_trials": tested,
        "found_folded_export": False,
        "best": best,
        "state": None,
        "warning": "failure to find a fold is not a topology theorem for resampling",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
