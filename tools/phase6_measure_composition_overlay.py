"""Count source-triangle incidences with second-layer horizontal grid strips.

This is a conservative overlay-complexity diagnostic, not a construction of
the exact common-refinement mesh and not a claim about minimal PL complexity.
"""

from __future__ import annotations

import argparse
import json

import torch

from qcopt.neural_bijection.dense import DenseMonotoneGridLayer


def strip_incidences(control: torch.Tensor) -> dict[str, int | float]:
    """One first-factor image; count open second-grid row strips per face."""
    if control.ndim != 3 or control.shape[0] != control.shape[1] or control.shape[-1] != 2:
        raise ValueError("control must have shape (side, side, 2)")
    n = control.shape[0] - 1
    y00 = control[:-1, :-1, 1]
    y10 = control[:-1, 1:, 1]
    y01 = control[1:, :-1, 1]
    y11 = control[1:, 1:, 1]
    triangles = ((y00, y10, y11), (y00, y11, y01))
    counts = []
    tolerance = 2.0e-5  # in grid-cell units; exact dyadic rows are represented exactly here
    for vertices in triangles:
        minimum = torch.minimum(torch.minimum(vertices[0], vertices[1]), vertices[2]) * n
        maximum = torch.maximum(torch.maximum(vertices[0], vertices[1]), vertices[2]) * n
        lower = torch.floor(minimum + tolerance)
        upper = torch.ceil(maximum - tolerance)
        count = (upper - lower).to(torch.int64)
        if torch.any(count < 1):
            raise AssertionError("a nondegenerate image face did not meet a row strip")
        counts.append(count.flatten())
    all_counts = torch.cat(counts)
    return {
        "source_faces": int(all_counts.numel()),
        "total_row_strip_incidences": int(all_counts.sum().item()),
        "mean_strips_per_face": float(all_counts.double().mean().item()),
        "faces_crossing_row_line": int((all_counts > 1).sum().item()),
        "fraction_faces_crossing_row_line": float((all_counts > 1).double().mean().item()),
        "maximum_strips_per_face": int(all_counts.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--state", default=None, help="optional alternating direct-map oracle checkpoint")
    args = parser.parse_args()
    side = args.side
    if side < 3:
        raise ValueError("side must be at least three")
    line = torch.linspace(0.0, 1.0, side, dtype=torch.float32)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)
    decoder = DenseMonotoneGridLayer(side, axis="vertical")
    torch.manual_seed(20260923)
    global_logits = 0.08 * torch.randn(1, side - 1)
    line_logits = 0.08 * torch.randn(1, side, side - 1)
    random_control = decoder(global_logits, line_logits)[0]
    outputs = {"identity": strip_incidences(identity), "random_vertical": strip_incidences(random_control)}
    if args.state:
        checkpoint = torch.load(args.state, map_location="cpu", weights_only=True)
        if checkpoint["method"] != "alternating" or checkpoint["side"] != side or checkpoint["layers"] != 2:
            raise ValueError("checkpoint must be a matching two-layer alternating oracle")
        first_global, _, first_line, _ = checkpoint["parameters"]
        fitted_control = decoder(first_global, first_line)[0]
        outputs["fitted_oracle_first_factor"] = strip_incidences(fitted_control)
    print(json.dumps({
        "question": "second-grid horizontal-strip overlay incidences after the first P1 factor",
        "side": side,
        "first_factor_vertices": side**2,
        "first_factor_faces": 2 * (side - 1)**2,
        "second_grid_horizontal_strips": side - 1,
        "dtype": "float32",
        "checkpoint": args.state,
        "counts": outputs,
        "limitation": "Only horizontal-strip incidences are counted. Diagonal crossings increase the full overlay; equal affine formulas can merge in a minimal representation.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
