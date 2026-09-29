"""Stream a certified fixed-diagonal P1 map over original WSI pixel centers.

This evaluates coordinates only; it does not read or warp the source image.
The saved map acts fixed-canvas -> moving-canvas and may include a positive
post-affine. Tile boundaries never create independent deformation maps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel,
    original_pixel_to_canvas_unit,
    p1_at_queries,
)
from tools.digital_q1_real_eval import load_effective_vertices


def evaluate_native_tile(
    vertices: np.ndarray, fixed_layout: dict, moving_layout: dict,
    canvas_side: int, x0: int, y0: int, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native moving pixel coordinates and inside-canvas mask."""
    if min(x0, y0) < 0 or min(width, height) < 1:
        raise ValueError("nonempty nonnegative original-image rectangle required")
    ys, xs = np.meshgrid(
        np.arange(y0, y0 + height, dtype=np.float64),
        np.arange(x0, x0 + width, dtype=np.float64), indexing="ij",
    )
    fixed_xy = np.stack((xs, ys), axis=-1).reshape(-1, 2)
    fixed_unit = original_pixel_to_canvas_unit(fixed_xy, fixed_layout, canvas_side)
    if not np.all((fixed_unit >= 0) & (fixed_unit <= 1)):
        raise ValueError("fixed native pixel centers leave the declared canvas")
    moving_unit = p1_at_queries(vertices, fixed_unit)
    moving_xy = canvas_unit_to_original_pixel(moving_unit, moving_layout, canvas_side)
    size = np.asarray(moving_layout["original_hw"][::-1], dtype=np.float64)
    inside = np.all((moving_xy >= 0) & (moving_xy < size), axis=1)
    return moving_xy.reshape(height, width, 2), inside.reshape(height, width)


def stream_roi(
    vertices: np.ndarray, layout: dict, *, roi_xywh: tuple[int, int, int, int],
    tile_side: int,
) -> dict:
    x0, y0, width, height = roi_xywh
    fixed_hw = layout["fixed"]["original_hw"]
    if (tile_side < 1 or min(x0, y0) < 0 or min(width, height) < 1
            or x0 + width > fixed_hw[1] or y0 + height > fixed_hw[0]):
        raise ValueError("ROI must be nonempty and inside original fixed slide")
    side = int(layout["side"])
    total = inside_count = tiles = 0
    minimum = np.array([np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf])
    probes: list[dict] = []
    start = time.perf_counter()
    for tile_y in range(y0, y0 + height, tile_side):
        nh = min(tile_side, y0 + height - tile_y)
        for tile_x in range(x0, x0 + width, tile_side):
            nw = min(tile_side, x0 + width - tile_x)
            mapped, inside = evaluate_native_tile(
                vertices, layout["fixed"], layout["moving"], side,
                tile_x, tile_y, nw, nh,
            )
            if not np.isfinite(mapped).all():
                raise ArithmeticError("nonfinite native coordinates")
            minimum = np.minimum(minimum, mapped.min(axis=(0, 1)))
            maximum = np.maximum(maximum, mapped.max(axis=(0, 1)))
            total += nh * nw
            inside_count += int(inside.sum())
            tiles += 1
            # At each tile's exact four pixel centers, re-evaluate a direct
            # single-point query to catch tile-origin/half-pixel mistakes.
            sample = ((0, 0), (0, nw - 1), (nh - 1, 0), (nh - 1, nw - 1))
            unique = sorted(set(sample))
            input_xy = np.asarray(
                [[tile_x + j, tile_y + i] for i, j in unique], dtype=np.float64,
            )
            unit = original_pixel_to_canvas_unit(input_xy, layout["fixed"], side)
            direct = canvas_unit_to_original_pixel(
                p1_at_queries(vertices, unit), layout["moving"], side,
            )
            from_tile = np.asarray([mapped[i, j] for i, j in unique])
            if not np.array_equal(direct, from_tile):
                raise AssertionError("direct and tile P1 coordinates differ")
            if len(probes) < 12:
                probes.append({"fixed_native_xy": input_xy[0].tolist(),
                               "moving_native_xy": direct[0].tolist()})
    elapsed = time.perf_counter() - start
    return {
        "question": "native-resolution P1 coordinate query with one global map",
        "roi_fixed_native_xywh": list(roi_xywh),
        "control_side": int(vertices.shape[0]),
        "canvas_side": side,
        "query_pixels": total,
        "tiles": tiles,
        "tile_side": tile_side,
        "moving_original_inside_fraction": inside_count / total,
        "moving_original_coordinate_min_xy": minimum.tolist(),
        "moving_original_coordinate_max_xy": maximum.tolist(),
        "elapsed_seconds_including_direct_tile_corner_checks": elapsed,
        "queries_per_second": total / elapsed,
        "direct_tile_corner_checks_bitwise_equal": True,
        "first_tile_probes": probes,
        "scope": "coordinate evaluation only; not native-image sampling or registration accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--tile-side", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output already exists")
    vertices, certificate = load_effective_vertices(args.map)
    if not certificate["composite_representation_valid"]:
        raise ValueError("saved P1/Q1 residual and affine certificate invalid")
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    result = stream_roi(vertices, layout, roi_xywh=tuple(args.roi), tile_side=args.tile_side)
    result["map"] = str(args.map)
    result["layout"] = str(args.layout)
    result["saved_map_certificate"] = certificate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
