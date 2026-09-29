"""Warp one real TIFF pyramid level with a single global fixed-diagonal P1 map.

This is an image-export/seam diagnostic, not a training or accuracy benchmark.
Both TIFFs must match the stored physical-layout dimensions and orientation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import tifffile

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel,
    original_pixel_to_canvas_unit,
    p1_at_queries,
)
from tools.digital_q1_real_eval import load_effective_vertices


def warped_tile(
    vertices: np.ndarray, layout: dict, moving_rgb: np.ndarray,
    fixed_level_hw: tuple[int, int], x0: int, y0: int, width: int, height: int,
) -> np.ndarray:
    """Sample moving pyramid RGB at a fixed pyramid-level output rectangle."""
    if (moving_rgb.ndim != 3 or moving_rgb.shape[2] != 3
            or moving_rgb.dtype != np.uint8
            or min(x0, y0) < 0 or min(width, height) < 1
            or x0 + width > fixed_level_hw[1] or y0 + height > fixed_level_hw[0]):
        raise ValueError("uint8 RGB moving level and fixed output rectangle required")
    fixed_original_hw = np.asarray(layout["fixed"]["original_hw"], dtype=np.float64)
    moving_original_hw = np.asarray(layout["moving"]["original_hw"], dtype=np.float64)
    yy, xx = np.meshgrid(
        np.arange(y0, y0 + height, dtype=np.float64),
        np.arange(x0, x0 + width, dtype=np.float64), indexing="ij",
    )
    fixed_level_xy = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    fixed_scale_xy = fixed_original_hw[::-1] / np.asarray(fixed_level_hw[::-1])
    fixed_original_xy = (fixed_level_xy + .5) * fixed_scale_xy - .5
    unit = original_pixel_to_canvas_unit(
        fixed_original_xy, layout["fixed"], int(layout["side"]),
    )
    if not np.all((unit >= 0) & (unit <= 1)):
        raise ValueError("pyramid fixed centers outside declared canvas")
    mapped_unit = p1_at_queries(vertices, unit)
    moving_original_xy = canvas_unit_to_original_pixel(
        mapped_unit, layout["moving"], int(layout["side"]),
    )
    moving_scale_xy = moving_original_hw[::-1] / np.asarray(moving_rgb.shape[:2][::-1])
    moving_level_xy = (moving_original_xy + .5) / moving_scale_xy - .5
    # An identity map can round a boundary pixel center to -1e-15 or
    # (W-1)+1e-15 after two physical-coordinate conversions. scipy's
    # constant mode would then return white instead of that exact edge pixel.
    # Snap only machine-roundoff-sized excursions; true outside samples still
    # use the declared constant-white policy.
    tolerance = 64 * np.finfo(np.float64).eps * max(moving_rgb.shape[:2])
    for component, extent in ((0, moving_rgb.shape[1]), (1, moving_rgb.shape[0])):
        coordinate = moving_level_xy[:, component]
        coordinate = np.where((coordinate < 0) & (coordinate >= -tolerance),
                              0., coordinate)
        coordinate = np.where((coordinate > extent - 1)
                              & (coordinate <= extent - 1 + tolerance),
                              float(extent - 1), coordinate)
        moving_level_xy[:, component] = coordinate
    coordinates_yx = np.stack((moving_level_xy[:, 1], moving_level_xy[:, 0]))
    sampled = np.stack([
        map_coordinates(moving_rgb[:, :, channel].astype(np.float32), coordinates_yx,
                        order=1, mode="constant", cval=255., prefilter=False)
        for channel in range(3)
    ], axis=-1)
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8).reshape(height, width, 3)


def warp_tiled(vertices: np.ndarray, layout: dict, moving_rgb: np.ndarray,
               fixed_level_hw: tuple[int, int], tile_side: int) -> tuple[np.ndarray, float]:
    if tile_side < 1:
        raise ValueError("positive tile size required")
    output = np.empty((*fixed_level_hw, 3), dtype=np.uint8)
    start = time.perf_counter()
    for y in range(0, fixed_level_hw[0], tile_side):
        for x in range(0, fixed_level_hw[1], tile_side):
            h = min(tile_side, fixed_level_hw[0] - y)
            w = min(tile_side, fixed_level_hw[1] - x)
            output[y:y + h, x:x + w] = warped_tile(
                vertices, layout, moving_rgb, fixed_level_hw, x, y, w, h,
            )
    return output, time.perf_counter() - start


def _read_level(path: Path, level: int, expected_original_hw: list[int], *,
                read_pixels: bool) -> tuple[tuple[int, int], np.ndarray | None]:
    with tifffile.TiffFile(path) as tif:
        if int(tif.pages[0].tags["Orientation"].value) != 1:
            raise ValueError("only top-left TIFF orientation is supported")
        levels = tif.series[0].levels
        if level < 0 or level >= len(levels):
            raise ValueError("requested pyramid level absent")
        if list(levels[0].shape[:2]) != list(expected_original_hw):
            raise ValueError("original TIFF size disagrees with saved physical layout")
        shape = tuple(int(v) for v in levels[level].shape[:2])
        pixels = levels[level].asarray() if read_pixels else None
    return shape, pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--moving-tiff", type=Path, required=True)
    parser.add_argument("--fixed-tiff", type=Path, required=True)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--tile-side", type=int, default=256)
    parser.add_argument("--compare-tile-side", type=int, default=511)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_image.exists() or args.output_report.exists():
        raise FileExistsError("image and report paths must be unused")
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    vertices, certificate = load_effective_vertices(args.map)
    if not certificate["composite_representation_valid"]:
        raise ValueError("invalid saved residual plus post-affine")
    fixed_hw, _ = _read_level(args.fixed_tiff, args.level,
                              layout["fixed"]["original_hw"], read_pixels=False)
    moving_hw, moving = _read_level(args.moving_tiff, args.level,
                                   layout["moving"]["original_hw"], read_pixels=True)
    if (moving is None or moving.ndim != 3 or moving.shape[-1] != 3
            or moving.dtype != np.uint8):
        raise ValueError("uint8 RGB moving TIFF pyramid level required")
    first, first_seconds = warp_tiled(vertices, layout, moving, fixed_hw, args.tile_side)
    second, second_seconds = warp_tiled(vertices, layout, moving, fixed_hw,
                                       args.compare_tile_side)
    max_difference = int(np.abs(first.astype(np.int16) - second.astype(np.int16)).max())
    report = {
        "question": "single global P1 map sampled on actual TIFF pyramid RGB without tile seams",
        "map": str(args.map), "layout": str(args.layout),
        "fixed_tiff": str(args.fixed_tiff), "moving_tiff": str(args.moving_tiff),
        "pyramid_level": args.level, "fixed_level_hw": list(fixed_hw),
        "moving_level_hw": list(moving_hw),
        "tile_sides": [args.tile_side, args.compare_tile_side],
        "elapsed_seconds": [first_seconds, second_seconds],
        "partition_max_absolute_rgb_difference": max_difference,
        "partition_pixels_bitwise_equal": bool(np.array_equal(first, second)),
        "out_of_bounds_rgb": [255, 255, 255],
        "out_of_bounds_policy": "constant white beyond the stated edge-snap tolerance; no edge blend",
        "roundoff_boundary_snap": "coordinates within 64*float64_eps*max(level_shape) of an edge are snapped to the edge",
        "saved_map_certificate": certificate,
        "scope": "real pyramid image export and partition agreement; no landmark or clinical accuracy claim",
    }
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(first).save(args.output_image)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
