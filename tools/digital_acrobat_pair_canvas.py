"""Make paired low-resolution WSI canvases without per-image aspect distortion.

Each TIFF carries its own scanner pixel spacing. Both images use one physical
micrometers-per-canvas-pixel scale, then receive white padding to a shared
square. This is a development canvas, not a known slide-to-slide alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import tifffile


def resolution_to_mpp(resolution: tuple[int, int], unit: int) -> float:
    """Convert TIFF pixels per centimeter/inch to micrometers per pixel."""
    if (len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0
            or unit not in (2, 3)):
        raise ValueError("valid physical resolution in inch/cm required")
    micrometers_per_unit = 25_400.0 if unit == 2 else 10_000.0
    result = micrometers_per_unit * resolution[1] / resolution[0]
    if not math.isfinite(result) or result <= 0:
        raise ValueError("valid physical resolution in inch/cm required")
    return result


def compute_layout(
    image_metadata: list[tuple[int, int, float, float]], *, side: int,
) -> list[dict]:
    """Input tuples are (height,width,mpp_y,mpp_x), one for each slide."""
    if len(image_metadata) != 2 or side < 16 or any(
        h < 1 or w < 1 or not all(math.isfinite(z) and z > 0 for z in (my, mx))
        for h, w, my, mx in image_metadata
    ):
        raise ValueError("two positive physical images and side>=16 required")
    largest_extent = max(max(w * mx, h * my)
                         for h, w, my, mx in image_metadata)
    canvas_mpp = largest_extent / side
    layouts = []
    for h, w, my, mx in image_metadata:
        new_w = max(1, min(side, round(w * mx / canvas_mpp)))
        new_h = max(1, min(side, round(h * my / canvas_mpp)))
        left = (side - new_w) // 2
        top = (side - new_h) // 2
        layouts.append({
            "resized_wh": [new_w, new_h],
            "padding_xy": [left, top],
            "canvas_mpp": canvas_mpp,
            "original_mpp_xy": [mx, my],
            "effective_original_to_canvas_scale_xy": [new_w / w, new_h / h],
            "original_center_to_canvas_center":
                "canvas_xy=(original_xy+0.5)*effective_scale_xy-0.5+padding_xy",
        })
    return layouts


def _inspect(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        orientation = int(page.tags["Orientation"].value) if "Orientation" in page.tags else 1
        if orientation != 1:
            raise ValueError(f"unsupported TIFF orientation {orientation}: {path}")
        x_res = page.tags["XResolution"].value
        y_res = page.tags["YResolution"].value
        unit = int(page.tags["ResolutionUnit"].value)
        mpp_x = resolution_to_mpp(x_res, unit)
        mpp_y = resolution_to_mpp(y_res, unit)
        levels = [tuple(level.shape[:2]) for level in tif.series[0].levels]
    return {"source": str(path), "original_hw": list(levels[0]),
            "pyramid_shapes_hw": [list(x) for x in levels],
            "mpp_xy": [mpp_x, mpp_y], "orientation": orientation}


def _render(source: Path, output: Path, layout: dict, *, side: int) -> int:
    new_w, new_h = layout["resized_wh"]
    with tifffile.TiffFile(source) as tif:
        levels = tif.series[0].levels
        candidates = [index for index, level in enumerate(levels)
                      if level.shape[0] >= new_h and level.shape[1] >= new_w]
        index = candidates[-1] if candidates else 0
        data = levels[index].asarray()
    image = Image.fromarray(data).convert("RGB")
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(resized, tuple(layout["padding_xy"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--moving-out", type=Path, required=True)
    parser.add_argument("--fixed-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=512)
    args = parser.parse_args()
    outputs = (args.moving_out, args.fixed_out, args.metadata_out)
    if len(set(outputs)) != 3 or any(path.exists() for path in outputs):
        raise ValueError("distinct new output paths required; existing files are preserved")
    moving, fixed = _inspect(args.moving), _inspect(args.fixed)
    metadata = [moving, fixed]
    layout = compute_layout([
        (item["original_hw"][0], item["original_hw"][1],
         item["mpp_xy"][1], item["mpp_xy"][0])
        for item in metadata
    ], side=args.side)
    for item, current, source, output in zip(
        metadata, layout, (args.moving, args.fixed),
        (args.moving_out, args.fixed_out),
    ):
        item.update(current)
        item["pyramid_level_read"] = _render(source, output, current, side=args.side)
        item["canvas_png"] = str(output)
    result = {"side": args.side, "moving": moving, "fixed": fixed,
              "warning": "common physical pixel scale and aspect, but no true inter-slide alignment"}
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
