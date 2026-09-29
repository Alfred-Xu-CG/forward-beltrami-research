"""Make a shared 512-square image canvas for one same-scale ANHIR sample pair.

The public BIRL examples are both labelled scale-5pc. This script assumes one
pixel of the fixed JPEG has the same physical scale as one pixel of its paired
moving JPEG; it does not invent scanner spacing or registration. It reads no
landmarks. The resize/padding transform is saved for separate evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from tools.digital_acrobat_pair_canvas import compute_layout


def make_canvas(moving: Path, fixed: Path, moving_out: Path, fixed_out: Path,
                metadata_out: Path, *, side: int = 512) -> dict:
    targets = (moving_out, fixed_out, metadata_out)
    if len(set(targets)) != 3 or any(path.exists() for path in targets):
        raise ValueError("distinct, nonexistent output paths required")
    sources = (moving, fixed)
    images = []
    for path in sources:
        with Image.open(path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"unsupported nontrivial EXIF orientation: {path}")
            images.append(image.convert("RGB"))
    layouts = compute_layout([(im.height, im.width, 1., 1.) for im in images],
                             side=side)
    records = []
    for source, image, layout, output in zip(
        sources, images, layouts, (moving_out, fixed_out), strict=True,
    ):
        resized = image.resize(tuple(layout["resized_wh"]), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(resized, tuple(layout["padding_xy"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        records.append({"source": str(source), "original_wh": list(image.size),
                        "canvas_png": str(output), **layout})
    report = {
        "side": side, "moving": records[0], "fixed": records[1],
        "scale_assumption": "same pixel scale within BIRL scale-5pc pair; no scanner spacing",
        "landmarks_used": False,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("moving", "fixed", "moving_out", "fixed_out", "metadata_out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--side", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(make_canvas(args.moving, args.fixed, args.moving_out,
                                 args.fixed_out, args.metadata_out, side=args.side)))


if __name__ == "__main__":
    main()
