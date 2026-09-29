"""Read an official ACROBAT TIFF pyramid at a small level into a D: preview.

The TIFF's base image is hundreds of megapixels. This utility reads only one
small pyramid level, resizes that level to the image encoder's square input,
and records the original and selected dimensions. It does not align slides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import tifffile


def choose_level(shapes: list[tuple[int, int]], side: int) -> int:
    if not shapes or side < 1 or any(min(shape) < 1 for shape in shapes):
        raise ValueError("positive pyramid shapes and side required")
    candidates = [index for index, shape in enumerate(shapes)
                  if min(shape) >= side]
    return candidates[-1] if candidates else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=512)
    args = parser.parse_args()
    if args.side < 16:
        raise ValueError("side must be >=16")
    with tifffile.TiffFile(args.input) as tif:
        levels = tif.series[0].levels
        shapes = [tuple(level.shape[:2]) for level in levels]
        index = choose_level(shapes, args.side)
        data = levels[index].asarray()
    image = Image.fromarray(data).convert("RGB")
    image = image.resize((args.side, args.side), Image.Resampling.BILINEAR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "source": str(args.input),
        "thumbnail": str(args.output),
        "original_hw": shapes[0],
        "pyramid_level": index,
        "read_hw": shapes[index],
        "thumbnail_hw": [args.side, args.side],
        "warning": "independent square resize, not physical alignment",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
