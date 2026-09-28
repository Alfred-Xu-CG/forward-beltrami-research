"""Image-only moving-to-fixed affine baselines for a real pathology pair.

The registration entry point deliberately has no landmark file arguments or reads.
Coordinates are continuous image pixel (x, y), origin at the top left, with y
increasing downward. The stored affine acts on column vectors [x, y, 1].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_identity(moving_image: str | Path, fixed_image: str | Path) -> dict:
    moving_path = Path(moving_image).resolve(strict=True)
    fixed_path = Path(fixed_image).resolve(strict=True)
    with Image.open(moving_path) as image:
        moving_size = list(image.size)
    with Image.open(fixed_path) as image:
        fixed_size = list(image.size)
    return {
        "method": "identity_pixel_xy",
        "direction": "moving_to_fixed",
        "coordinate_frame": "top_left_origin_pixel_xy",
        "moving_image": str(moving_path),
        "fixed_image": str(fixed_path),
        "moving_size_xy": moving_size,
        "fixed_size_xy": fixed_size,
        "matrix_moving_to_fixed_xy": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "det_linear": 1.0,
    }


def _tissue_centroid_xy(image_path: str, original_size_xy: list[int]) -> tuple[np.ndarray, float]:
    with Image.open(image_path) as image:
        image.draft("L", (384, 384))
        image.thumbnail((384, 384), Image.Resampling.BILINEAR)
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    tissue = gray < 230
    fraction = float(tissue.mean())
    if not 0.01 <= fraction <= 0.95:
        raise ValueError(f"tissue mask fraction {fraction:.3f} is outside usable range")
    y_index, x_index = np.nonzero(tissue)
    width, height = original_size_xy
    thumbnail_height, thumbnail_width = gray.shape
    xy = np.array(
        [
            (x_index.mean() + 0.5) * width / thumbnail_width - 0.5,
            (y_index.mean() + 0.5) * height / thumbnail_height - 0.5,
        ],
        dtype=np.float64,
    )
    return xy, fraction


def make_tissue_centroid_translation(moving_image: str | Path, fixed_image: str | Path) -> dict:
    """Align thresholded tissue centroids using only the two image thumbnails."""
    transform = make_identity(moving_image, fixed_image)
    moving_xy, moving_fraction = _tissue_centroid_xy(transform["moving_image"], transform["moving_size_xy"])
    fixed_xy, fixed_fraction = _tissue_centroid_xy(transform["fixed_image"], transform["fixed_size_xy"])
    dx, dy = (fixed_xy - moving_xy).tolist()
    transform["method"] = "tissue_centroid_translation"
    transform["matrix_moving_to_fixed_xy"] = [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]
    transform["image_feature"] = {
        "thumbnail_max_xy": [384, 384],
        "grayscale_tissue_threshold_strictly_below": 230,
        "moving_tissue_fraction": moving_fraction,
        "fixed_tissue_fraction": fixed_fraction,
    }
    return transform


def save_transform(transform: dict, output: str | Path) -> None:
    Path(output).write_text(json.dumps(transform, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-image", required=True)
    parser.add_argument("--fixed-image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("identity", "tissue-centroid-translation"), default="identity")
    args = parser.parse_args()
    if args.method == "identity":
        transform = make_identity(args.moving_image, args.fixed_image)
    else:
        transform = make_tissue_centroid_translation(args.moving_image, args.fixed_image)
    save_transform(transform, args.output)
    print(json.dumps(transform))


if __name__ == "__main__":
    main()
