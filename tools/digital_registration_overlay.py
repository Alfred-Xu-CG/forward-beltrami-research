"""Render fixed/warped-moving overlays for qualitative inspection only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from tools.digital_birl_landmark_score import p1_at_queries


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32)


def overlay(fixed: np.ndarray, moving: np.ndarray, vertices: np.ndarray,
            matrix: np.ndarray, offset: np.ndarray) -> Image.Image:
    axis = (np.arange(512, dtype=np.float64) + .5) / 512
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    query = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    mapped = p1_at_queries(vertices, query)
    moving_unit = mapped @ matrix.T + offset
    moving_ij = 512 * moving_unit - .5
    warped = map_coordinates(
        moving, [moving_ij[:, 1], moving_ij[:, 0]], order=1,
        mode="constant", cval=255.).reshape(512, 512)
    rgb = np.stack((fixed, warped, .5 * (fixed + warped)), axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--map", action="append", nargs=2,
                        metavar=("LABEL", "NPZ"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixed, moving = load_gray(args.fixed), load_gray(args.moving)
    axis = np.linspace(0, 1, 257, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    identity = np.stack((xx, yy), axis=-1)
    panels = []
    for label, path_text in args.map:
        with np.load(path_text) as data:
            matrix = data["post_affine_matrix"].astype(np.float64)
            offset = data["post_affine_offset"].astype(np.float64)
            vertices = (identity if label == "affine" else
                        data["vertices"][0].astype(np.float64))
        panel = overlay(fixed, moving, vertices, matrix, offset)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 240, 25), fill=(0, 0, 0))
        draw.text((8, 5), label, fill=(255, 255, 255))
        panels.append(panel)
    width = 512 * len(panels)
    result = Image.new("RGB", (width, 512))
    for index, panel in enumerate(panels):
        result.paste(panel, (512 * index, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"saved {args.output} {width}x512")


if __name__ == "__main__":
    main()
