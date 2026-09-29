"""Conjugate a native-normalized histology affine into a padded canvas frame.

ImageJ landmark files are not used. The input map must be an independently
normalized fixed-JPEG to moving-JPEG positive affine.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from tools.digital_birl_pair_canvas import make_canvas


def native_unit_to_canvas_affine(layout: dict, side: int) -> tuple[np.ndarray, np.ndarray]:
    resized = np.asarray(layout["resized_wh"], dtype=np.float64)
    padding = np.asarray(layout["padding_xy"], dtype=np.float64)
    if resized.shape != (2,) or padding.shape != (2,) or np.any(resized <= 0):
        raise ValueError("positive two-axis layout required")
    return np.diag(resized / side), padding / side


def convert(matrix: np.ndarray, offset: np.ndarray,
            layout: dict) -> tuple[np.ndarray, np.ndarray]:
    """A_canvas = T_moving o A_native o T_fixed^{-1}."""
    side = int(layout["side"])
    fixed_scale, fixed_pad = native_unit_to_canvas_affine(layout["fixed"], side)
    moving_scale, moving_pad = native_unit_to_canvas_affine(layout["moving"], side)
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not np.isfinite(matrix).all() or not np.isfinite(offset).all()
    ):
        raise ValueError("finite native affine required")
    canvas_matrix = moving_scale @ matrix @ np.linalg.inv(fixed_scale)
    canvas_offset = moving_scale @ offset + moving_pad - canvas_matrix @ fixed_pad
    return canvas_matrix, canvas_offset


def run(*, fixed: Path, moving: Path, native_affine: Path,
        fixed_out: Path, moving_out: Path, layout_out: Path,
        affine_out: Path) -> dict:
    if affine_out.exists():
        raise FileExistsError(affine_out)
    layout = make_canvas(moving, fixed, moving_out, fixed_out, layout_out, side=512)
    layout["scale_assumption"] = (
        "HistoReg full-resolution CD4/CD68 native JPEGs at assumed common "
        "pixel scale; not BIRL scale-5pc")
    layout_out.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    with np.load(native_affine) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float64)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float64)
    cm, cb = convert(matrix, offset, layout)
    exact = [Fraction.from_float(float(x)) for x in cm.flat]
    if exact[0] * exact[3] - exact[1] * exact[2] <= 0:
        raise ValueError("converted canvas affine must preserve orientation")
    np.savez_compressed(affine_out, post_affine_matrix=cm.astype(np.float32),
                        post_affine_offset=cb.astype(np.float32))
    report = {"fixed": str(fixed), "moving": str(moving),
              "native_affine": str(native_affine), "canvas_affine": str(affine_out),
              "native_matrix": matrix.tolist(), "native_offset": offset.tolist(),
              "canvas_matrix_float64": cm.tolist(), "canvas_offset_float64": cb.tolist(),
              "canvas_determinant_float64": float(np.linalg.det(cm)),
              "landmarks_used": False}
    affine_out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "native_affine", "fixed_out", "moving_out",
                 "layout_out", "affine_out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args))))


if __name__ == "__main__":
    main()
