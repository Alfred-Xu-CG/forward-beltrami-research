"""Rank saved Q1 and DHR maps with the exact Q1 optimizer image objective.

This is a read-only same-pair diagnostic, not a baseline registration run.
All three candidate maps are evaluated on identical 512-square grayscale
thumbnails and pixel-center samples; the DHR field was itself fit under a
different preprocessing and is not a topology-safe Q1 output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.q1_image_sampling import (
    fixed_pixel_centers,
    q1_map_at_pixel_centers,
)
from tools.digital_q1_real_optimize import _image_correlation_loss, _read_gray_thumbnail


def dhr_map_at_fixed_pixel_centers(
    field: np.ndarray, params: dict, *,
    fixed_size: tuple[int, int], moving_size: tuple[int, int], image_side: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Return DHR's fixed->moving map in original JPEG normalized coordinates."""
    query = fixed_pixel_centers(
        image_side, image_side, dtype=torch.float32, device=torch.device(device),
    )
    return dhr_map_at_unit_queries(
        field, params, fixed_size=fixed_size, moving_size=moving_size,
        query=query,
    )


def dhr_map_at_unit_queries(
    field: np.ndarray, params: dict, *,
    fixed_size: tuple[int, int], moving_size: tuple[int, int], query: torch.Tensor,
) -> torch.Tensor:
    """Sample a saved DHR field at arbitrary fixed-frame unit-square queries."""
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] != 2 or min(field.shape[1:]) < 2:
        raise ValueError("saved field must have shape (2,H>=2,W>=2)")
    if not np.all(np.isfinite(field)):
        raise ValueError("saved field must be finite")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-1] != 2:
        raise ValueError("query must have shape (1,H,W,2)")
    device = query.device
    scale_f = float(params["target_resample_ratio"])
    scale_m = float(params["source_resample_ratio"])
    if min(scale_f, scale_m) <= 0:
        raise ValueError("resample ratios must be positive")
    fixed_pad = torch.tensor(
        [params["pad_2"][1][0], params["pad_2"][0][0]],
        device=device, dtype=torch.float32,
    )
    moving_pad = torch.tensor(
        [params["pad_1"][1][0], params["pad_1"][0][0]],
        device=device, dtype=torch.float32,
    )
    fixed_extent = torch.tensor(fixed_size, device=device, dtype=torch.float32)
    moving_extent = torch.tensor(moving_size, device=device, dtype=torch.float32)
    fixed_canvas = query * fixed_extent * scale_f - .5 + fixed_pad
    field_extent = torch.tensor(
        [field.shape[2] - 1, field.shape[1] - 1], device=device, dtype=torch.float32,
    )
    sampled = F.grid_sample(
        torch.as_tensor(field, device=device)[None],
        2 * fixed_canvas / field_extent - 1,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).permute(0, 2, 3, 1)
    moving_canvas = fixed_canvas + sampled
    return (moving_canvas - moving_pad + .5) / (moving_extent * scale_m)


def _appearance(fixed: torch.Tensor, moving: torch.Tensor,
                mapped_query: torch.Tensor) -> dict[str, float | int]:
    warped = F.grid_sample(
        moving, 2 * mapped_query - 1, mode="bilinear",
        padding_mode="border", align_corners=False,
    )
    foreground_fixed = fixed > .08
    foreground_warped = warped > .08
    intersection = (foreground_fixed & foreground_warped).sum()
    foreground_dice = 2 * intersection / (
        foreground_fixed.sum() + foreground_warped.sum()
    ).clamp_min(1)
    report = {
        "one_minus_ncc": float(_image_correlation_loss(fixed, warped)),
        "grayscale_mse": float((fixed - warped).square().mean()),
        "foreground_dice_threshold_0p08": float(foreground_dice),
        "out_of_unit_square_queries": int(((mapped_query < 0) | (mapped_query > 1)).any(-1).sum()),
    }
    for factor in (4, 8, 16):
        if min(fixed.shape[-2:]) >= factor * 2:
            reduced_size = tuple(side // factor for side in fixed.shape[-2:])
            reduced_fixed = F.interpolate(fixed, size=reduced_size, mode="area")
            reduced_warped = F.interpolate(warped, size=reduced_size, mode="area")
            report[f"one_minus_ncc_area_{reduced_size[-1]}"] = float(
                _image_correlation_loss(reduced_fixed, reduced_warped)
            )
    return report


def q1_query_from_saved_map(map_path: Path, *, image_side: int) -> torch.Tensor:
    """Query a stored residual Q1 map and then apply any saved affine factor."""
    with np.load(map_path) as archive:
        vertices = torch.from_numpy(archive["vertices"].astype(np.float32))
        affine = archive["post_affine_matrix"] if "post_affine_matrix" in archive else None
        offset = archive["post_affine_offset"] if "post_affine_offset" in archive else None
    if (affine is None) != (offset is None):
        raise ValueError("affine matrix and offset must be stored together")
    query = q1_map_at_pixel_centers(vertices, image_side, image_side)
    if affine is not None:
        query = query @ torch.from_numpy(affine.astype(np.float32)).T + torch.from_numpy(
            offset.astype(np.float32)
        )
    return query


def compare(
    dhr_field: Path, dhr_params: Path, q1_map: Path,
    moving_image: Path, fixed_image: Path, *, image_side: int,
) -> dict:
    import SimpleITK as sitk

    fixed, fixed_size = _read_gray_thumbnail(fixed_image, image_side)
    moving, moving_size = _read_gray_thumbnail(moving_image, image_side)
    query = fixed_pixel_centers(
        image_side, image_side, dtype=torch.float32, device=torch.device("cpu"),
    )
    q1_query = q1_query_from_saved_map(q1_map, image_side=image_side)
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(dhr_field)))
    params = json.loads(dhr_params.read_text())
    if float(params.get("initial_resample_ratio", 1)) != 1:
        raise ValueError("unaccounted DHR initial resample ratio")
    dhr_query = dhr_map_at_fixed_pixel_centers(
        field, params, fixed_size=fixed_size, moving_size=moving_size,
        image_side=image_side,
    )
    return {
        "image_side": image_side,
        "moving_original_size_xy": list(moving_size),
        "fixed_original_size_xy": list(fixed_size),
        "objective": "one_minus_ncc on independently resampled inverted-grayscale thumbnails",
        "identity_normalized": _appearance(fixed, moving, query),
        "optimized_safe_q1": _appearance(fixed, moving, q1_query),
        "dhr_saved_field": _appearance(fixed, moving, dhr_query),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dhr_field", "dhr_params", "q1_map", "moving_image", "fixed_image"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.dhr_field, args.dhr_params, args.q1_map,
                     args.moving_image, args.fixed_image, image_side=args.image_side)
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
