"""Image-conditioned local displacement hint for a safe radial P1 layer.

This computes an optical-flow-style latent proposal, not a deformation on
its own. The colored radial safety layer must still enforce topology.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def physical_image_gradient(image: torch.Tensor) -> torch.Tensor:
    """Central differences in unit-square coordinates, endpoint replicated."""
    if image.ndim != 4 or image.shape[1] != 1 or image.shape[-2] != image.shape[-1]:
        raise ValueError("image must be a square single-channel BCHW tensor")
    side = image.shape[-1]
    gx = 0.5 * (side - 1) * (image[..., 2:] - image[..., :-2])
    gy = 0.5 * (side - 1) * (image[..., 2:, :] - image[..., :-2, :])
    gx = F.pad(gx, (1, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 1, 1), mode="replicate")
    return torch.cat((gx, gy), dim=1)


def bilinear_image_value_and_gradient(
    image: torch.Tensor, coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a one-channel image and its exact piecewise-bilinear gradient.

    Coordinates are BHWC2 in unit-square physical units. The derivative is
    with respect to these units, not pixel indices. At grid-cell boundaries
    the bilinear interpolant has only one-sided derivatives; this function
    chooses the cell to the right/bottom except at the final image boundary.
    Border padding is constant outside the unit square.
    """
    if image.ndim != 4 or image.shape[1] != 1 or coordinates.ndim != 4 or (
        coordinates.shape[-1] != 2 or coordinates.shape[0] != image.shape[0]
    ):
        raise ValueError("image must be B1HW and coordinates BHWC2 with matching batch")
    batch, _, height, width = image.shape
    if min(height, width) < 2:
        raise ValueError("image height and width must be at least two")
    x = coordinates[..., 0].clamp(0, 1) * (width - 1)
    y = coordinates[..., 1].clamp(0, 1) * (height - 1)
    ix0 = torch.floor(x).long().clamp(0, width - 2)
    iy0 = torch.floor(y).long().clamp(0, height - 2)
    tx = x - ix0.to(x.dtype)
    ty = y - iy0.to(y.dtype)
    flat = image[:, 0].reshape(batch, height * width)

    def gather(row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        return torch.gather(flat, 1, (row * width + column).flatten(1)).reshape_as(x)

    v00 = gather(iy0, ix0)
    v10 = gather(iy0, ix0 + 1)
    v01 = gather(iy0 + 1, ix0)
    v11 = gather(iy0 + 1, ix0 + 1)
    upper = (1 - tx) * v00 + tx * v10
    lower = (1 - tx) * v01 + tx * v11
    value = (1 - ty) * upper + ty * lower
    gx = (width - 1) * ((1 - ty) * (v10 - v00) + ty * (v11 - v01))
    gy = (height - 1) * (lower - upper)
    gx = gx * ((coordinates[..., 0] >= 0) & (coordinates[..., 0] <= 1))
    gy = gy * ((coordinates[..., 1] >= 0) & (coordinates[..., 1] <= 1))
    return value[:, None], torch.stack((gx, gy), dim=1)


def local_photometric_logits(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    base: torch.Tensor,
    *,
    window: int,
    ridge: float,
    raw_span: float,
    gradient_mode: str = "central",
) -> torch.Tensor:
    """Local normal-equation flow on the current map, expressed as finite logits.

    The output shape is (B,N-2,N-2,2). It may be combined with learned
    logits before the radial safety layer; by itself it offers no topology
    theorem. The 2x2 normal matrix receives a positive ridge, so the solve
    is nonsingular for finite image values.
    """
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
        raise ValueError("fixed/moving must have matching BCHW single-channel shape")
    if base.ndim != 4 or base.shape[0] != fixed.shape[0] or base.shape[-1] != 2 or base.shape[1] != base.shape[2]:
        raise ValueError("base must have shape (B,N,N,2)")
    if window < 1 or window % 2 != 1 or ridge <= 0 or raw_span <= 0:
        raise ValueError("window must be odd positive; ridge and raw_span positive")
    if gradient_mode not in ("central", "bilinear_exact"):
        raise ValueError("gradient_mode must be central or bilinear_exact")
    side = base.shape[1]
    sample_grid = 2 * base - 1
    fixed_at_control = F.interpolate(fixed, size=(side, side), mode="bilinear", align_corners=True)
    if gradient_mode == "central":
        moved_at_control = F.grid_sample(
            moving, sample_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        gradient_at_control = F.grid_sample(
            physical_image_gradient(moving), sample_grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
    else:
        moved_at_control, gradient_at_control = bilinear_image_value_and_gradient(
            moving, base,
        )
    gx = gradient_at_control[:, 0:1]
    gy = gradient_at_control[:, 1:2]
    residual = fixed_at_control - moved_at_control

    def local_mean(values: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(values, window, stride=1, padding=window // 2, count_include_pad=False)

    gxx = local_mean(gx * gx) + ridge
    gxy = local_mean(gx * gy)
    gyy = local_mean(gy * gy) + ridge
    bx = local_mean(gx * residual)
    by = local_mean(gy * residual)
    determinant = gxx * gyy - gxy * gxy
    dx = (gyy * bx - gxy * by) / determinant
    dy = (gxx * by - gxy * bx) / determinant
    displacement = torch.cat((dx, dy), dim=1)
    scale = raw_span / (side - 1)
    logits = torch.atanh((displacement / scale).clamp(-0.95, 0.95))
    return logits[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
