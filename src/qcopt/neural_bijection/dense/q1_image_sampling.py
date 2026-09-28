"""Evaluate a Q1 geometry map at physical image pixel centers.

Map vertices are values at source-grid nodes ``j/(columns-1),i/(rows-1)``.
Image samples instead occupy pixel centers ``(j+1/2)/width,(i+1/2)/height``.
The first interpolation evaluates the Q1 map; the second samples image
intensity. No image-grid resampling is used as a topology certificate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fixed_pixel_centers(
    height: int, width: int, *, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Return normalized fixed-image pixel centers with shape ``(1,H,W,2)``."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    x = (torch.arange(width, device=device, dtype=dtype) + .5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + .5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def q1_map_at_pixel_centers(
    vertices: torch.Tensor, height: int, width: int,
) -> torch.Tensor:
    """Evaluate the continuous Q1 map at fixed-image pixel centers."""
    if vertices.ndim != 4 or vertices.shape[-1] != 2 or vertices.shape[0] < 1 or (
        min(vertices.shape[1:3]) < 2
    ):
        raise ValueError("vertices must have shape (B,R>=2,C>=2,2)")
    centers = fixed_pixel_centers(height, width, dtype=vertices.dtype, device=vertices.device)
    return F.grid_sample(
        vertices.permute(0, 3, 1, 2), 2 * centers.expand(vertices.shape[0], -1, -1, -1) - 1,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).permute(0, 2, 3, 1)


def warp_moving_at_q1_map(
    moving: torch.Tensor, vertices: torch.Tensor, *, height: int, width: int,
) -> torch.Tensor:
    """Return ``I_M(Phi(q))`` at the requested fixed-image pixel centers."""
    if moving.ndim != 4 or moving.shape[0] != vertices.shape[0] or (
        moving.device != vertices.device or moving.dtype != vertices.dtype
    ):
        raise ValueError("moving image and map need matching batch, dtype and device")
    query_map = q1_map_at_pixel_centers(vertices, height, width)
    return F.grid_sample(
        moving, 2 * query_map - 1,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
