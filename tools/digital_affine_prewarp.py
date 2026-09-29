"""Pure-Torch fixed-to-moving affine image sampling; no feature-matcher dependency."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers


def warp_moving_to_fixed(
    moving: torch.Tensor, matrix: torch.Tensor, offset: torch.Tensor, *,
    height: int, width: int,
) -> torch.Tensor:
    """Sample J(q)=I_M(A(q)) at normalized fixed-image pixel centers."""
    if moving.ndim != 4 or moving.shape[1] != 1 or moving.shape[0] != 1:
        raise ValueError("expected one grayscale moving image")
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not bool(torch.isfinite(matrix).all() and torch.isfinite(offset).all())
    ):
        raise ValueError("finite 2x2 matrix and 2-vector offset required")
    q = fixed_pixel_centers(height, width, dtype=moving.dtype, device=moving.device)
    mapped = q @ matrix.to(moving).T + offset.to(moving)
    return F.grid_sample(moving, 2 * mapped - 1, mode="bilinear",
                         padding_mode="border", align_corners=False)
