"""Device-native blocked whole-plane Beurling quadrature for scattered points."""

from __future__ import annotations

import math

import torch


def direct_beurling_apply_torch(
    points: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_points: torch.Tensor | None = None,
    block_size: int = 4096,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply ``-1/pi sum_j values_j weights_j/(z-w_j)^2`` on any device.

    The source and target sets are arbitrary scattered complex points; no
    periodic wrapping or uniform-grid assumption is made.  Target blocks are
    materialized one at a time, so GPU memory scales as
    ``O(block_size * n_source)``.  If ``target_points`` is omitted, matching
    source/target diagonal terms are masked as a principal-value diagnostic.
    The operation is differentiable with respect to ``values`` and ``weights``
    (and with respect to points away from the masked self terms).
    """

    source_input = torch.as_tensor(points)
    value_input = torch.as_tensor(values)
    weight_input = torch.as_tensor(weights)
    if source_input.ndim != 1 or value_input.ndim != 1 or weight_input.ndim != 1:
        raise ValueError("points, values, and weights must be one-dimensional")
    if source_input.shape != value_input.shape or source_input.shape != weight_input.shape:
        raise ValueError("points, values, and weights must have matching shapes")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if dtype is None:
        dtype = torch.complex64 if value_input.dtype == torch.complex64 else torch.complex128
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be complex64 or complex128")
    source = source_input.to(dtype=dtype)
    density = value_input.to(dtype=dtype)
    area_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    area = weight_input.to(dtype=area_dtype)
    if target_points is None:
        targets = source
    else:
        target_input = torch.as_tensor(target_points)
        if target_input.ndim != 1:
            raise ValueError("target_points must be one-dimensional")
        targets = target_input.to(device=source.device, dtype=dtype)
    if not bool(torch.all(torch.isfinite(source))) or not bool(torch.all(torch.isfinite(density))):
        raise ValueError("points and values must be finite")
    if not bool(torch.all(torch.isfinite(targets))) or not bool(torch.all(torch.isfinite(area))):
        raise ValueError("target_points and weights must be finite")
    if bool(torch.any(area <= 0.0)):
        raise ValueError("weights must be positive")

    weighted = density * area.to(dtype=dtype)
    output_blocks = []
    same_source = target_points is None
    for start in range(0, int(targets.numel()), block_size):
        stop = min(start + block_size, int(targets.numel()))
        delta = targets[start:stop, None] - source[None, :]
        if same_source:
            delta = delta.clone()
            rows = torch.arange(stop - start, device=source.device)
            columns = rows + start
            delta[rows, columns] = complex(float("inf"), 0.0)
        nonzero = torch.abs(delta) > 0.0
        safe_delta = torch.where(nonzero, delta, torch.ones_like(delta))
        kernel = torch.where(
            nonzero,
            -1.0 / (math.pi * safe_delta * safe_delta),
            torch.zeros_like(safe_delta),
        )
        output_blocks.append(kernel @ weighted)
    if not output_blocks:
        return torch.zeros((0,), dtype=dtype, device=source.device)
    return torch.cat(output_blocks, dim=0).contiguous()
