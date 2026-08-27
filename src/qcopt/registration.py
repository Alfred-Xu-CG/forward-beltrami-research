"""Differentiable analytic glyph fields for large-deformation registration."""

from __future__ import annotations

import math

import torch


def _rectangle_sdf(
    points: torch.Tensor, center: tuple[float, float], half_size: tuple[float, float]
) -> torch.Tensor:
    center_tensor = points.new_tensor(center)
    half_tensor = points.new_tensor(half_size)
    offset = torch.abs(points - center_tensor) - half_tensor
    outside = torch.linalg.vector_norm(torch.clamp(offset, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(offset, dim=-1).values, max=0.0)
    return outside + inside


def i_field(
    points: torch.Tensor, edge_softness: float = 0.012, union_softness: float = 0.008
) -> torch.Tensor:
    """Soft occupancy of a thick serif capital I on the unit square."""

    distances = torch.stack(
        (
            _rectangle_sdf(points, (0.5, 0.5), (0.065, 0.29)),
            _rectangle_sdf(points, (0.5, 0.80), (0.23, 0.055)),
            _rectangle_sdf(points, (0.5, 0.20), (0.23, 0.055)),
        ),
        dim=-1,
    )
    smooth_union = -union_softness * torch.logsumexp(
        -distances / union_softness, dim=-1
    )
    return torch.sigmoid(-smooth_union / edge_softness)


def s_field(
    points: torch.Tensor,
    thickness: float = 0.065,
    edge_softness: float = 0.012,
    minimum_softness: float = 0.010,
) -> torch.Tensor:
    """Soft occupancy around a differentiable polyline S centerline."""

    path = points.new_tensor(
        [
            [0.72, 0.79],
            [0.58, 0.87],
            [0.38, 0.86],
            [0.27, 0.75],
            [0.32, 0.61],
            [0.55, 0.52],
            [0.72, 0.41],
            [0.70, 0.27],
            [0.58, 0.16],
            [0.36, 0.15],
            [0.25, 0.23],
        ]
    )
    starts = path[:-1]
    directions = path[1:] - starts
    offsets = points[..., None, :] - starts
    parameters = (offsets * directions).sum(dim=-1) / directions.square().sum(dim=-1)
    parameters = torch.clamp(parameters, 0.0, 1.0)
    closest = starts + parameters[..., None] * directions
    distances = torch.linalg.vector_norm(points[..., None, :] - closest, dim=-1)
    soft_distance = -minimum_softness * (
        torch.logsumexp(-distances / minimum_softness, dim=-1)
        - math.log(distances.shape[-1])
    )
    return torch.sigmoid((thickness - soft_distance) / edge_softness)


def registration_loss(
    source_values: torch.Tensor,
    warped_target_values: torch.Tensor,
    foreground_weight: float = 4.0,
) -> torch.Tensor:
    if source_values.shape != warped_target_values.shape:
        raise ValueError("source and warped target fields must have matching shapes")
    weights = 1.0 + foreground_weight * source_values.detach()
    return (weights * (source_values - warped_target_values).square()).mean()


def soft_dice(
    first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    numerator = 2.0 * torch.sum(first * second) + epsilon
    denominator = torch.sum(first.square()) + torch.sum(second.square()) + epsilon
    return numerator / denominator
