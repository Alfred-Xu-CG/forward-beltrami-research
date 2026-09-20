"""Differentiable positive-increment hard decoder for latent experiments."""

from __future__ import annotations

import torch


def positive_increment_knots(logits: torch.Tensor) -> torch.Tensor:
    """Decode logits to normalized monotone knot positions in ``[0,1]``."""

    if logits.ndim != 1 or logits.numel() < 1 or not torch.is_floating_point(logits):
        raise ValueError("logits must be a floating one-dimensional tensor")
    increments = torch.nn.functional.softplus(logits) + torch.finfo(logits.dtype).eps
    cumulative = torch.cat((torch.zeros(1, dtype=logits.dtype, device=logits.device), torch.cumsum(increments, dim=0)))
    return cumulative / cumulative[-1]
