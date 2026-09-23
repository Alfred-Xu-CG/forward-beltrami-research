"""Differentiable Dirichlet-zero spectral processing on rectangular grids.

The transform uses an odd extension before FFT. It does not impose periodic
boundary values on the original displacement field.
"""

from __future__ import annotations

import torch


def dst1(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Unnormalized discrete sine transform type I along one dimension."""
    dimension = dim % values.ndim
    size = values.shape[dimension]
    if size < 1:
        raise ValueError("transform dimension must be nonempty")
    zero = torch.zeros_like(values.narrow(dimension, 0, 1))
    odd_extension = torch.cat((zero, values, zero, -values.flip(dimension)), dim=dimension)
    frequency = torch.fft.fft(odd_extension, dim=dimension)
    return -0.5 * frequency.imag.narrow(dimension, 1, size)


def dst2(values: torch.Tensor) -> torch.Tensor:
    """Two-dimensional DST-I on the last two axes."""
    return dst1(dst1(values, -1), -2)


def idst2(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse of dst2 with the same differentiable FFT primitives."""
    height, width = coefficients.shape[-2:]
    return (4.0 / ((height + 1) * (width + 1))) * dst2(coefficients)


def keep_vector_sine_modes(displacement: torch.Tensor, count: int) -> torch.Tensor:
    """Keep the top-count joint x/y sine frequencies of an interior field.

    Input/output have shape (batch, interior_rows, interior_columns, 2).
    The hard top-k mask is differentiable with respect to retained values
    away from coefficient-order ties. Reinsert zero boundary values when
    using this field as a full-grid displacement.
    """
    if displacement.ndim != 4 or displacement.shape[-1] != 2:
        raise ValueError("displacement must have shape (batch,height,width,2)")
    batch, height, width, _ = displacement.shape
    if count < 1 or count > height * width:
        raise ValueError("count must lie between one and the number of sine modes")
    coefficients = dst2(displacement.permute(0, 3, 1, 2))
    strength = coefficients.square().sum(dim=1).reshape(batch, -1)
    indices = strength.topk(count, dim=-1).indices
    mask = torch.zeros_like(strength).scatter_(1, indices, 1.0).reshape(batch, 1, height, width)
    filtered = idst2(coefficients * mask)
    return filtered.permute(0, 2, 3, 1)


def spectralize_bounded_logits(
    logits: torch.Tensor, *, side: int, raw_span: float, count: int,
) -> torch.Tensor:
    """Filter the raw local proposal and encode it back as finite radial logits."""
    if logits.shape[1:3] != (side - 2, side - 2):
        raise ValueError("logit grid does not match side")
    scale = raw_span / (side - 1)
    displacement = scale * logits.tanh()
    filtered = keep_vector_sine_modes(displacement, count)
    return torch.atanh((filtered / scale).clamp(-0.95, 0.95))
