"""First-order-differentiable descriptor-force cue at a fixed safe P1 map.

This approximates the spatial derivative of the MIND-like image objective by
central differences of the moving descriptor. It does not call autograd.grad,
so its own VJP requires no unsupported grid_sample second derivative.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tools.digital_mind_objective_probe import sampled_p1_descriptor


def image_force_features(fixed_descriptor: torch.Tensor,
                         moving_descriptor: torch.Tensor,
                         mask: torch.Tensor,
                         current: torch.Tensor) -> torch.Tensor:
    if (fixed_descriptor.ndim != 4 or
            fixed_descriptor.shape != moving_descriptor.shape or
            fixed_descriptor.shape[1:] != (8, 256, 256) or
            mask.shape != fixed_descriptor.shape[:1] + (1, 256, 256) or
            current.shape != (fixed_descriptor.shape[0], 257, 257, 2)):
        raise ValueError("eight-channel 256 descriptors, mask and 257 map required")
    horizontal = F.pad(moving_descriptor, (1, 1, 0, 0), mode="replicate")
    vertical = F.pad(moving_descriptor, (0, 0, 1, 1), mode="replicate")
    dx = 128 * (horizontal[..., 2:] - horizontal[..., :-2])
    dy = 128 * (vertical[..., 2:, :] - vertical[..., :-2, :])
    warped = sampled_p1_descriptor(moving_descriptor, current)
    warped_dx = sampled_p1_descriptor(dx, current)
    warped_dy = sampled_p1_descriptor(dy, current)
    residual = fixed_descriptor - warped
    smooth_sign = residual / torch.sqrt(residual.square() + .05 ** 2)
    force = torch.cat((
        -(smooth_sign * warped_dx).mean(dim=1, keepdim=True),
        -(smooth_sign * warped_dy).mean(dim=1, keepdim=True),
    ), dim=1) * mask
    rms = force.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    normalized = torch.tanh(force / (3 * rms + 1e-5))
    broad = F.avg_pool2d(normalized, 7, stride=1, padding=3)
    mismatch = residual.abs().mean(dim=1, keepdim=True) * mask
    return torch.cat((normalized, broad, mismatch, mask), dim=1)
