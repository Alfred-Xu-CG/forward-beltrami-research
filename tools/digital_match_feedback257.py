"""One differentiable, topology-safe, current-map machine-match feedback pass.

The Gaussian field is only a desired displacement. The existing current-edge
F1 operator supplies the actual per-vertex area safety and fixed boundary.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qcopt.neural_bijection.dense.digital_q1 import AdaptiveSoftRadialQ1Relaxation
from tools.digital_mind_sparse_match_finetune import p1_at_points


class AnalyticMatchFeedback257(nn.Module):
    def __init__(self, *, sigma: float = .04, tau: float = .05,
                 gain: float = 1.) -> None:
        super().__init__()
        if not (math.isfinite(sigma) and sigma > 0 and math.isfinite(tau)
                and tau > 0 and math.isfinite(gain) and gain > 0):
            raise ValueError("positive finite sigma, tau and gain required")
        self.sigma, self.tau, self.gain = sigma, tau, gain
        self.safe_update = AdaptiveSoftRadialQ1Relaxation(
            257, raw_span=8., safety_fraction=.75, minimum_jacobian=.05)

    def forward(self, current: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if current.shape != (1, 257, 257, 2) or source.ndim != 2 or (
            source.shape != target.shape or source.shape[1] != 2 or
            len(source) < 1 or current.device != source.device or
            source.device != target.device or current.dtype != source.dtype or
            source.dtype != target.dtype or not bool(torch.isfinite(current).all()) or
            not bool(torch.isfinite(source).all()) or
            not bool(torch.isfinite(target).all())
        ):
            raise ValueError("finite same-device 257 map and paired points required")
        residual = target - p1_at_points(current, source)
        axis = torch.arange(1, 256, device=current.device,
                            dtype=current.dtype) / 256.
        xw = torch.exp(-((axis[:, None] - source[None, :, 0]).square()) /
                       (2 * self.sigma * self.sigma))
        yw = torch.exp(-((axis[:, None] - source[None, :, 1]).square()) /
                       (2 * self.sigma * self.sigma))
        mass = yw @ xw.T
        desired = self.gain * torch.stack((
            (yw * residual[None, :, 0]) @ xw.T,
            (yw * residual[None, :, 1]) @ xw.T,
        ), dim=-1) / (mass[..., None] + self.tau)
        horizontal = (current[:, 1:-1, 2:] - current[:, 1:-1, :-2]) / 2
        vertical = (current[:, 2:, 1:-1] - current[:, :-2, 1:-1]) / 2
        horizontal, vertical = horizontal[0], vertical[0]
        determinant = (horizontal[..., 0] * vertical[..., 1] -
                       horizontal[..., 1] * vertical[..., 0])
        valid = determinant.abs() > 1e-10
        denominator = torch.where(valid, 8 * determinant,
                                  torch.ones_like(determinant))
        u = (desired[..., 0] * vertical[..., 1] -
             desired[..., 1] * vertical[..., 0]) / denominator
        v = (horizontal[..., 0] * desired[..., 1] -
             horizontal[..., 1] * desired[..., 0]) / denominator
        logits = torch.atanh(torch.stack((u, v), dim=-1).clamp(-.98, .98))
        logits = torch.where(valid[..., None], logits, torch.zeros_like(logits))
        return self.safe_update(current, logits[None])
