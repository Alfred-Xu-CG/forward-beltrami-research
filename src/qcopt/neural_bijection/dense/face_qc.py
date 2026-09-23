"""Original-triangle quasiconformal distortion for structured P1 maps."""

from __future__ import annotations

import torch


def structured_p1_face_beltrami_modulus(control: torch.Tensor) -> torch.Tensor:
    """Return |mu| on both original triangles of every structured cell.

    Shape: (batch, 2, side-1, side-1). Triangle zero is (a,b,c), triangle
    one is (a,c,d), with a southwest, b southeast, c northeast, d northwest.
    Coordinates use the unit-square source grid.
    """
    if control.ndim != 4 or control.shape[1] != control.shape[2] or control.shape[-1] != 2:
        raise ValueError("control must have shape (batch,side,side,2)")
    side = control.shape[1]
    a = control[:, :-1, :-1]
    b = control[:, :-1, 1:]
    c = control[:, 1:, 1:]
    d = control[:, 1:, :-1]
    lower_dx = (side - 1) * (b - a)
    lower_dy = (side - 1) * (c - b)
    upper_dx = (side - 1) * (c - d)
    upper_dy = (side - 1) * (d - a)

    def modulus(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        fz_real = 0.5 * (dx[..., 0] + dy[..., 1])
        fz_imag = 0.5 * (dx[..., 1] - dy[..., 0])
        fzbar_real = 0.5 * (dx[..., 0] - dy[..., 1])
        fzbar_imag = 0.5 * (dx[..., 1] + dy[..., 0])
        numerator = fzbar_real.square() + fzbar_imag.square()
        denominator = fz_real.square() + fz_imag.square()
        tiny = torch.finfo(control.dtype).tiny
        return torch.sqrt(numerator.clamp_min(tiny) / denominator.clamp_min(tiny))

    return torch.stack((modulus(lower_dx, lower_dy), modulus(upper_dx, upper_dy)), dim=1)


def structured_p1_qc_tail_penalty(control: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mean squared excess |mu| above a declared threshold."""
    if not 0 <= threshold < 1:
        raise ValueError("threshold must lie in [0,1)")
    modulus = structured_p1_face_beltrami_modulus(control)
    return (modulus - threshold).clamp_min(0).square().mean()
