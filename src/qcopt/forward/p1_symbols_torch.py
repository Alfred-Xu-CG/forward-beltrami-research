"""Batched torch implementation of the periodic P1 two-face block symbol."""

from __future__ import annotations

import math

import torch


def p1_block_symbol_grid_torch(
    nx: int,
    ny: int,
    *,
    face_weights: tuple[float, float] = (1.0, 1.0),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batched ``(S, B@P)`` tensors over all Fourier modes.

    The formulas match :func:`qcopt.forward.p1_symbols.p1_block_symbol_grid`
    but evaluate all modes in one tensor batch, making the operation suitable
    for CUDA or accelerator benchmarking.  The constant derivative-null mode
    is explicitly masked to zero.
    """

    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least two")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be complex64 or complex128")
    weight_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    w = torch.as_tensor(face_weights, dtype=weight_dtype, device=device)
    if w.shape != (2,) or torch.any(w <= 0) or not torch.all(torch.isfinite(w)):
        raise ValueError("face_weights must be two positive finite values")
    kx = torch.arange(nx, dtype=torch.float64, device=device)[None, :]
    ky = torch.arange(ny, dtype=torch.float64, device=device)[:, None]
    phase_x = 2.0 * math.pi * kx / nx
    phase_y = 2.0 * math.pi * ky / ny
    X = torch.exp(1j * phase_x).to(dtype)
    Y = torch.exp(1j * phase_y).to(dtype)
    fx_plus = (X - 1.0) * nx
    fy_plus = X * (Y - 1.0) * ny
    fx_minus = Y * (X - 1.0) * nx
    fy_minus = (Y - 1.0) * ny
    fx = torch.stack((fx_plus.expand(ny, nx), fx_minus.expand(ny, nx)), dim=-1)
    fy = torch.stack((fy_plus.expand(ny, nx), fy_minus.expand(ny, nx)), dim=-1)
    a = 0.5 * (fx - 1j * fy)
    b = 0.5 * (fx + 1j * fy)
    denominator = torch.sum(torch.conj(b) * w * b, dim=-1)
    safe = torch.abs(denominator) > 1e-14
    p = torch.zeros((ny, nx, 1, 2), dtype=dtype, device=device)
    numerator = torch.conj(b) * w
    p[..., 0, :] = torch.where(safe[..., None], numerator / torch.where(safe, denominator, torch.ones_like(denominator))[..., None], torch.zeros_like(numerator))
    s = a[..., :, None] * p
    projector = b[..., :, None] * p
    return s.contiguous(), projector.contiguous()
