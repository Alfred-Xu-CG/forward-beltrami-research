"""O(N²) forward synthesis of bounded positive edge weights from a desired map.

One pass cancels the linearized weighted-equilibrium residual exactly before
weight clipping; repeated passes are heuristic fixed-point corrections.
"""

from __future__ import annotations

import torch

from .sine_pcg_tutte import _laplacian


def _integrated_edge_correction(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose centered row/column flux increments for the two coordinates."""
    batch, side, _, _ = residual.shape
    spacing_inverse = side - 1
    interior_x = spacing_inverse * residual[:, 1:-1, 1:-1, 0]
    interior_y = spacing_inverse * residual[:, 1:-1, 1:-1, 1]
    horizontal_rows = torch.cat((torch.zeros_like(interior_x[:, :, :1]), interior_x.cumsum(dim=2)), dim=2)
    horizontal_rows = horizontal_rows - horizontal_rows.mean(dim=2, keepdim=True)
    vertical_columns = torch.cat((torch.zeros_like(interior_y[:, :1]), interior_y.cumsum(dim=1)), dim=1)
    vertical_columns = vertical_columns - vertical_columns.mean(dim=1, keepdim=True)
    horizontal = torch.nn.functional.pad(horizontal_rows, (0, 0, 1, 1))
    vertical = torch.nn.functional.pad(vertical_columns, (1, 1, 0, 0))
    if horizontal.shape != (batch, side, side - 1) or vertical.shape != (batch, side - 1, side):
        raise AssertionError("integrated edge increments have incorrect grid shape")
    return horizontal, vertical


def synthesize_bounded_conductances(
    desired: torch.Tensor,
    *,
    passes: int = 1,
    minimum: float = 1.0,
    maximum: float = 16.0,
    base: float = 4.0,
    gauge: str = "centered",
    return_stats: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, object] | None]:
    """Return horizontal, vertical, diagonal logits for SinePreconditionedTutteLayer.

    The input has shape (B,N,N,2), with source coordinates in [0,1]^2 and
    identity boundary. The output is piecewise differentiable through clamp;
    it does not assert the desired map is represented exactly.
    """
    if desired.ndim != 4 or desired.shape[1] != desired.shape[2] or desired.shape[-1] != 2:
        raise ValueError("desired must have shape (batch,side,side,2)")
    if desired.shape[1] < 3 or desired.dtype not in (torch.float32, torch.float64):
        raise ValueError("side >= 3 and floating input required")
    if passes < 1 or not 0 < minimum < base < maximum:
        raise ValueError("passes and strict conductance bounds are invalid")
    if gauge not in ("centered", "range"):
        raise ValueError("gauge must be centered or range")
    batch, side = desired.shape[:2]
    if not bool(torch.isfinite(desired).all()):
        raise ValueError("desired map must be finite")
    line = torch.linspace(0, 1, side, device=desired.device, dtype=desired.dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    source = torch.stack((xx, yy), dim=-1)
    boundary_valid = (
        torch.allclose(desired[:, 0], source[None, 0], atol=1e-6, rtol=0)
        and torch.allclose(desired[:, -1], source[None, -1], atol=1e-6, rtol=0)
        and torch.allclose(desired[:, :, 0], source[None, :, 0], atol=1e-6, rtol=0)
        and torch.allclose(desired[:, :, -1], source[None, :, -1], atol=1e-6, rtol=0)
    )
    if not boundary_valid:
        raise ValueError("desired map must retain the identity boundary")
    horizontal = desired.new_full((batch, side, side - 1), base)
    vertical = desired.new_full((batch, side - 1, side), base)
    diagonal = desired.new_full((batch, side - 1, side - 1), base)
    epsilon = 1e-6 * (maximum - minimum)
    statistics = []
    for index in range(passes):
        residual = _laplacian(desired, horizontal, vertical, diagonal)
        correction_h, correction_v = _integrated_edge_correction(residual)
        raw_h, raw_v = horizontal + correction_h, vertical + correction_v
        if gauge == "range":
            def shift_to_bounds(field: torch.Tensor, dimension: int) -> torch.Tensor:
                lower = minimum + epsilon - field.amin(dim=dimension, keepdim=True)
                upper = maximum - epsilon - field.amax(dim=dimension, keepdim=True)
                zero = torch.zeros_like(lower)
                nearest = torch.maximum(lower, torch.minimum(zero, upper))
                shift = torch.where(lower <= upper, nearest, 0.5 * (lower + upper))
                return field + shift
            raw_h = torch.cat((raw_h[:, :1], shift_to_bounds(raw_h[:, 1:-1], 2), raw_h[:, -1:]), dim=1)
            raw_v = torch.cat((raw_v[:, :, :1], shift_to_bounds(raw_v[:, :, 1:-1], 1), raw_v[:, :, -1:]), dim=2)
        clipped_h = raw_h.clamp(minimum + epsilon, maximum - epsilon)
        clipped_v = raw_v.clamp(minimum + epsilon, maximum - epsilon)
        horizontal, vertical = clipped_h, clipped_v
        if return_stats:
            new_residual = _laplacian(desired, horizontal, vertical, diagonal)[:, 1:-1, 1:-1]
            statistics.append({
                "pass": index + 1,
                "residual_l2": float(torch.linalg.vector_norm(new_residual)),
                "unclipped_horizontal_min": float(raw_h.min()),
                "unclipped_horizontal_max": float(raw_h.max()),
                "unclipped_vertical_min": float(raw_v.min()),
                "unclipped_vertical_max": float(raw_v.max()),
                "clipped_horizontal_fraction": float((raw_h != clipped_h).float().mean()),
                "clipped_vertical_fraction": float((raw_v != clipped_v).float().mean()),
            })
    def to_logit(weight: torch.Tensor) -> torch.Tensor:
        probability = ((weight - minimum) / (maximum - minimum)).clamp(1e-6, 1 - 1e-6)
        return torch.logit(probability)
    logits = (to_logit(horizontal), to_logit(vertical), to_logit(diagonal))
    return logits, {"gauge": gauge, "passes": statistics} if return_stats else None
