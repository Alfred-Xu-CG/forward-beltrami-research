"""Factorized, differentiable fixed-diagonal P1 view of a safe Q1 decoder.

The P1 mesh is specified procedurally, so a dense face table is not retained
in the layer. Finite-precision map exports still need saved-binary validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import torch
from torch import nn

from tools.digital_q1_forward_kernel import forward_kernel_map


def _all_affines_positive_exact(matrix: torch.Tensor) -> bool:
    """Check determinant signs of the actual binary32/binary64 input values."""
    for item in matrix.detach().cpu().numpy():
        a, b, c, d = (Fraction.from_float(float(value)) for value in item.flat)
        if a * d - b * c <= 0:
            return False
    return True


def fixed_sw_ne_faces(side: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Materialize two oriented faces per square only when mesh export needs it."""
    if side < 2:
        raise ValueError("side must be at least two")
    row, col = torch.meshgrid(torch.arange(side - 1, device=device),
                              torch.arange(side - 1, device=device),
                              indexing="ij")
    a = row * side + col
    b = a + 1
    d = a + side
    c = d + 1
    return torch.stack((torch.stack((a, b, c), -1),
                        torch.stack((a, c, d), -1)), -2).reshape(-1, 3)


@dataclass(frozen=True)
class FactorizedP1Map:
    residual_vertices: torch.Tensor  # (batch,side,side,2)
    post_affine_matrix: torch.Tensor  # (batch,2,2), positive determinant
    post_affine_offset: torch.Tensor  # (batch,2)
    diagonal: str = "SW-NE"

    @property
    def side(self) -> int:
        return self.residual_vertices.shape[1]

    def faces(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return fixed_sw_ne_faces(
            self.side, device=self.residual_vertices.device if device is None else device)


def evaluate_factorized_p1(mapped: FactorizedP1Map,
                           queries: torch.Tensor) -> torch.Tensor:
    """Evaluate fixed-diagonal P1 at BxQx2 source queries and postcompose affine."""
    vertices = mapped.residual_vertices
    if vertices.ndim != 4 or vertices.shape[-1] != 2 or (
        vertices.shape[1] != vertices.shape[2] or vertices.shape[1] < 2 or
        queries.ndim != 3 or queries.shape[0] != vertices.shape[0] or
        queries.shape[-1] != 2 or queries.dtype != vertices.dtype or
        queries.device != vertices.device or
        not bool(torch.isfinite(queries).all()) or
        bool(torch.any((queries < 0) | (queries > 1)))
    ):
        raise ValueError("matching BxNxNx2 vertices and BxQx2 unit queries required")
    side = vertices.shape[1]
    cells = side - 1
    scaled = queries * cells
    indices = torch.floor(scaled).long().clamp(max=cells - 1)
    col, row = indices[..., 0], indices[..., 1]
    s = scaled[..., 0] - col
    t = scaled[..., 1] - row
    flat = vertices.reshape(vertices.shape[0], side * side, 2)
    def get(index: torch.Tensor) -> torch.Tensor:
        return torch.gather(flat, 1, index[..., None].expand(-1, -1, 2))
    a = get(row * side + col)
    b = get(row * side + col + 1)
    d = get((row + 1) * side + col)
    c = get((row + 1) * side + col + 1)
    lower = a + s[..., None] * (b - a) + t[..., None] * (c - b)
    upper = a + s[..., None] * (c - d) + t[..., None] * (d - a)
    residual = torch.where((t <= s)[..., None], lower, upper)
    return (torch.bmm(residual, mapped.post_affine_matrix.transpose(1, 2)) +
            mapped.post_affine_offset[:, None, :])


class SafeSparseMatchP1Layer(nn.Module):
    """Batch of independent sparse-match decodes with a procedural P1 output.

    The latent inputs are image-feature match source/target coordinates, not
    raw image pixels. External matching is not made differentiable by this
    layer. The guarantee is exact-arithmetic plus verified exported binary
    maps, subject to the decoder's F1-D hypotheses and an ordered boundary.
    """

    def __init__(self, *, side: int = 257,
                 update_sigmas: tuple[float, ...] = (.12, .06, .03),
                 proposal_mode: str = "residual",
                 gain_mode: str = "fixed") -> None:
        super().__init__()
        self.side = side
        self.update_sigmas = tuple(update_sigmas)
        self.proposal_mode = proposal_mode
        self.gain_mode = gain_mode

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, *,
        calibration_source: torch.Tensor | None = None,
        calibration_target: torch.Tensor | None = None,
        post_affine_matrix: torch.Tensor | None = None,
        post_affine_offset: torch.Tensor | None = None,
    ) -> FactorizedP1Map:
        if source.ndim != 3 or source.shape != target.shape or (
            source.shape[-1] != 2 or source.shape[0] < 1 or
            source.dtype not in (torch.float32, torch.float64) or
            target.dtype != source.dtype or target.device != source.device or
            not bool(torch.isfinite(source).all()) or
            not bool(torch.isfinite(target).all()) or
            bool(torch.any((source < 0) | (source > 1)))
        ):
            raise ValueError("matching finite BxMx2 unit-source and target inputs required")
        batch = source.shape[0]
        if self.gain_mode == "calibrated" and (
            calibration_source is None or calibration_target is None or
            calibration_source.ndim != 3 or
            calibration_source.shape != calibration_target.shape or
            calibration_source.shape[0] != batch or
            calibration_source.shape[-1] != 2 or
            calibration_source.dtype != source.dtype or
            calibration_source.device != source.device or
            calibration_target.dtype != source.dtype or
            calibration_target.device != source.device or
            not bool(torch.isfinite(calibration_source).all()) or
            not bool(torch.isfinite(calibration_target).all()) or
            bool(torch.any((calibration_source < 0) | (calibration_source > 1)))
        ):
            raise ValueError("calibrated mode needs matching in-domain BxKx2 points")
        if post_affine_matrix is None:
            matrix = torch.eye(2, dtype=source.dtype, device=source.device).expand(
                batch, -1, -1)
        else:
            matrix = post_affine_matrix
        if post_affine_offset is None:
            offset = torch.zeros((batch, 2), dtype=source.dtype, device=source.device)
        else:
            offset = post_affine_offset
        if matrix.shape != (batch, 2, 2) or offset.shape != (batch, 2) or (
            matrix.dtype != source.dtype or offset.dtype != source.dtype or
            matrix.device != source.device or offset.device != source.device or
            not bool(torch.isfinite(matrix).all()) or
            not bool(torch.isfinite(offset).all()) or
            not _all_affines_positive_exact(matrix)
        ):
            raise ValueError("exact-positive finite Bx2x2 affine and Bx2 offset required")
        vertices = torch.cat([
            forward_kernel_map(
                source[b], target[b], final_side=self.side,
                update_sigmas=self.update_sigmas,
                proposal_mode=self.proposal_mode, gain_mode=self.gain_mode,
                calibration_source=(None if calibration_source is None else
                                    calibration_source[b]),
                calibration_target=(None if calibration_target is None else
                                    calibration_target[b]),
            ) for b in range(batch)
        ], dim=0)
        return FactorizedP1Map(vertices, matrix, offset)
