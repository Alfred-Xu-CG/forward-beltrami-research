"""Solve-free hierarchical convex-cell homeomorphisms on a dyadic grid."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def split_convex_quad_grid(
    grid: torch.Tensor,
    horizontal_logits: torch.Tensor | None,
    vertical_logits: torch.Tensor | None,
    *,
    fraction_span: float = 0.25,
    center_logits: torch.Tensor | None = None,
    center_span: float = 0.35,
) -> torch.Tensor:
    """Refine every convex cell; shared edge points are computed only once.

    Input grid has shape (batch,n,n,2), with image-row order increasing in y.
    The two logit arrays have shapes (batch,n,n-1) and (batch,n-1,n).
    Boundary split fractions are exactly one half and do not use logits.
    """
    if grid.ndim != 4 or grid.shape[1] != grid.shape[2] or grid.shape[-1] != 2:
        raise ValueError("grid must have shape (batch,n,n,2)")
    if not 0.0 < fraction_span < 0.5:
        raise ValueError("fraction_span must lie strictly between zero and one half")
    if not 0.0 < center_span < 0.5:
        raise ValueError("center_span must lie strictly between zero and one half")
    batch, n, _, _ = grid.shape
    if horizontal_logits is None or vertical_logits is None:
        if not (horizontal_logits is None and vertical_logits is None and n == 2):
            raise ValueError("only the root 2-by-2 grid may omit both logit arrays")
        h_fraction = grid.new_full((batch, n, n - 1), 0.5)
        v_fraction = grid.new_full((batch, n - 1, n), 0.5)
    else:
        if horizontal_logits.shape != (batch, n, n - 1):
            raise ValueError("horizontal logits have the wrong shape")
        if vertical_logits.shape != (batch, n - 1, n):
            raise ValueError("vertical logits have the wrong shape")
        if horizontal_logits.device != grid.device or vertical_logits.device != grid.device:
            raise ValueError("all tensors must be on the same device")
        if horizontal_logits.dtype != grid.dtype or vertical_logits.dtype != grid.dtype:
            raise ValueError("all tensors must have the same dtype")
        h_fraction = 0.5 + fraction_span * torch.tanh(horizontal_logits)
        v_fraction = 0.5 + fraction_span * torch.tanh(vertical_logits)
        h_interior = torch.ones((1, n, 1), device=grid.device, dtype=torch.bool)
        h_interior[:, 0, :] = False
        h_interior[:, -1, :] = False
        v_interior = torch.ones((1, 1, n), device=grid.device, dtype=torch.bool)
        v_interior[:, :, 0] = False
        v_interior[:, :, -1] = False
        h_fraction = torch.where(h_interior, h_fraction, 0.5)
        v_fraction = torch.where(v_interior, v_fraction, 0.5)

    horizontal = grid[:, :, :-1] + h_fraction[..., None] * (grid[:, :, 1:] - grid[:, :, :-1])
    vertical = grid[:, :-1] + v_fraction[..., None] * (grid[:, 1:] - grid[:, :-1])
    bottom = horizontal[:, :-1]
    top = horizontal[:, 1:]
    left = vertical[:, :, :-1]
    right = vertical[:, :, 1:]
    if center_logits is None:
        bottom_to_top = top - bottom
        left_to_right = right - left
        denominator = _cross(bottom_to_top, left_to_right)
        numerator = _cross(left - bottom, left_to_right)
        center = bottom + (numerator / denominator)[..., None] * bottom_to_top
    else:
        if center_logits.shape != (batch, n - 1, n - 1, 2):
            raise ValueError("center logits have the wrong shape")
        if center_logits.device != grid.device or center_logits.dtype != grid.dtype:
            raise ValueError("center logits must match the grid's device and dtype")
        fractions = 0.5 + center_span * torch.tanh(center_logits)
        u = fractions[..., 0, None]
        v = fractions[..., 1, None]
        # Strictly positive bilinear weights on the cyclic inner quad B,R,T,L.
        center = (1 - v) * ((1 - u) * bottom + u * right) + v * ((1 - u) * left + u * top)

    fine = grid.new_empty((batch, 2 * n - 1, 2 * n - 1, 2))
    fine[:, ::2, ::2] = grid
    fine[:, ::2, 1::2] = horizontal
    fine[:, 1::2, ::2] = vertical
    fine[:, 1::2, 1::2] = center
    return fine


def certify_convex_quad_output(control: torch.Tensor) -> float:
    """Check the represented P1 grid, not merely exact-arithmetic hypotheses."""
    if control.ndim != 4 or control.shape[1] != control.shape[2] or control.shape[-1] != 2:
        raise ValueError("control must have shape (batch,n,n,2)")
    side = control.shape[1]
    if not torch.isfinite(control).all().item():
        raise ValueError("nonfinite output coordinate")
    if not torch.all((control >= 0.0) & (control <= 1.0)).item():
        raise ValueError("represented output lies outside the unit square")
    # Evaluate orientation in float64 even when the returned coordinates are
    # float32, and demand a margin above elementary roundoff at the explicitly
    # checked unit scale. This is a numerical filter, not exact-predicate proof.
    represented = control.to(torch.float64)
    a = represented[:, :-1, :-1]
    b = represented[:, :-1, 1:]
    c = represented[:, 1:, 1:]
    d = represented[:, 1:, :-1]
    lower = _cross(b - a, c - a)
    upper = _cross(c - a, d - a)
    minimum_double_area = min(lower.min().item(), upper.min().item())
    minimum = minimum_double_area * (side - 1)**2
    if minimum_double_area <= 64.0 * torch.finfo(torch.float64).eps:
        raise ValueError("represented output has a nonpositive or numerically unresolved face")
    line = torch.arange(side, device=control.device, dtype=control.dtype) / (side - 1)
    if not (
        torch.equal(control[:, 0, :, 0], line.expand(control.shape[0], -1))
        and torch.equal(control[:, -1, :, 0], line.expand(control.shape[0], -1))
        and torch.equal(control[:, :, 0, 1], line.expand(control.shape[0], -1))
        and torch.equal(control[:, :, -1, 1], line.expand(control.shape[0], -1))
        and torch.all(control[:, 0, :, 1] == 0).item()
        and torch.all(control[:, -1, :, 1] == 1).item()
        and torch.all(control[:, :, 0, 0] == 0).item()
        and torch.all(control[:, :, -1, 0] == 1).item()
    ):
        raise ValueError("represented boundary is not pointwise identity")
    return minimum


class HierarchicalConvexQuadLayer(torch.nn.Module):
    """P1 homeomorphism on the final standard triangulation, without a solve.

    For side 2^L+1, input one (horizontal, vertical) logit pair at each
    current side 3,5,...,2^(L-1)+1. The first 2->3 subdivision is fixed,
    because all four root-square edges are boundary edges.
    """

    def __init__(self, side: int, *, fraction_span: float = 0.25, certify: bool = True) -> None:
        super().__init__()
        if side < 5 or (side - 1) & (side - 2):
            raise ValueError("side must be 2^L+1 with L at least two")
        if not 0.0 < fraction_span < 0.5:
            raise ValueError("fraction_span must lie strictly between zero and one half")
        self.side = side
        self.fraction_span = fraction_span
        self.certify = certify
        self.latent_sides = tuple(2**k + 1 for k in range(1, (side - 1).bit_length() - 1))

    def forward(self, latents: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if len(latents) != len(self.latent_sides):
            raise ValueError("one logit pair is required per non-root dyadic level")
        first = latents[0][0]
        if first.ndim != 3:
            raise ValueError("logits must have a batch dimension")
        batch = first.shape[0]
        grid = first.new_tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]).unsqueeze(0).expand(batch, -1, -1, -1)
        grid = split_convex_quad_grid(grid, None, None, fraction_span=self.fraction_span)
        for expected_side, (horizontal, vertical) in zip(self.latent_sides, latents):
            if grid.shape[1] != expected_side:
                raise RuntimeError("dyadic level bookkeeping error")
            grid = split_convex_quad_grid(grid, horizontal, vertical, fraction_span=self.fraction_span)
        if self.certify:
            certify_convex_quad_output(grid)
        return grid


class HierarchicalConvexQuadFreeCenterLayer(HierarchicalConvexQuadLayer):
    """More expressive convex hierarchy with two interior-center logits/cell."""

    def __init__(
        self,
        side: int,
        *,
        fraction_span: float = 0.25,
        center_span: float = 0.35,
        certify: bool = True,
    ) -> None:
        super().__init__(side, fraction_span=fraction_span, certify=certify)
        if not 0.0 < center_span < 0.5:
            raise ValueError("center_span must lie strictly between zero and one half")
        self.center_span = center_span

    def forward(
        self,
        root_center_logits: torch.Tensor,
        latents: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if len(latents) != len(self.latent_sides):
            raise ValueError("one edge/center logit triple is required per non-root level")
        if root_center_logits.ndim != 4 or root_center_logits.shape[1:] != (1, 1, 2):
            raise ValueError("root center logits must have shape (batch,1,1,2)")
        batch = root_center_logits.shape[0]
        grid = root_center_logits.new_tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]).unsqueeze(0).expand(batch, -1, -1, -1)
        grid = split_convex_quad_grid(
            grid, None, None, fraction_span=self.fraction_span,
            center_logits=root_center_logits, center_span=self.center_span,
        )
        for expected_side, (horizontal, vertical, center) in zip(self.latent_sides, latents):
            if grid.shape[1] != expected_side:
                raise RuntimeError("dyadic level bookkeeping error")
            grid = split_convex_quad_grid(
                grid, horizontal, vertical, fraction_span=self.fraction_span,
                center_logits=center, center_span=self.center_span,
            )
        if self.certify:
            certify_convex_quad_output(grid)
        return grid
