"""Exact-arithmetic coarse quadrature of existing dyadic fine P1 priors.

The functional uses the original forward-edge strain OR actual-face ARAP and
four-corner shape of uniformly refined P1 vertices, not a newly selected coarse
regularizer. Rounded
fine materialization is a different floating-point evaluation: agreement is NOT
bitwise, especially at thin float32 cells. Positive geometry is a caller
assumption; no determinant clamp, fold repair or topology certificate is added.
"""
from __future__ import annotations

import torch

from .coordinated_arap import p1_arap_energy


def _sizes(coarse_side, fine_side):
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2
           for n in (coarse_side, fine_side)):
        raise ValueError("integer coarse/fine sides >=2 required")
    coarse_cells, fine_cells = coarse_side - 1, fine_side - 1
    if (coarse_cells & (coarse_cells - 1) or fine_cells & (fine_cells - 1) or
            fine_cells < coarse_cells or fine_cells % coarse_cells):
        raise ValueError("nested DYADIC coarse/fine cell counts required")
    return coarse_cells, fine_cells, fine_cells // coarse_cells


class ExactNestedP1Priors(torch.nn.Module):
    """Cache ONLY source identity/counts, never changing mapped geometry.

    Constants are private-to-instance buffers built from sizes, not a mutable
    global cache or a detached user reference. Identity is made in float32 THEN
    cast, matching the existing production strain reference. Dyadic source
    coordinates are represented exactly at the supported practical grid sizes.
    Inputs are square (B,coarse_side,coarse_side,2) float32/64 vertex tables;
    outputs are two scalars averaged over batch. ALL mapped vertices may train.
    ``strain_model='displacement_gradient'`` preserves the original weighted
    forward-edge strain. Optional ``'p1_arap'`` uses the actual coarse P1 face
    mean: globally aligned same-diagonal refinement repeats each face Jacobian
    exactly factor**2 times in real arithmetic. Shape ALWAYS remains the same
    fine four-corner quadrature, independent of this strain choice. Rounded fine
    coordinates need not produce bitwise-identical values or derivatives.
    """
    def __init__(self, coarse_side: int, fine_side: int, *, diagonal: str = "ac",
                 dtype=torch.float64, device="cpu",
                 strain_model: str = "displacement_gradient"):
        super().__init__()
        self.coarse_cells, self.fine_cells, self.factor = _sizes(coarse_side, fine_side)
        if diagonal not in ("ac", "bd"):
            raise ValueError("global diagonal must be ac or bd")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("float32/float64 prior precision required")
        if strain_model not in ("displacement_gradient", "p1_arap"):
            raise ValueError("strain_model must be displacement_gradient or p1_arap")
        self.coarse_side, self.fine_side, self.diagonal = coarse_side, fine_side, diagonal
        self.strain_model = strain_model
        axis = torch.arange(coarse_side, dtype=torch.float32, device=device) / self.coarse_cells
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("source_reference", torch.stack((xx, yy), -1)[None].to(dtype))
        weights = torch.full((coarse_side,), float(self.factor**2), dtype=dtype, device=device)
        weights[0] = weights[-1] = self.factor * (self.factor + 1) / 2
        self.register_buffer("edge_weights", weights)

    def forward(self, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (vertices.ndim != 4 or vertices.shape[0] < 1 or
                vertices.shape[1:] != (self.coarse_side, self.coarse_side, 2) or
                vertices.dtype not in (torch.float32, torch.float64) or
                vertices.dtype != self.source_reference.dtype or
                vertices.device != self.source_reference.device):
            raise ValueError("square batch-positive vertices matching float32/64 buffer dtype/device required")
        if self.source_reference.requires_grad or self.edge_weights.requires_grad:
            raise ValueError("source reference/count buffers must remain constant")
        n, m, fine_n = self.coarse_cells, self.factor, self.fine_cells
        if self.strain_model == "p1_arap":
            # Positive geometry remains a caller assumption, as for shape.
            # No detached rotation, determinant clamp or new safety check.
            strain = p1_arap_energy(vertices, self.diagonal, validate=False)
        else:
            residual = vertices - self.source_reference
            dx_residual = (residual[:, :, 1:] - residual[:, :, :-1]) * n
            dy_residual = (residual[:, 1:] - residual[:, :-1]) * n
            numerator = (dx_residual.square().sum(-1) * self.edge_weights[None, :, None]).sum()
            numerator = numerator + (dy_residual.square().sum(-1) * self.edge_weights[None, None, :]).sum()
            strain = numerator / (2 * vertices.shape[0] * fine_n * (fine_n + 1))
        # EXACT same corner columns/order and phi arithmetic as the old prior.
        a, b = vertices[:, :-1, :-1], vertices[:, :-1, 1:]
        d, c = vertices[:, 1:, :-1], vertices[:, 1:, 1:]
        dx = torch.stack((b-a, b-a, c-d, c-d), dim=-2) * n
        dy = torch.stack((d-a, c-b, c-b, d-a), dim=-2) * n
        determinant = dx[..., 0]*dy[..., 1] - dx[..., 1]*dy[..., 0]
        frobenius = dx.square().sum(-1) + dy.square().sum(-1)
        phi = frobenius * (1 + determinant.reciprocal().square()) - 4
        if m == 1:
            shape = phi.mean()
        else:
            i, j = (1, 3) if self.diagonal == "ac" else (0, 2)
            shape = ((m-1)/(2*m) * (phi[..., i]+phi[..., j]) + phi.sum(-1)/(4*m)).mean()
        return strain, shape

    @property
    def resident_bytes(self):
        return sum(buffer.numel()*buffer.element_size() for buffer in self.buffers())


def exact_nested_p1_priors(coarse_vertices: torch.Tensor, fine_side: int,
                           diagonal: str = "ac", *,
                           strain_model: str = "displacement_gradient") -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless API; constant-reference construction is included in this call.

    Use ExactNestedP1Priors to amortize that explicitly recorded construction.
    No support for non-dyadic production-reference rounding is inferred.
    """
    if coarse_vertices.ndim != 4 or coarse_vertices.shape[1] != coarse_vertices.shape[2]:
        raise ValueError("square vertices(B,coarse,coarse,2) required")
    module = ExactNestedP1Priors(coarse_vertices.shape[1], fine_side, diagonal=diagonal,
                               dtype=coarse_vertices.dtype, device=coarse_vertices.device,
                               strain_model=strain_model)
    return module(coarse_vertices)
