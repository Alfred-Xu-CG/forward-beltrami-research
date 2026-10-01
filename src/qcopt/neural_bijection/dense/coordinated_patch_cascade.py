"""Four sequential shifted regional updates of the SAME vertex table."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .coordinated_patches import CoordinatedPatchQ1Pass


@dataclass
class CoordinatedPatchCascadeResult:
    vertices: torch.Tensor
    patch_scales: torch.Tensor
    patch_gauges: torch.Tensor
    normalized_margin_min: torch.Tensor
    pass_margin_min: torch.Tensor
    geometry_pass_count: int
    pass_patch_counts: tuple[int, ...]
    pass_covered_cells: tuple[int, ...]
    pass_uncovered_cells: tuple[int, ...]


class CoordinatedPatchCascade(torch.nn.Module):
    """Common-direction four-pass cascade; no composition or map resampling.

    Each pass uses the SAME physical raw scalar proposal, independently tapered
    by the existing primitive, and refreshes its geometry from the preceding
    connected output. Global/material reference is unchanged. Offsets are
    (0,0), (P/2,0), (0,P/2), (P/2,P/2). Aggregate diagnostics concatenate patches
    in that order; pass_patch_counts identifies the splits. There is no detach
    of geometry or proposal and no new safety-scale formula.

    Even P>=2 and at least 2P cells per axis ensure every shifted layout has a
    complete patch. Complete patches only: nondivisible leading/tail regions
    follow the primitive's rules. Per-pass affected-cell coverage is NOT union
    interior-vertex coverage. Divisible grids cover all interior vertices over
    the four passes. More generally all interior vertices are covered iff both
    cell counts are divisible by P/2; otherwise trailing vertices can remain
    frozen. The global boundary is fixed. Fixed P has decreasing physical
    support width as the grid is refined; no scale-independent support claim.

    As with the primitive, a globally homeomorphic anchor and valid reference
    are caller assumptions; actual all-four corner checks do not establish
    global injectivity of an arbitrary input. Default validation checks each
    rounded substep, not only the final map. This is not an affine/new map class.
    Optional backward_backend='manual' retains full-Y/proposal first-order
    gradients through all four passes, but reference/trial must be constant
    and ALL auxiliary diagnostics are nondifferentiable. Default is ordinary.
    """
    def __init__(self, rows: int, columns: int | None = None, patch_cells: int = 32, *,
                 direction: tuple[float, float] = (1., 0.), mode: str = "radial",
                 minimum_jacobian: float = .001, theta: float = .95,
                 backward_backend: str = "ordinary") -> None:
        super().__init__()
        columns = rows if columns is None else columns
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (rows, columns, patch_cells)):
            raise ValueError("integer grid sizes and patch_cells required")
        if patch_cells < 2 or patch_cells % 2:
            raise ValueError("even patch_cells >=2 required")
        if min(rows-1, columns-1) < 2*patch_cells:
            raise ValueError("at least 2P cells per grid axis required")
        self.rows, self.columns, self.patch_cells = rows, columns, patch_cells
        self.backward_backend=backward_backend
        half = patch_cells//2
        self.offsets = ((0, 0), (half, 0), (0, half), (half, half))
        self.passes = torch.nn.ModuleList([
            CoordinatedPatchQ1Pass(rows, columns, patch_cells, direction=direction, mode=mode,
                                   offset_row=row, offset_column=column,
                                   minimum_jacobian=minimum_jacobian, theta=theta,
                                   backward_backend=backward_backend)
            for row, column in self.offsets
        ])

    def forward(self, vertices: torch.Tensor, proposal: torch.Tensor, *,
                reference: torch.Tensor | None = None,
                alpha_trial: float | torch.Tensor = 1., validate: bool = True
                ) -> CoordinatedPatchCascadeResult:
        if not isinstance(validate, bool):
            raise ValueError("validate must be a Python bool")
        if (not isinstance(vertices, torch.Tensor) or vertices.ndim != 4 or
                vertices.shape[0] < 1 or vertices.shape[1:] != (self.rows,self.columns,2)):
            raise ValueError("vertices must have declared (B,R,C,2) shape")
        if (not isinstance(proposal, torch.Tensor) or proposal.shape != vertices.shape[:-1]
                or proposal.dtype != vertices.dtype or proposal.device != vertices.device):
            raise ValueError("proposal must have matching shape, dtype and device")
        # A single scalar or per-batch trial has identical meaning in all passes.
        # Per-patch arrays cannot be shared because shifted patch counts differ.
        trial = torch.as_tensor(alpha_trial, dtype=torch.float64, device=vertices.device)
        if trial.ndim != 0 and (trial.ndim != 1 or trial.shape != (vertices.shape[0],)):
            raise ValueError("cascade alpha_trial must be scalar or (B,)")
        current = vertices
        results = []
        for layer in self.passes:
            result = layer(current, proposal, reference=reference, alpha_trial=trial, validate=validate)
            current = result.vertices
            results.append(result)
        return CoordinatedPatchCascadeResult(
            current, torch.cat([result.patch_scales for result in results], dim=1),
            torch.cat([result.patch_gauges for result in results], dim=1),
            results[-1].normalized_margin_min,
            torch.stack([result.normalized_margin_min for result in results], dim=1), 4,
            tuple(layer.patch_count for layer in self.passes),
            tuple(result.covered_cells for result in results),
            tuple(result.uncovered_cells for result in results),
        )
