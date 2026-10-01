"""Independent coordinated scales on disjoint affected-cell supports.

This changes one shared vertex table, not a composition or resampling of maps.
Each patch freezes its perimeter; adjacent patch footprints may share those
frozen vertices, but never an affected cell. Four-corner positivity and the
unchanged patch boundary preserve each patch's map under the globally valid
anchor assumption. This local operation does not certify that assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .coordinated_update import CoordinatedQ1Update
from .digital_q1 import q1_corner_determinants


@dataclass
class CoordinatedPatchResult:
    vertices: torch.Tensor
    amplitude: torch.Tensor
    patch_scales: torch.Tensor
    patch_gauges: torch.Tensor
    patch_alpha_max: torch.Tensor
    normalized_margin_min: torch.Tensor
    covered_cells: int
    uncovered_cells: int


class CoordinatedPatchQ1Pass(torch.nn.Module):
    """One nonconflicting P-cell patch pass with a smooth proposal window.

    ``proposal`` has shape (B,R,C) in physical displacement units. A
    sin²(pi*r/P) sin²(pi*c/P) window gives a P-cell transition and is exactly
    zero on each patch edge. The fine-grid reference patch is passed intact
    to the existing safe operator: a local unit-square reference would use
    the wrong determinant normalization. Every patch uses the same direction
    but an independent radial or analytic scale.

    Complete patches start at the declared offsets, in increments of P.
    Uncovered leading/tail cells remain unchanged and are counted explicitly.
    An empty layout is an error. Four P/2-shifted passes can subsequently
    update seams, but they must execute sequentially with refreshed geometry.

    The anchor is assumed to be a global homeomorphism with the declared
    fixed boundary. ``validate`` checks finite coordinates and all actual
    corner margins, not global boundary injectivity or saved binary signs.
    """

    def __init__(self, rows: int, columns: int | None = None, patch_cells: int = 8, *,
                 direction: tuple[float, float] = (1., 0.), mode: str = "radial",
                 offset_row: int = 0, offset_column: int = 0,
                 minimum_jacobian: float = .001, theta: float = .95) -> None:
        super().__init__()
        columns = rows if columns is None else columns
        if not all(isinstance(value, int) for value in (rows, columns, patch_cells, offset_row, offset_column)):
            raise ValueError("integer grid, patch size and offsets required")
        if min(rows, columns) < 3 or patch_cells < 2:
            raise ValueError("at least 3 grid vertices per axis and P>=2 required")
        if not 0 <= offset_row < patch_cells or not 0 <= offset_column < patch_cells:
            raise ValueError("offsets must lie in [0,P)")
        starts = [(row, column) for row in range(offset_row, rows-patch_cells, patch_cells)
                  for column in range(offset_column, columns-patch_cells, patch_cells)]
        if not starts:
            raise ValueError("layout contains no complete patch")
        self.rows, self.columns, self.patch_cells = rows, columns, patch_cells
        self.offset_row, self.offset_column = offset_row, offset_column
        self.operator = CoordinatedQ1Update(direction, mode=mode, boundary="fixed",
                                             minimum_jacobian=minimum_jacobian, theta=theta)
        ids = torch.tensor([[[((row+r)*columns+column+c) for c in range(patch_cells+1)]
                             for r in range(patch_cells+1)] for row,column in starts], dtype=torch.long)
        interior = ids[:,1:-1,1:-1].flatten()
        affected = torch.tensor([[(row+r)*(columns-1)+column+c for r in range(patch_cells)
                                 for c in range(patch_cells)] for row,column in starts], dtype=torch.long)
        if torch.unique(interior).numel()!=interior.numel() or torch.unique(affected).numel()!=affected.numel():
            raise AssertionError("patch interiors and affected-cell sets must be disjoint")
        coordinate = torch.arange(patch_cells+1, dtype=torch.float64)/patch_cells
        axis = torch.sin(torch.pi*coordinate).square()
        axis[0] = axis[-1] = 0  # sin(pi) is not exactly zero in floating point.
        window = axis[:,None]*axis[None,:]
        self.register_buffer("patch_ids",ids,persistent=False)
        self.register_buffer("interior_ids",interior,persistent=False)
        self.register_buffer("affected_cell_ids",affected,persistent=False)
        self.register_buffer("patch_starts",torch.tensor(starts,dtype=torch.long),persistent=False)
        self.register_buffer("window",window,persistent=False)

    @property
    def patch_count(self) -> int:
        return self.patch_ids.shape[0]

    @property
    def covered_cells(self) -> int:
        return self.affected_cell_ids.numel()

    @property
    def uncovered_cells(self) -> int:
        return (self.rows-1)*(self.columns-1)-self.covered_cells

    def _proposal_patches(self, proposal: torch.Tensor) -> torch.Tensor:
        if proposal.shape[1:] != (self.rows,self.columns) or proposal.ndim!=3:
            raise ValueError("proposal must have shape (B,R,C) for the declared grid")
        return proposal.reshape(proposal.shape[0],-1)[:,self.patch_ids]*self.window.to(proposal.dtype)

    def windowed_proposal(self, proposal: torch.Tensor) -> torch.Tensor:
        """Expose the identical tapered field for a fair global-scale comparison."""
        patch = self._proposal_patches(proposal)
        output = torch.zeros_like(proposal).flatten(1)
        return output.index_copy(1,self.interior_ids,patch[:,:,1:-1,1:-1].flatten(1)).reshape_as(proposal)

    def _reference(self, vertices: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
        if reference is None:
            y,x = torch.meshgrid(torch.arange(self.rows,dtype=torch.float64,device=vertices.device)/(self.rows-1),
                                 torch.arange(self.columns,dtype=torch.float64,device=vertices.device)/(self.columns-1),
                                 indexing="ij")
            reference = torch.stack((x,y),dim=-1)[None]
        elif reference.ndim!=4 or reference.shape[1:]!=vertices.shape[1:] or reference.shape[0] not in (1,vertices.shape[0]):
            raise ValueError("reference must match grid and have batch 1 or B")
        return reference.to(device=vertices.device,dtype=torch.float64).expand(vertices.shape[0],-1,-1,-1)

    def forward(self, vertices: torch.Tensor, proposal: torch.Tensor, *,
                reference: torch.Tensor | None = None,
                alpha_trial: float | torch.Tensor = 1., validate: bool = True) -> CoordinatedPatchResult:
        if vertices.ndim!=4 or vertices.shape[1:]!=(self.rows,self.columns,2) or vertices.shape[0]<1:
            raise ValueError("vertices must have declared (B,R,C,2) shape")
        if proposal.shape!=vertices.shape[:-1] or proposal.dtype!=vertices.dtype or proposal.device!=vertices.device:
            raise ValueError("proposal must have matching shape, dtype and device")
        batch = vertices.shape[0]
        reference64 = self._reference(vertices,reference)
        qref = q1_corner_determinants(reference64)
        if validate:
            anchor_margin = q1_corner_determinants(vertices.double())/qref-self.operator.minimum_jacobian
            if not bool(torch.isfinite(vertices).all() and torch.isfinite(proposal).all()
                        and torch.isfinite(reference64).all() and (qref>0).all()
                        and torch.isfinite(anchor_margin).all() and (anchor_margin>0).all()):
                raise ValueError("finite full-grid anchor/reference corner margins required")
        patch = vertices.reshape(batch,-1,2)[:,self.patch_ids]
        source_patch = reference64.reshape(batch,-1,2)[:,self.patch_ids]
        raw = self._proposal_patches(proposal)
        width = self.patch_cells+1
        trial = torch.as_tensor(alpha_trial,dtype=torch.float64,device=vertices.device)
        if trial.ndim==1 and trial.shape==(batch,):
            trial = trial[:,None].expand(-1,self.patch_count).flatten()
        elif trial.ndim==2 and trial.shape==(batch,self.patch_count):
            trial = trial.flatten()
        elif trial.ndim!=0:
            raise ValueError("alpha_trial must be scalar, (B,), or (B,patch_count)")
        result = self.operator(patch.reshape(-1,width,width,2),raw.reshape(-1,width,width),
                               reference=source_patch.reshape(-1,width,width,2),
                               alpha_trial=trial,validate=validate)
        updated = result.vertices.reshape(batch,self.patch_count,width,width,2)[:,:,1:-1,1:-1]
        candidate = vertices.reshape(batch,-1,2).index_copy(1,self.interior_ids,updated.reshape(batch,-1,2))
        candidate = candidate.reshape_as(vertices)
        amplitude = torch.zeros_like(proposal).flatten(1).index_copy(
            1,self.interior_ids,result.amplitude.reshape(batch,self.patch_count,width,width)[:,:,1:-1,1:-1].flatten(1),
        ).reshape_as(proposal)
        margins = q1_corner_determinants(candidate.double())/qref-self.operator.minimum_jacobian
        margin_min = margins.flatten(1).amin(1)
        if validate and not bool(torch.isfinite(candidate).all() and (margin_min>0).all()):
            raise RuntimeError("actual regional output failed full-grid corner margins")
        return CoordinatedPatchResult(candidate,amplitude,result.scale.reshape(batch,-1),
                                      result.gauge.reshape(batch,-1),result.alpha_max.reshape(batch,-1),
                                      margin_min,self.covered_cells,self.uncovered_cells)
