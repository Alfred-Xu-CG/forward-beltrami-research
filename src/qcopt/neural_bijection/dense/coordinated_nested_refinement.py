"""Cached linear, diagonal-aware refinement of a declared nested P1 map.

No topology is inferred from interpolation alone: callers check rounded output.
Buffers describe fixed SOURCE queries, not the changing mapped geometry.
"""
from __future__ import annotations

import torch

from .coordinated_fixed_sampling import FrozenP1Evaluator


class FrozenNestedP1Refinement(torch.nn.Module):
    def __init__(self, coarse_side, fine_side, *, diagonal="ac", dtype=torch.float64,
                 device="cpu"):
        super().__init__()
        if (any(isinstance(n,bool) or not isinstance(n,int) or n<3
                for n in (coarse_side,fine_side)) or fine_side<coarse_side or
                (fine_side-1)%(coarse_side-1)):
            raise ValueError("nested integer-refinement control sizes required")
        if diagonal not in ("ac","bd"):
            raise ValueError("unchanged ac/bd diagonal required")
        self.coarse_side,self.fine_side=coarse_side,fine_side
        self.factor=(fine_side-1)//(coarse_side-1)
        self.diagonal=diagonal
        if coarse_side==fine_side:
            self.evaluator=None
        else:
            y,x=torch.meshgrid(torch.linspace(0,1,fine_side,dtype=dtype,device=device),
                torch.linspace(0,1,fine_side,dtype=dtype,device=device),indexing="ij")
            self.evaluator=FrozenP1Evaluator(coarse_side,coarse_side,
                torch.stack((x,y),-1)[None],diagonal)

    def forward(self,vertices):
        if (vertices.ndim!=4 or vertices.shape[1:]!=(self.coarse_side,self.coarse_side,2)
                or not vertices.is_floating_point()):
            raise ValueError("floating vertices(B,coarse,coarse,2) required")
        if self.evaluator is None:
            return vertices
        fine=self.evaluator(vertices)
        # Exactly retain old nodes; this also retains the exact corresponding VJP.
        fine[:,::self.factor,::self.factor]=vertices
        return fine

    @property
    def resident_bytes(self):
        return sum(b.numel()*b.element_size() for b in self.buffers())
