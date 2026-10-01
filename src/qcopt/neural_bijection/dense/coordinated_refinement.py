"""Integer refinement of the SAME globally declared P1 source triangulation.

In exact arithmetic uniform factor m on both axes and unchanged ac/bd diagonals
refines every old triangle. A fine cell within one triangle has its affine face
Jacobian; a fine cell crossed by an old diagonal is a translated 1/m copy of the
mapped old quad. Thus the function is unchanged, and all four normalized corner
determinants inherit the old minimum for strictly convex mapped old quads.

These are real-arithmetic statements. Rounded output coordinates require fresh
actual-corner checks. Refinement does not certify a previously invalid map or its
global boundary; nor does it preserve a Q1 function or allow changing diagonals.
"""
from __future__ import annotations

import torch

from .coordinated_sampling import p1_map_at_queries


def refine_p1_vertices(vertices: torch.Tensor, factor: int, diagonal: str="ac") -> torch.Tensor:
    """Return (B,m*(R-1)+1,m*(C-1)+1,2), with differentiable vertex values.

    Input node coordinates are the regular unit endpoint mesh. Mapped values may
    be arbitrary finite floating tensors; positivity/global embedding assumptions
    belong to the caller. Original nodes are copied exactly after evaluation to
    avoid a roundoff-only perturbation of the existing vertex table.
    """
    if isinstance(factor,bool) or not isinstance(factor,int) or factor<1:
        raise ValueError("factor must be a positive integer (same factor on both axes)")
    if diagonal not in ("ac","bd"):
        raise ValueError("diagonal must remain ac or bd")
    if vertices.ndim!=4 or vertices.shape[0]<1 or vertices.shape[-1]!=2 or min(vertices.shape[1:3])<2:
        raise ValueError("vertices must have shape(B,R>=2,C>=2,2)")
    if not vertices.is_floating_point() or not bool(torch.isfinite(vertices).all()):
        raise ValueError("finite floating vertices required")
    if factor==1:
        return vertices.clone()
    rows=(vertices.shape[1]-1)*factor+1
    columns=(vertices.shape[2]-1)*factor+1
    y,x=torch.meshgrid(torch.linspace(0,1,rows,device=vertices.device,dtype=vertices.dtype),
                       torch.linspace(0,1,columns,device=vertices.device,dtype=vertices.dtype),indexing="ij")
    fine=p1_map_at_queries(vertices,torch.stack((x,y),-1)[None],diagonal)
    fine[:,::factor,::factor]=vertices
    if not bool(torch.isfinite(fine).all()):
        raise ValueError("rounded P1 refinement produced nonfinite coordinates")
    return fine
