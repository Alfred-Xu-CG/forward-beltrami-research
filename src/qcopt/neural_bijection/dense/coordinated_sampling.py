"""Actual P1 evaluation on the fixed source triangulation (no map resampling)."""
from __future__ import annotations

import torch

from .q1_image_sampling import fixed_pixel_centers


def p1_map_at_queries(vertices: torch.Tensor, queries: torch.Tensor, diagonal: str="ac", *, validate_queries: bool=True) -> torch.Tensor:
    """Evaluate either diagonal, with a,b,c,d=(00,10,11,01) cell coordinates.

    Vertices: (B,R,C,2); queries: (1 or B,...,2) in[0,1]^2. This locates
    queries in the FIXED reference mesh, not the deformed target mesh. Values
    are continuous across triangle edges; derivatives there need not be unique.
    """
    if diagonal not in ("ac","bd"):
        raise ValueError("diagonal must be ac or bd")
    if vertices.ndim!=4 or vertices.shape[0]<1 or vertices.shape[-1]!=2 or min(vertices.shape[1:3])<2:
        raise ValueError("vertices must have shape(B,R>=2,C>=2,2)")
    if queries.ndim<3 or queries.shape[-1]!=2 or queries.shape[0] not in (1,vertices.shape[0]):
        raise ValueError("queries must have shape(1 or B,...,2)")
    if queries.device!=vertices.device or queries.dtype!=vertices.dtype:
        raise ValueError("matching geometry/query dtype and device required")
    if validate_queries and not bool(torch.isfinite(queries).all() and (queries>=0).all() and (queries<=1).all()):
        raise ValueError("queries must be finite and inside[0,1]^2")
    batch,rows,columns,_=vertices.shape
    shape=(batch,*queries.shape[1:])
    query=queries.expand(batch,*queries.shape[1:]).reshape(batch,-1,2)
    x,y=query[...,0]*(columns-1),query[...,1]*(rows-1)
    column=x.floor().long().clamp(0,columns-2)
    row=y.floor().long().clamp(0,rows-2)
    xi,zeta=(x-column)[...,None],(y-row)[...,None]
    index=row*columns+column
    table=vertices.reshape(batch,-1,2)
    fetch=lambda offset: table.gather(1,(index+offset)[...,None].expand(-1,-1,2))
    a,b,c,d=fetch(0),fetch(1),fetch(columns+1),fetch(columns)
    if diagonal=="ac":
        lower=(1-xi)*a+(xi-zeta)*b+zeta*c
        upper=(1-zeta)*a+xi*c+(zeta-xi)*d
        result=torch.where(zeta<=xi,lower,upper)
    else:
        lower=(1-xi-zeta)*a+xi*b+zeta*d
        upper=(1-zeta)*b+(xi+zeta-1)*c+(1-xi)*d
        result=torch.where(xi+zeta<=1,lower,upper)
    return result.reshape(shape)


def p1_map_at_pixel_centers(vertices: torch.Tensor,height: int,width: int,diagonal: str="ac") -> torch.Tensor:
    queries=fixed_pixel_centers(height,width,dtype=vertices.dtype,device=vertices.device)
    # Constructed pixel centers satisfy the public domain precondition exactly.
    return p1_map_at_queries(vertices,queries,diagonal,validate_queries=False)
