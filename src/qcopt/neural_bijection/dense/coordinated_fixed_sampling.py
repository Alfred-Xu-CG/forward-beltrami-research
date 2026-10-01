"""Frozen reference P1 query geometry; differentiable only in mapped vertices."""
from __future__ import annotations

import torch
from torch import nn


class FrozenP1Evaluator(nn.Module):
    """Evaluate one fixed shared query array on any batch of vertex tables.

    Queries have shape (1,...,2) in [0,1]^2 on a uniform source grid. The
    declared global diagonal and source grid never change. Geometry is copied
    into three selected vertex IDs and barycentric weights per query. Query
    gradients are deliberately unsupported (and rejected), not detached.

    Forward is linear in vertices and its VJP is the usual weighted scatter.
    Buffers can be moved/cast with Module.to; vertices must match the resulting
    weight dtype/device. Casting stored weights is ordinary rounded conversion,
    not recomputation from differently rounded query coordinates. This is not
    a sampler on the deformed target triangulation or a topology certificate.
    """

    def __init__(self, rows: int, columns: int, queries: torch.Tensor, diagonal: str="ac"):
        super().__init__()
        if any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in (rows,columns)):
            raise ValueError("rows and columns must be integers >=2")
        if diagonal not in ("ac","bd"):
            raise ValueError("diagonal must be ac or bd")
        if queries.requires_grad:
            raise ValueError("fixed queries must not require gradients")
        if queries.ndim<3 or queries.shape[0]!=1 or queries.shape[-1]!=2 or queries.numel()==0:
            raise ValueError("queries must have nonempty shape(1,...,2), shared across batches")
        if not queries.is_floating_point():
            raise ValueError("queries must have floating dtype")
        if not bool(torch.isfinite(queries).all() and (queries>=0).all() and (queries<=1).all()):
            raise ValueError("queries must be finite and inside[0,1]^2")
        self.rows,self.columns,self.diagonal=rows,columns,diagonal
        self.query_shape=tuple(queries.shape[1:-1])
        # Setup alone locates queries in the ORIGINAL fixed reference mesh.
        q=queries.reshape(-1,2)
        x,y=q[:,0]*(columns-1),q[:,1]*(rows-1)
        column=x.floor().long().clamp(0,columns-2)
        row=y.floor().long().clamp(0,rows-2)
        xi,zeta=x-column,y-row
        a=row*columns+column
        b,c,d=a+1,a+columns+1,a+columns
        if diagonal=="ac":
            choose=zeta<=xi
            lower_ids=torch.stack((a,b,c),-1)
            upper_ids=torch.stack((a,c,d),-1)
            lower_weights=torch.stack((1-xi,xi-zeta,zeta),-1)
            upper_weights=torch.stack((1-zeta,xi,zeta-xi),-1)
        else:
            choose=xi+zeta<=1
            lower_ids=torch.stack((a,b,d),-1)
            upper_ids=torch.stack((b,c,d),-1)
            lower_weights=torch.stack((1-xi-zeta,xi,zeta),-1)
            upper_weights=torch.stack((1-zeta,xi+zeta-1,1-xi),-1)
        self.register_buffer("vertex_ids",torch.where(choose[:,None],lower_ids,upper_ids))
        self.register_buffer("weights",torch.where(choose[:,None],lower_weights,upper_weights))

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        if vertices.ndim!=4 or vertices.shape[0]<1 or tuple(vertices.shape[1:])!=(self.rows,self.columns,2):
            raise ValueError("vertices must have shape(B>=1,rows,columns,2)")
        if vertices.dtype!=self.weights.dtype or vertices.device!=self.weights.device:
            raise ValueError("vertices must match buffer dtype and device")
        batch=vertices.shape[0]
        selected=vertices.reshape(batch,-1,2).index_select(1,self.vertex_ids.reshape(-1))
        selected=selected.reshape(batch,-1,3,2)
        result=(selected*self.weights[None,:,:,None]).sum(2)
        return result.reshape(batch,*self.query_shape,2)
