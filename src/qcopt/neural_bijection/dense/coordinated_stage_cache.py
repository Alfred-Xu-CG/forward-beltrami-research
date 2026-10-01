"""Fixed-anchor instance-stage cache, NOT a trainable-anchor neural layer.

Anchor/reference are cloned constants. A changed anchor requires construction
of a new cache. Existing gauge arithmetic, proposal gradients, diagnostics and
actual rounded-output corner evaluation are retained. No dtype/device conversion
API is exposed: construct on the intended device and precision.
"""
from __future__ import annotations

import torch

from .coordinated_update import CoordinatedQ1Update,_cross
from .digital_q1 import q1_corner_determinants


class _BoundLayer(CoordinatedQ1Update):
    def __init__(self,anchor,reference,**options):
        super().__init__(**options)
        current=anchor.double()
        self._cached_qref=q1_corner_determinants(reference)
        self._cached_slack=(q1_corner_determinants(current)/self._cached_qref-self.minimum_jacobian).flatten(1)
        if not bool(torch.isfinite(current).all() and torch.isfinite(self._cached_slack).all()
                    and (self._cached_qref>0).all() and (self._cached_slack>0).all()):
            raise ValueError("finite anchor and positive reference/current margins required")
        a,b,c,d=current[:,:-1,:-1],current[:,:-1,1:],current[:,1:,1:],current[:,1:,:-1]
        e=current.new_tensor(self.direction)
        triples=((a,b,d),(a,b,c),(d,b,c),(a,c,d))
        self._k1=torch.stack([_cross(e,r-p) for p,q,r in triples],-1)
        self._k2=torch.stack([_cross(q-p,e) for p,q,r in triples],-1)

    def _constraints(self,vertices,proposal,reference):
        # This private layer is called ONLY on the outer cache's bound anchor.
        ua,ub,uc,ud=proposal[:,:-1,:-1],proposal[:,:-1,1:],proposal[:,1:,1:],proposal[:,1:,:-1]
        first=torch.stack((ub-ua,ub-ua,ub-ud,uc-ua),-1)
        second=torch.stack((ud-ua,uc-ua,uc-ud,ud-ua),-1)
        changes=(first*self._k1+second*self._k2)/self._cached_qref
        return self._cached_slack,changes.flatten(1),self._cached_qref


class FrozenAnchorCoordinatedUpdate:
    """Callable scalar-latent decoder for ONE constant instance-stage anchor.

    Returns the ordinary CoordinatedUpdateResult, including differentiable
    proposal-dependent diagnostics. Rejects trainable anchor/reference instead
    of silently detaching them. Internal constant geometry uses float64.
    """
    def __init__(self,anchor,*,reference=None,direction=(1.,0.),mode="radial",
                 minimum_jacobian=.001,theta=.95):
        if anchor.requires_grad or (reference is not None and reference.requires_grad):
            raise ValueError("stage anchor and reference must be constants, not require gradients")
        if anchor.ndim!=4 or anchor.shape[-1]!=2 or anchor.shape[0]<1 or min(anchor.shape[1:3])<2:
            raise ValueError("anchor must have shape(B,R>=2,C>=2,2)")
        if anchor.dtype not in (torch.float32,torch.float64):
            raise ValueError("float32/float64 anchor required")
        self._anchor=anchor.clone()
        if reference is None:
            rows,columns=anchor.shape[1:3]
            y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64,device=anchor.device),
                torch.linspace(0,1,columns,dtype=torch.float64,device=anchor.device),indexing="ij")
            reference=torch.stack((x,y),-1)[None]
        if reference.ndim!=4 or reference.shape[1:]!=anchor.shape[1:] or reference.shape[0] not in (1,anchor.shape[0]):
            raise ValueError("reference must have matching grid and batch1 or B")
        self._reference=reference.to(device=anchor.device,dtype=torch.float64).clone()
        self._layer=_BoundLayer(self._anchor,self._reference,direction=direction,mode=mode,
            boundary="fixed",minimum_jacobian=minimum_jacobian,theta=theta)

    def __call__(self,proposal,*,alpha_trial=1.,validate=True):
        return self._layer(self._anchor,proposal,reference=self._reference,
            alpha_trial=alpha_trial,validate=validate)

    @property
    def resident_constant_bytes(self):
        tensors=(self._anchor,self._reference,self._layer._cached_qref,
            self._layer._cached_slack,self._layer._k1,self._layer._k2)
        storages={tensor.untyped_storage().data_ptr():tensor.untyped_storage().nbytes() for tensor in tensors}
        return sum(storages.values())
