"""Candidate-only, first-order coordinated adjoint with compact active rows.

Same vertex-table update, not a new geometry mechanism. Fixed boundary only;
reference/direction/trial constants. Higher derivatives and auxiliary diagnostic
gradients are intentionally not supported. Both current-Y and proposal gradients
are supported, including first-order cascades of these candidates.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .coordinated_update import CoordinatedQ1Update,single_direction_corner_change
from .digital_q1 import q1_corner_determinants


def _direct_stencil_vjp(local_y,local_raw,corner,adj_q,adj_delta,e):
    """Termwise q1-edge differential plus the original delta triangle stencil."""
    fetch=lambda index:local_y.gather(1,index[:,None,None].expand(-1,1,2))[:,0]
    fetch_raw=lambda index:local_raw.gather(1,index[:,None])[:,0]
    # Exact q1 expressions: (b-a,d-a),(b-a,c-b),(c-d,c-b),(c-d,d-a).
    ei=(corner//2)*2;ej=ei+1
    fi=((corner+1)//2)%2;fj=fi+2
    edge_e=fetch(ej)-fetch(ei);edge_f=fetch(fj)-fetch(fi)
    rotate=lambda vector:torch.stack((-vector[:,1],vector[:,0]),-1)
    re=rotate(edge_e)*adj_q[:,None];rf=rotate(edge_f)*adj_q[:,None]
    q_sources=torch.stack((rf,-rf,-re,re),1)
    q_ids=torch.stack((ei,ej,fi,fj),1)
    # Existing delta triples: (a,b,d),(a,b,c),(d,b,c),(a,c,d).
    p=(corner==2).long()*2;q=1+(corner==3).long()*2
    r=2+((corner==1)|(corner==2)).long()
    tri_e=fetch(q)-fetch(p);tri_f=fetch(r)-fetch(p)
    h=fetch_raw(q)-fetch_raw(p);v=fetch_raw(r)-fetch_raw(p)
    rotation_e=torch.stack((-e[1],e[0]))
    # Weight before combining: zero-share rows never form 0*(v-h overflow).
    weighted_h=adj_delta*h;weighted_v=adj_delta*v
    delta_sources=torch.stack((weighted_v-weighted_h,-weighted_v,weighted_h),1)[...,None]*rotation_e
    triangle_ids=torch.stack((p,q,r),1)
    all_ids=torch.cat((q_ids,triangle_ids),1)
    all_sources=torch.cat((q_sources,delta_sources),1)
    gy=torch.zeros_like(local_y).scatter_add_(1,all_ids[...,None].expand(-1,-1,2),all_sources)
    first=adj_delta*(e[0]*tri_f[:,1]-e[1]*tri_f[:,0])
    second=adj_delta*(tri_e[:,0]*e[1]-tri_e[:,1]*e[0])
    gu=torch.zeros_like(local_raw).scatter_add_(1,triangle_ids,torch.stack((-first-second,first,second),1))
    return gy,gu


class _Candidate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,vertices,proposal,reference,layer,trial,validate,backward_backend):
        current=vertices.double();raw=layer._mask(proposal.double())
        batch,rows,columns,_=current.shape
        if reference is None:
            y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64,device=vertices.device),
                torch.linspace(0,1,columns,dtype=torch.float64,device=vertices.device),indexing="ij")
            reference=torch.stack((x,y),-1)[None]
        else:reference=reference.to(device=vertices.device,dtype=torch.float64)
        slack,delta,qref=layer._constraints(current,raw,reference)
        if validate and not bool(torch.isfinite(current).all() and torch.isfinite(raw).all()
                and torch.isfinite(slack).all() and torch.isfinite(delta).all()
                and (qref>0).all() and (slack>0).all()):
            raise ValueError("finite inputs and strictly positive reference/current margins required")
        adverse=(-delta).clamp_min(0)
        ratios=adverse/slack  # DO NOT cancel qref before exact tie selection.
        gauge=ratios.amax(1)
        if validate and not bool(torch.isfinite(gauge).all()):
            raise ValueError("proposal gauge overflowed finite-precision safety arithmetic")
        if layer.mode=="radial":
            scale=1/(1+gauge)
            potential=raw.flatten(1).ne(0).any(1) # identically zero beta otherwise
        else:
            potential=gauge*trial>layer.theta
            denominator=torch.where(potential,gauge,torch.ones_like(gauge))
            scale=torch.where(potential,layer.theta/denominator,trial)
        e=current.new_tensor(layer.direction)
        candidate=(current+raw[...,None]*scale[:,None,None,None]*e).to(vertices.dtype)
        if validate:
            margin=layer._output_slack(candidate.double(),reference,qref).amin(1)
            if not bool(torch.isfinite(candidate).all() and (margin>0).all()):
                raise RuntimeError("rounded candidate failed strict margins; reject this proposal")
        # Zero raw / inactive analytic batches cannot contribute a gauge VJP.
        tied=(ratios==gauge[:,None]) & potential[:,None]
        counts=tied.sum(1)
        active=tied.flatten().nonzero(as_tuple=False).flatten()
        reference_rows=qref.expand(batch,-1,-1,-1).reshape(-1)
        ctx.save_for_backward(vertices,proposal,active,slack.reshape(-1)[active],
            adverse.reshape(-1)[active],reference_rows[active],delta.reshape(-1)[active]<=0,
            scale,gauge,counts,potential)
        ctx.layer=layer
        ctx.backward_backend=backward_backend
        return candidate

    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        vertices,proposal,active,slack,adverse,qref,clamp_active,scale,gauge,counts,potential=ctx.saved_tensors
        layer=ctx.layer
        current=vertices.double();raw=layer._mask(proposal.double())
        batch,rows,columns,_=current.shape
        e=current.new_tensor(layer.direction);weight=(upstream.double()*e).sum(-1)
        beta=(weight*raw).flatten(1).sum(1)
        if layer.mode=="radial":
            gauge_adjoint=-beta*scale.square()
        else:
            needed=potential & (beta!=0)
            denominator=torch.where(needed,gauge,torch.ones_like(gauge))
            # Reverse quotient chain, NOT beta*(-theta/g**2): the latter can
            # overflow an intermediate derivative even when the VJP is finite.
            gauge_adjoint=torch.where(needed,-(beta*scale)/denominator,torch.zeros_like(beta))
        gradient_y=upstream.double().clone().reshape(-1,2)
        gradient_raw=(scale[:,None,None]*weight).reshape(-1)
        current_table=current.reshape(-1,2);raw_table=raw.reshape(-1)
        per_batch=4*(rows-1)*(columns-1)
        # Process ALL exact ties, in bounded local-stencil chunks. No new row or
        # selected subset is added to the amax subgradient's denominator.
        for start in range(0,active.numel(),4096):
            stop=min(start+4096,active.numel());flat=active[start:stop]
            owner=flat//per_batch
            if ctx.backward_backend=="autograd":
                keep=gauge_adjoint[owner]!=0
                if not bool(keep.any()):continue # includes beta=0, no 0*huge path
                flat=flat[keep];owner=owner[keep]
            else:
                # No per-chunk host synchronization or dynamic boolean slicing.
                # Stable normalized arithmetic safely leaves zero-share rows zero.
                keep=slice(None)
            local=(flat%per_batch)//4;corner=flat%4
            row=local//(columns-1);column=local%(columns-1)
            a=owner*(rows*columns)+row*columns+column
            ids=torch.stack((a,a+1,a+columns,a+columns+1),1) # row-major a,b,d,c
            share=gauge_adjoint[owner]/counts[owner]
            ss=slack[start:stop][keep];aa=adverse[start:stop][keep];qr=qref[start:stop][keep]
            # Preserve the normalized reverse chain, including clamp at zero.
            adj_delta=-(share/ss)*clamp_active[start:stop][keep]/qr
            adj_q=-(share*(aa/ss))/ss/qr
            if ctx.backward_backend=="torch_manual":
                gy,gu=_direct_stencil_vjp(current_table[ids],raw_table[ids],corner,adj_q,adj_delta,e)
            else:
                with torch.enable_grad():
                    local_y=current_table[ids].detach().requires_grad_()
                    local_raw=raw_table[ids].detach().requires_grad_()
                    cells=local_y.reshape(-1,2,2,2);amplitudes=local_raw.reshape(-1,2,2)
                    q=q1_corner_determinants(cells).reshape(-1,4).gather(1,corner[:,None])[:,0]
                    change=single_direction_corner_change(cells,amplitudes,layer.direction).reshape(-1,4).gather(1,corner[:,None])[:,0]
                    gy,gu=torch.autograd.grad((q,change),(local_y,local_raw),grad_outputs=(adj_q,adj_delta))
            gradient_y.index_add_(0,ids.flatten(),gy.reshape(-1,2))
            gradient_raw.index_add_(0,ids.flatten(),gu.reshape(-1))
        gradient_y=gradient_y.reshape_as(vertices).to(vertices.dtype)
        gradient_z=layer._mask(gradient_raw.reshape_as(proposal)).to(proposal.dtype)
        return gradient_y,gradient_z,None,None,None,None,None


def explicit_coordinated_candidate(vertices: torch.Tensor,proposal: torch.Tensor,*,
        reference: torch.Tensor|None=None,direction=(1.,0.),mode="radial",boundary="fixed",
        minimum_jacobian=.001,theta=.95,alpha_trial=1.,validate=True,backward_backend="autograd") -> torch.Tensor:
    """Candidate Tensor only. Default input/rounded-margin checks match the layer.

    validate=False is for trusted inner trials; callers must retain a separate
    actual-coordinate margin check and accepted/exported-map audit. This optional
    first-order API is not a drop-in replacement for differentiable diagnostics.
    It does not silently detach trainable reference or trial tensors.
    backward_backend="autograd" preserves the initial local-stencil adjoint;
    "torch_manual" uses direct selected-edge/triangle derivatives without a
    local reverse-mode graph. Both work on CPU/CUDA without optional compilers.
    """
    if boundary!="fixed":raise ValueError("candidate-only adjoint supports fixed boundary only")
    if backward_backend not in ("autograd","torch_manual"):
        raise ValueError("backward_backend must be autograd or torch_manual")
    if reference is not None and reference.requires_grad:
        raise ValueError("reference must be constant, not require gradients")
    if isinstance(alpha_trial,torch.Tensor) and alpha_trial.requires_grad:
        raise ValueError("alpha_trial must be constant, not require gradients")
    if vertices.ndim!=4 or vertices.shape[-1]!=2 or vertices.shape[0]<1 or min(vertices.shape[1:3])<2 or proposal.shape!=vertices.shape[:-1]:
        raise ValueError("vertices(B,R>=2,C>=2,2) and proposal(B,R,C) required")
    if vertices.dtype not in (torch.float32,torch.float64) or vertices.dtype!=proposal.dtype or vertices.device!=proposal.device:
        raise ValueError("matching float32/float64 geometry/proposal dtype and device required")
    if reference is not None and (reference.shape[1:]!=vertices.shape[1:] or reference.shape[0] not in (1,vertices.shape[0])):
        raise ValueError("reference must have matching grid and batch1 orB")
    trial=torch.as_tensor(alpha_trial,device=vertices.device,dtype=torch.float64)
    if trial.ndim>1 or (trial.ndim==1 and trial.shape!=(vertices.shape[0],)) or not bool(torch.isfinite(trial).all() and (trial>=0).all()):
        raise ValueError("trial must be finite nonnegative constant scalar or shape(B,)")
    layer=CoordinatedQ1Update(direction,mode=mode,boundary=boundary,minimum_jacobian=minimum_jacobian,theta=theta)
    return _Candidate.apply(vertices,proposal,reference,layer,trial,validate,backward_backend)
