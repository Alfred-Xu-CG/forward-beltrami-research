"""Fixed finite-displacement descriptor proposal, NOT a safe exported map.

Instance-only enumeration has no argmin VJP. The caller supplies immutable
original128 descriptor fields/mask and the frozen positive post-affine; this
module cannot prove how those inputs were produced or anatomical correctness.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DiscreteCaptureResult:
    proposal: torch.Tensor
    chosen_labels: torch.Tensor
    chosen_cost: torch.Tensor
    patch_denominator: torch.Tensor
    empty_patch: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def discrete_capture_proposal(fixed_feature: torch.Tensor, moving_feature: torch.Tensor,
                             mask: torch.Tensor, matrix: torch.Tensor, offset: torch.Tensor,
                             *, geometry_dtype: torch.dtype=torch.float64) -> DiscreteCaptureResult:
    """Enumerate81 ORIGINAL-moving-pixel labels at fixed33 identity-grid nodes.

    Features are(1,C,128,128), mask(1,1,128,128), A(2,2), b(2,), all on one
    device. Features/mask share f32/f64 dtype; affine may use either precision.
    Query geometry/OOB/returned proposal use geometry_dtype. Raster grids are
    cast to feature dtype, exactly align_corners=False, bilinear ZERO padding.
    The nine fixed patch-mask samples are reused for EVERY label: no moving
    overlap dropping. Empty patches cost0. Zero label is first, then fixed(x,y)
    lexicographic ordering; first minimum wins exact ties. Boundary labels are
    forced0 and their reported costs are the applied ZERO-label costs, not an
    unconstrained boundary argmin. p=A^-1(k/128), not componentwise division.
    Only a raw proposal is returned; caller must use existing safe geometry and
    complete-objective acceptance. No learned matching, map export or repair.
    """
    values=(fixed_feature,moving_feature,mask,matrix,offset)
    if not all(isinstance(value,torch.Tensor) for value in values):
        raise ValueError('tensor descriptor/mask/affine inputs required')
    if any(value.requires_grad for value in values):
        raise ValueError('immutable instance inputs only; no argmin gradient is supported')
    if geometry_dtype not in (torch.float32,torch.float64):
        raise ValueError('geometry_dtype must be float32 or float64')
    if (fixed_feature.ndim!=4 or fixed_feature.shape[0]!=1 or fixed_feature.shape[1]<1
            or fixed_feature.shape[-2:]!=(128,128) or moving_feature.shape!=fixed_feature.shape
            or mask.shape!=(1,1,128,128) or matrix.shape!=(2,2) or offset.shape!=(2,)):
        raise ValueError('fixed128 B1 descriptor/mask and2x2 affine shapes required')
    if (fixed_feature.dtype not in (torch.float32,torch.float64)
            or moving_feature.dtype!=fixed_feature.dtype or mask.dtype!=fixed_feature.dtype
            or matrix.dtype not in (torch.float32,torch.float64) or offset.dtype!=matrix.dtype
            or any(value.device!=fixed_feature.device for value in values)):
        raise ValueError('matching feature/mask dtype/device and floating affine required')
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError('finite immutable evidence required')
    if not bool(((mask>=0)&(mask<=1)).all()):
        raise ValueError('fixed mask weights must be in[0,1]')
    device=fixed_feature.device
    a,b=matrix.to(geometry_dtype),offset.to(geometry_dtype)
    det=a[0,0]*a[1,1]-a[0,1]*a[1,0]
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(det) and det>0):
        raise ValueError('finite positive affine in declared geometry precision required')
    axis=torch.arange(33,device=device,dtype=geometry_dtype)/32
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    nodes=torch.stack((x,y),-1).reshape(1089,2)
    patch=torch.tensor(list(itertools.product((-1,0,1),repeat=2)),device=device,dtype=geometry_dtype)/128
    queries=nodes[:,None]+patch[None]
    fixed_grid=(2*queries-1).to(fixed_feature.dtype)[None]
    fixed_samples=F.grid_sample(fixed_feature,fixed_grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    mask_samples=F.grid_sample(mask,fixed_grid,mode='bilinear',padding_mode='zeros',align_corners=False)[:,0]
    denominator=mask_samples.sum(-1)
    label_values=[(0,0)]+[label for label in itertools.product(range(-4,5),repeat=2) if label!=(0,0)]
    labels=torch.tensor(label_values,device=device,dtype=torch.long)
    targets=(queries@a.T+b)[None]+labels.to(geometry_dtype)[:,None,None]/128
    outside=(F.relu(-targets)+F.relu(targets-1)).square().sum(-1)
    moving_grid=(2*targets-1).to(moving_feature.dtype)
    warped=F.grid_sample(moving_feature.expand(81,-1,-1,-1),moving_grid,
                         mode='bilinear',padding_mode='zeros',align_corners=False)
    del targets,moving_grid
    # These buffers are sampler outputs owned here, never immutable inputs.
    warped.sub_(fixed_samples).abs_()
    terms=warped.mean(1).to(geometry_dtype)
    del warped
    terms.add_(outside).mul_(mask_samples)
    safe_denominator=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    costs=terms.sum(-1)/safe_denominator
    costs=torch.where(denominator>0,costs,torch.zeros_like(costs))
    if not bool(torch.isfinite(costs).all()):
        raise ValueError('nonfinite finite-label costs in declared arithmetic')
    unconstrained=costs.argmin(0)  # first entry is ZERO; exact ties retain it
    boundary=((x==0)|(x==1)|(y==0)|(y==1)).reshape(1089)
    chosen=torch.where(boundary,torch.zeros_like(unconstrained),unconstrained)
    chosen_labels=labels[chosen]
    chosen_cost=costs.gather(0,chosen[None])[0]
    displacement=chosen_labels.to(geometry_dtype)/128
    proposal=torch.linalg.solve(a,displacement.T).T.reshape(1,33,33,2)
    if not bool(torch.isfinite(proposal).all()):
        raise ValueError('nonfinite pre-affine proposal')
    diagnostics=dict(label_count=81,patch_sample_count=9,control_side=33,image_side=128,
        label_range_original_moving_px=4,boundary_nodes_forced_zero=int(boundary.sum()),
        boundary_unconstrained_nonzero_count=int(((unconstrained!=0)&boundary).sum()),
        nonzero_interior_count=int(((chosen!=0)&~boundary).sum()),empty_patch_count=int((denominator==0).sum()),
        feature_dtype=str(fixed_feature.dtype),geometry_dtype=str(geometry_dtype),
        mask_scope='original fixed patch mask, same denominator for all labels; no moving-overlap dropping',
        label_order='zero first, then(x,y)lexicographic; exact tie first minimum',
        proposal_scope='A^-1 times original-moving normalized displacement; raw proposal only, not a safe map',
        provenance_scope='caller-supplied frozen original descriptor fields; input origin/anatomy not proved here',
        argmin_vjp_supported=False)
    return DiscreteCaptureResult(proposal,chosen_labels.reshape(1,33,33,2),chosen_cost.reshape(1,33,33),
                                 denominator.reshape(1,33,33),(denominator==0).reshape(1,33,33),diagnostics)
