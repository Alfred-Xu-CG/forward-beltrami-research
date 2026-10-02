"""Image-derived finite-label search and one ALWAYS-LEGAL initial construction.

The label/continuous surrogate is nonconvex. Its screened global solves are not
the published ConvexAdam averaging operation. Proposals are never exported as
certified maps; sixteen existing coordinated steps construct the initial map.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
import time

import torch
from torch.nn import functional as F

from .coordinated_stiffness import DirichletGalerkinStiffness,dst1_orthonormal
from .coordinated_update import CoordinatedQ1Update
from .digital_q1 import q1_corner_determinants


COUPLING_SCHEDULE=(.003,.01,.03,.1,.3,1.)


@dataclass
class NodalCostVolume:
    costs:torch.Tensor
    labels:torch.Tensor
    weights:torch.Tensor
    denominator:torch.Tensor
    proposal_side:int
    image_side:int
    diagnostics:dict


@dataclass
class CoupledLabels:
    displacement:torch.Tensor  # (2,l-2,l-2), x/y shifts in image_side pixel units
    selected_labels:torch.Tensor
    diagnostics:dict


@dataclass
class CoupledSeedResult:
    vertices:torch.Tensor
    coefficients_pixels:torch.Tensor  # (1,l,l,2), exact zero perimeter
    diagnostics:dict


def _synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def finite_displacement_cost(fixed_feature,moving_feature,mask,matrix,offset,*,
                             proposal_side=65,label_radius=16,label_batch=16):
    tensors=(fixed_feature,moving_feature,mask,matrix,offset)
    if not all(isinstance(t,torch.Tensor) for t in tensors): raise ValueError('tensor inputs required')
    if any(t.requires_grad for t in tensors): raise ValueError('frozen instance-only inputs required')
    if (fixed_feature.ndim!=4 or fixed_feature.shape[0]!=1 or fixed_feature.shape[1]<1
            or fixed_feature.shape[-1]!=fixed_feature.shape[-2] or fixed_feature.shape[-1]<4
            or moving_feature.shape!=fixed_feature.shape or mask.shape!=(1,1,*fixed_feature.shape[-2:])
            or matrix.shape!=(2,2) or offset.shape!=(2,)
            or any(t.device!=fixed_feature.device for t in tensors)
            or fixed_feature.dtype not in (torch.float32,torch.float64)
            or moving_feature.dtype!=fixed_feature.dtype or mask.dtype!=fixed_feature.dtype
            or matrix.dtype not in (torch.float32,torch.float64) or offset.dtype!=matrix.dtype
            or any(not bool(torch.isfinite(t).all()) for t in tensors)
            or not bool(((mask>=0)&(mask<=1)).all())):
        raise ValueError('finite batch-one square descriptors/mask and positive affine required')
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in (proposal_side,label_radius,label_batch))
            or proposal_side<3 or label_radius<1 or label_batch<1):
        raise ValueError('valid integer proposal grid, label radius and batch required')
    device=fixed_feature.device; side=fixed_feature.shape[-1]
    a,b=matrix.double(),offset.double()
    if not bool(torch.linalg.det(a)>0): raise ValueError('positive frozen post-affine required')
    axis=torch.arange(1,proposal_side-1,device=device,dtype=torch.float64)/(proposal_side-1)
    y,x=torch.meshgrid(axis,axis,indexing='ij'); nodes=torch.stack((x,y),-1).reshape(-1,2)
    patch=torch.tensor(list(itertools.product((-1,0,1),repeat=2)),device=device,dtype=torch.float64)/side
    queries=nodes[:,None,:]+patch[None,:,:]
    grid=(2*queries-1).to(fixed_feature.dtype)[None]
    fixed=F.grid_sample(fixed_feature,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    fixed_mask=F.grid_sample(mask,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0].double()
    patch_denominator=fixed_mask.sum(-1)
    weights=patch_denominator/9.; denominator=weights.sum()
    if not bool(denominator>0): raise ValueError('positive frozen nodal support required')
    values=[(0,0)]+[p for p in itertools.product(range(-label_radius,label_radius+1),repeat=2) if p!=(0,0)]
    labels=torch.tensor(values,device=device,dtype=torch.long)
    costs=torch.empty((len(values),len(nodes)),device=device,dtype=torch.float64)
    safe_denominator=torch.where(patch_denominator>0,patch_denominator,torch.ones_like(patch_denominator))
    for start in range(0,len(values),label_batch):
        shifts=labels[start:start+label_batch].double()/side
        residual=queries[None]+shifts[:,None,None,:]
        original=residual@a.T+b
        outside=(F.relu(-original)+F.relu(original-1)).square().sum(-1)
        warped=F.grid_sample(moving_feature.expand(len(shifts),-1,-1,-1),
            (2*residual-1).to(moving_feature.dtype),mode='bilinear',padding_mode='zeros',align_corners=False)
        errors=warped.sub_(fixed).abs_().mean(1).double()
        costs[start:start+len(shifts)]=((errors+outside)*fixed_mask).sum(-1)/safe_denominator
    if not bool(torch.isfinite(costs).all()): raise ValueError('nonfinite finite-displacement cost')
    diagnostics=dict(image_side=side,proposal_side=proposal_side,interior_nodes=len(nodes),
        labels=len(values),label_radius_pixels=label_radius,label_batch=label_batch,
        label_step_unit=1./side,patch_samples=9,empty_patches=int((weights==0).sum()),
        fixed_denominator=float(denominator),cost_volume_bytes=costs.numel()*costs.element_size(),
        feature_distance='eight-channel L1 in production; channel mean, not squared distance',
        query_scope='identity residual vertex nodes plus nine raster offsets; nodal surrogate, not original raster energy',
        mask_scope='immutable fixed patch samples/denominators; no moving overlap gating',
        oob_scope='quadratic excess of A*(q+offset+label/image_side)+b with coefficient1',
        label_order='zero first then(x,y)lexicographic; exact ties first minimum')
    return NodalCostVolume(costs,labels,weights,denominator,proposal_side,side,diagnostics)


def _dst2(value): return dst1_orthonormal(dst1_orthonormal(value,-1),-2)


@torch.no_grad()
def couple_cost_volume(volume:NodalCostVolume,*,fine_side=257,strain_weight=3.,
                       schedule=COUPLING_SCHEDULE,alternations=2):
    if (not schedule or any(not math.isfinite(c) or c<=0 for c in schedule)
            or isinstance(alternations,bool) or not isinstance(alternations,int) or alternations<1
            or not math.isfinite(strain_weight) or strain_weight<=0):
        raise ValueError('positive coupling schedule, integer alternations and positive strain weight required')
    costs,labels,weights=volume.costs,volume.labels,volume.weights
    n=volume.proposal_side-2; device=costs.device
    if costs.shape!=(len(labels),n*n) or labels.shape[1:]!=(2,) or weights.shape!=(n*n,):
        raise ValueError('cost/label/node dimensions differ')
    metric=DirichletGalerkinStiffness(fine_side,volume.proposal_side,weight=1.,device=device,dtype=torch.float64)
    label_values=labels.double(); weighted=costs*weights[None]
    z_index=weighted.argmin(0)
    z=label_values[z_index].T.reshape(2,n,n)
    def screened(value,c):
        denominator=2*c+(strain_weight*volume.denominator/volume.image_side**2)*metric.eigenvalues
        return _dst2((2*c)*_dst2(value)/denominator)
    def energy(index,value,c):
        discrete=label_values[index].T.reshape(2,n,n)
        data=weighted.gather(0,index[None])[0].sum()/volume.denominator
        coupling=c*(discrete-value).square().sum()/volume.denominator
        prior=(strain_weight/(2*volume.image_side**2))*(value*metric.apply(value)).sum()
        return dict(data=float(data),coupling=float(coupling),prior=float(prior),total=float(data+coupling+prior))
    u=screened(z,float(schedule[0])); blocks=[]; solves=1; minimizations=1
    for c in schedule:
        c=float(c)
        for iteration in range(alternations):
            before=energy(z_index,u,c)
            best=torch.full((n*n,),math.inf,device=device,dtype=torch.float64)
            chosen=torch.zeros(n*n,device=device,dtype=torch.long)
            soft=u.reshape(2,-1).T
            # Strict '<' preserves the declared first-minimum label order.
            for start in range(0,len(labels),16):
                candidates=weighted[start:start+16]+c*(label_values[start:start+16,None,:]-soft[None]).square().sum(-1)
                local,index=candidates.min(0); improve=local<best
                chosen=torch.where(improve,index+start,chosen); best=torch.minimum(best,local)
            z_index=chosen; minimizations+=1
            after_labels=energy(z_index,u,c)
            z=label_values[z_index].T.reshape(2,n,n)
            u=screened(z,c); solves+=1
            after_continuous=energy(z_index,u,c)
            blocks.append(dict(c=c,iteration=iteration,before=before,
                after_labels=after_labels,after_continuous=after_continuous))
    if not bool(torch.isfinite(u).all()): raise ValueError('nonfinite screened displacement')
    return CoupledLabels(u,z_index.reshape(n,n),dict(schedule=list(schedule),alternations_per_value=alternations,
        two_component_screened_solves=solves,scalar_screened_systems=2*solves,label_minimizations=minimizations,
        blocks=blocks,coupling_scope='UNWEIGHTED all interior nodes, divided by fixed Z',
        screened_equation='(2c I + strain_weight*Z/image_side^2 G)u=2c z; G=P.T Kfine P',
        global_optimum_claim=False,contrast_to_ConvexAdam='same six schedule values, exact screened Galerkin solve instead of local averaging'))


@torch.no_grad()
def construct_safe_seed(reference,displacement_pixels,*,image_side=128,cycles=8,minimum_jacobian=.001):
    if (reference.ndim!=4 or reference.shape[0]!=1 or reference.shape[-1]!=2
            or reference.shape[1]!=reference.shape[2] or reference.dtype!=torch.float64
            or displacement_pixels.ndim!=3 or displacement_pixels.shape[0]!=2
            or displacement_pixels.shape[1]!=displacement_pixels.shape[2]
            or displacement_pixels.dtype!=torch.float64 or displacement_pixels.device!=reference.device
            or not bool(torch.isfinite(reference).all() and torch.isfinite(displacement_pixels).all())
            or isinstance(cycles,bool) or not isinstance(cycles,int) or cycles<1
            or isinstance(image_side,bool) or not isinstance(image_side,int) or image_side<2):
        raise ValueError('finite batch-one square double reference and two scalar pixel-coefficient fields required')
    side=reference.shape[1]
    axis=torch.arange(side,device=reference.device,dtype=torch.float64)/(side-1)
    y,x=torch.meshgrid(axis,axis,indexing='ij'); identity=torch.stack((x,y),-1)[None]
    if not torch.equal(reference,identity): raise ValueError('construct from exact unit identity only')
    coefficients=F.pad(displacement_pixels,(1,1,1,1))[None]
    raw=F.interpolate(coefficients,size=(side,side),mode='bilinear',align_corners=True).permute(0,2,3,1)/image_side
    target=reference+raw
    qref=q1_corner_determinants(reference)
    raw_q=q1_corner_determinants(target)/qref
    current=reference.clone(); steps=[]
    operators=[CoordinatedQ1Update(direction,mode='analytic',minimum_jacobian=minimum_jacobian,theta=.95)
               for direction in ((1.,0.),(0.,1.))]
    for cycle in range(cycles):
        for component,operator in enumerate(operators):
            proposal=target[...,component]-current[...,component]
            result=operator(current,proposal,reference=reference,validate=True)
            current=result.vertices
            actual_q=q1_corner_determinants(current)/qref
            if not bool(torch.isfinite(current).all() and (actual_q>minimum_jacobian).all()):
                raise RuntimeError('rounded construction step violated actual strict floor')
            if not (torch.equal(current[:,0],reference[:,0]) and torch.equal(current[:,-1],reference[:,-1])
                    and torch.equal(current[:,:,0],reference[:,:,0]) and torch.equal(current[:,:,-1],reference[:,:,-1])):
                raise RuntimeError('construction changed boundary')
            steps.append(dict(cycle=cycle,component=component,scale=float(result.scale.min()),
                gauge=float(result.gauge.max()),actual_minimum_corner_ratio=float(actual_q.min()),
                remaining_target_rms=float((current-target).square().sum(-1).mean().sqrt())))
    decoded=current-reference
    norm=lambda field:dict(rms=float(field.square().sum(-1).mean().sqrt()),maximum=float(torch.linalg.vector_norm(field,dim=-1).max()))
    return CoupledSeedResult(current,coefficients.permute(0,2,3,1),dict(
        coordinate_steps=len(steps),cycles=cycles,steps=steps,
        raw_target_nonpositive_corner_count=int((raw_q<=0).sum()),raw_target_minimum_corner_ratio=float(raw_q.min()),
        raw_displacement_unit=norm(raw),decoded_displacement_unit=norm(decoded),
        raw_to_decoded_distance_unit=norm(current-target),actual_minimum_corner_ratio=float(actual_q.min()),
        construction_scope='same frozen image proposal tracked by16 existing safe scalar steps; no raw target map accepted/repaired',
        energy_acceptance_during_construction=False))


@torch.no_grad()
def build_coupled_mind_seed(fixed_feature,moving_feature,mask,matrix,offset,reference):
    """One fixed production recipe; no configurable radius/weight/seed sweep."""
    if fixed_feature.shape[-2:]!=(128,128) or fixed_feature.shape[1]!=8 or reference.shape!=(1,257,257,2):
        raise ValueError('fixed production128/eight-channel evidence and257 actual map required')
    _synchronize(reference.device); tick=time.perf_counter()
    volume=finite_displacement_cost(fixed_feature,moving_feature,mask,matrix,offset)
    _synchronize(reference.device); cost_seconds=time.perf_counter()-tick; tick=time.perf_counter()
    coupled=couple_cost_volume(volume)
    _synchronize(reference.device); coupling_seconds=time.perf_counter()-tick; tick=time.perf_counter()
    seed=construct_safe_seed(reference,coupled.displacement)
    _synchronize(reference.device); construction_seconds=time.perf_counter()-tick
    seed.diagnostics.update(cost=volume.diagnostics,coupling=coupled.diagnostics,
        cost_volume_seconds=cost_seconds,coupling_seconds=coupling_seconds,construction_seconds=construction_seconds,
        initial_center='identity residual after original frozen affine; no existing deformation field',
        annotations_read=False,point_matches_used_in_seed=False,
        original_seed_energy_may_increase=True)
    return seed
