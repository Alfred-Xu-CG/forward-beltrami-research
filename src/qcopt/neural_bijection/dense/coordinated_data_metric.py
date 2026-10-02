"""Timed final-axis search with an SPD metric; the production objective is unchanged.

Instance optimization only. Sampling slopes are production AD slopes at rounded
float32 grids, not derivatives of floating-point quantization. No second image
derivative, objective smoothing, map repair or optimizer-trajectory AD is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import torch
from torch.nn import functional as F

from .coordinated_fixed_sampling import FrozenP1Evaluator
from .coordinated_sampling import p1_map_at_pixel_centers, p1_map_at_queries
from .q1_image_sampling import fixed_pixel_centers
from .coordinated_stiffness import DirichletGalerkinStiffness, dst1_orthonormal
from .coordinated_update import single_direction_corner_change
from .coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from .digital_q1 import q1_corner_determinants


class FrozenScalarP1Rows:
    """Fixed fine-P1 scalar rows restricted to interior unknowns, with exact transpose."""
    def __init__(self,side,queries):
        if side<3 or queries.dtype!=torch.float64:
            raise ValueError('float64 fixed queries and at least3 vertices required')
        evaluator=FrozenP1Evaluator(side,side,queries,'ac')
        ids=evaluator.vertex_ids
        row,column=ids//side,ids%side
        interior=(row>0)&(row<side-1)&(column>0)&(column<side-1)
        self.indices=((row-1)*(side-2)+column-1).clamp(0,(side-2)**2-1)
        self.weights=evaluator.weights*interior.to(torch.float64)
        self.side=side;self.interior_side=side-2
        # Three different source vertices; discarded boundary entries have zero weight.
        self.row_norm_squared=self.weights.square().sum(-1)

    def apply(self,coefficients):
        if (coefficients.shape!=(1,self.interior_side,self.interior_side)
                or coefficients.dtype!=self.weights.dtype or coefficients.device!=self.weights.device):
            raise ValueError('one scalar interior coefficient grid required')
        return (coefficients.reshape(-1)[self.indices]*self.weights).sum(-1)

    def transpose(self,values):
        if (values.shape!=self.row_norm_squared.shape or values.dtype!=self.weights.dtype
                or values.device!=self.weights.device):
            raise ValueError('one cotangent per query required')
        output=self.weights.new_zeros(self.interior_side**2)
        output.scatter_add_(0,self.indices.reshape(-1),(values[:,None]*self.weights).reshape(-1))
        return output.reshape(1,self.interior_side,self.interior_side)


class MetricGeometry:
    """Only fixed source-query row geometry; no current-map/metric weights cached."""
    def __init__(self,evidence,side):
        queries=fixed_pixel_centers(*evidence.fixed.shape[-2:],dtype=torch.float64,device=evidence.fixed.device)
        self.image_rows=FrozenScalarP1Rows(side,queries)
        self.point_rows=(FrozenScalarP1Rows(side,evidence.matches.source) if evidence.match_weight else None)


class DataAwareMetric:
    """One frozen H=3K+T' diag(DI+DO) T+L' diag(beta) L, final equal-sized grid."""
    def __init__(self,evidence,vertices,direction,*,geometry=None,work_counts=None):
        if (vertices.dtype!=torch.float64 or vertices.ndim!=4 or vertices.shape[0]!=1
                or vertices.shape[1]!=vertices.shape[2] or vertices.shape[-1]!=2
                or evidence.loss!='mind' or evidence.mind_order!='transport'
                or evidence.mind_frame!='shared_affine' or evidence.interpolation!='p1_ac'
                or evidence.fixed_feature.shape[1]!=8 or evidence.moving_feature.dtype!=torch.float32
                or evidence.strain_weight!=3. or evidence.oob_weight!=1. or evidence.image_weight!=1.):
            raise ValueError('declared float64 P1ac shared float32 eight-channel MIND/ARAP3/OOB1 required')
        side=vertices.shape[1]
        e=vertices.new_tensor(direction)
        if tuple(direction) not in ((1.,0.),(0.,1.)):
            raise ValueError('one final coordinate axis required')
        geometry=geometry or MetricGeometry(evidence,side)
        self.image_rows,self.point_rows=geometry.image_rows,geometry.point_rows
        self.stiffness=DirichletGalerkinStiffness(side,side,weight=3.,device=vertices.device,dtype=vertices.dtype)
        with torch.no_grad():
            query=(evidence.fixed_p1_evaluator(vertices) if evidence.fixed_p1_evaluator is not None
                   else p1_map_at_pixel_centers(vertices,*evidence.fixed.shape[-2:],'ac'))
        query=query.detach().double().requires_grad_(True)
        warped=F.grid_sample(evidence.moving_feature,(2*query-1).to(evidence.moving_feature.dtype),
            mode='bilinear',padding_mode='zeros',align_corners=False)
        if work_counts is not None:work_counts['descriptor_forwards']+=1
        denominator=evidence.denominator.detach().double()
        mask=evidence.mask.detach().double().reshape(-1)
        residual=(evidence.fixed_feature-warped).detach().abs().double().clamp_min(1e-3)
        diagonal=query.new_zeros(query.shape[:-1])
        for channel in range(8):
            slope=torch.autograd.grad(warped[:,channel].sum(),query,retain_graph=channel<7)[0]
            if work_counts is not None:work_counts['descriptor_vjps']+=1
            j=(slope*e).sum(-1)
            diagonal+=j.square()/residual[:,channel]
        self.query=query.detach()
        self.image_diagonal=(diagonal.reshape(-1)*mask/(8*denominator)).detach()
        with torch.no_grad():
            world=self.query@evidence.matrix.T+evidence.offset
            ae=evidence.matrix@e
            self.oob_diagonal=(2*mask/denominator)*(((world<0)|(world>1)).to(torch.float64)
                *ae.square()).sum(-1).reshape(-1)
            self.point_diagonal=None
            if self.point_rows is not None:
                matches=evidence.matches
                mapped=(matches.fixed_p1_evaluator(vertices) if matches.fixed_p1_evaluator is not None
                        else p1_map_at_queries(vertices,matches.source,'ac',validate_queries=False))
                scale=matches.pixel_scale/matches.robust_scale
                z=((mapped-matches.target)@evidence.matrix.T*scale)[0]
                a=scale*ae
                cross=a[0]*z[:,1]-a[1]*z[:,0]
                self.point_diagonal=(evidence.match_weight*matches.weights*(a.square().sum()+cross.square())
                    /(1+z.square().sum(-1)).pow(1.5))
            self.raster_diagonal=self.image_diagonal+self.oob_diagonal
            trace=(self.raster_diagonal*self.image_rows.row_norm_squared).sum()
            if self.point_rows is not None:trace+=(self.point_diagonal*self.point_rows.row_norm_squared).sum()
            self.gamma=float(trace/((side-2)**2))
        weights=[self.raster_diagonal]+([] if self.point_diagonal is None else [self.point_diagonal])
        if (not math.isfinite(self.gamma) or self.gamma<0
                or any(not bool(torch.isfinite(w).all() and (w>=0).all()) for w in weights)):
            raise ValueError('nonfinite or negative frozen metric weights')

    def apply(self,value):
        self.stiffness._check(value)
        # This experiment is final equal-grid only: its exact K is five-point.
        # Keep FFTs only in the shifted inverse, not every H matrix-vector call.
        padded=F.pad(value,(1,1,1,1))
        stiffness=3*(4*value-padded[:,:-2,1:-1]-padded[:,2:,1:-1]
                     -padded[:,1:-1,:-2]-padded[:,1:-1,2:])
        result=stiffness+self.image_rows.transpose(self.raster_diagonal*self.image_rows.apply(value))
        if self.point_rows is not None:
            result+=self.point_rows.transpose(self.point_diagonal*self.point_rows.apply(value))
        return result

    def precondition(self,value):
        self.stiffness._check(value)
        transform=dst1_orthonormal(dst1_orthonormal(value,-1),-2)
        return dst1_orthonormal(dst1_orthonormal(transform/(self.stiffness.eigenvalues+self.gamma),-1),-2)


@torch.no_grad()
def pcg(operator,rhs,precondition,*,rtol=.1,max_steps=20):
    """Zero-start bounded PCG; a capped/nonfinite solve is never called converged."""
    if not 0<rtol<1 or max_steps<1:raise ValueError('positive PCG budget and relative tolerance required')
    x=torch.zeros_like(rhs);r=rhs.clone();norm0=float(torch.linalg.vector_norm(r))
    record=dict(iterations=0,matvecs=0,preconditioner_calls=0,converged=False,relative_residual=None,
        residual_kind='recursive Euclidean norm divided by initial norm',reason='nonfinite_rhs')
    if not math.isfinite(norm0):return x,record
    if norm0==0:return x,{**record,'converged':True,'relative_residual':0.,'reason':'zero_rhs'}
    z=precondition(r);record['preconditioner_calls']+=1;p=z.clone();rz=float((r*z).sum())
    if not math.isfinite(rz) or rz<=0:return x,{**record,'reason':'nonpositive_preconditioner'}
    for iteration in range(max_steps):
        hp=operator(p);record['matvecs']+=1;denominator=float((p*hp).sum())
        if not math.isfinite(denominator) or denominator<=0:
            record['reason']='nonpositive_operator';break
        alpha=rz/denominator;x+=alpha*p;r-=alpha*hp
        record['iterations']=iteration+1
        relative=float(torch.linalg.vector_norm(r))/norm0;record['relative_residual']=relative if math.isfinite(relative) else None
        if not math.isfinite(relative) or not bool(torch.isfinite(x).all()):
            record['reason']='nonfinite_iterate';break
        if relative<=rtol:
            record.update(converged=True,reason='relative_residual');break
        if iteration==max_steps-1:record['reason']='iteration_cap';break
        z=precondition(r);record['preconditioner_calls']+=1;next_rz=float((r*z).sum())
        if not math.isfinite(next_rz) or next_rz<=0:
            record['reason']='nonpositive_preconditioner';break
        p=z+(next_rz/rz)*p;rz=next_rz
    return x,record


@dataclass
class TimedFiberResult:
    vertices:torch.Tensor
    initial_objective:float
    final_objective:float
    trace:list[dict]
    counts:dict[str,int]
    stop_reason:str
    elapsed_seconds:float
    time_overrun_seconds:float
    minimum_contracted_slack:float


def _sync(value):
    if value.device.type=='cuda':torch.cuda.synchronize(value.device)


def _setup(anchor,direction,seconds,minimum_jacobian):
    if (anchor.dtype!=torch.float64 or anchor.ndim!=4 or anchor.shape[0]!=1
            or anchor.shape[1]!=anchor.shape[2] or anchor.shape[-1]!=2
            or not bool(torch.isfinite(anchor).all()) or not math.isfinite(seconds) or seconds<=0
            or tuple(direction) not in ((1.,0.),(0.,1.)) or not 0<minimum_jacobian<1):
        raise ValueError('finite square batch-one float64 geometry, one axis and positive time required')
    base=anchor.detach().clone();e=base.new_tensor(direction);normalize=float((base.shape[1]-1)**2)
    q0=q1_corner_determinants(base)*normalize
    if not bool((q0>minimum_jacobian).all()):raise ValueError('strict positive incoming floor required')
    floor=minimum_jacobian+.05*(q0-minimum_jacobian)
    return base,e,normalize,floor


def _counts():
    return dict(gradient_steps=0,accepted_steps=0,objective_evaluations=0,trial_evaluations=0,
        failed_trials=0,backtracks=0,metric_refreshes=0,descriptor_forwards=0,descriptor_vjps=0,
        pcg_iterations=0,pcg_matvecs=0,preconditioner_calls=0)


def _finish(current,initial,final,trace,counts,stop,start,seconds,clock,base,floor,normalize,*,original_adam_floor=None):
    current=current.detach()
    slack=q1_corner_determinants(current)*normalize-floor
    valid=((slack>0).all() if original_adam_floor is None
           else (q1_corner_determinants(current)*normalize>original_adam_floor).all())
    if not bool(torch.isfinite(current).all() and valid):
        raise RuntimeError('retained timed-stage map violates its actual required floor')
    if not all(torch.equal(a,b) for a,b in ((current[:,0],base[:,0]),(current[:,-1],base[:,-1]),
            (current[:,:,0],base[:,:,0]),(current[:,:,-1],base[:,:,-1]))):
        raise RuntimeError('retained timed-stage boundary changed')
    _sync(current);elapsed=clock()-start
    return TimedFiberResult(current,float(initial),float(final),trace,counts,stop,elapsed,
        max(0.,elapsed-seconds),float(slack.min()))


def optimize_data_metric_fiber(anchor,objective,*,direction=(1.,0.),seconds=2.,minimum_jacobian=.001,clock=time.perf_counter):
    """Physical scalar fiber, refreshed SPD metric and unchanged complete-E Armijo."""
    _sync(anchor);start=clock()
    base,e,normalize,floor=_setup(anchor,direction,seconds,minimum_jacobian)
    geometry=MetricGeometry(objective,base.shape[1]);c=base.new_zeros((1,base.shape[1]-2,base.shape[2]-2))
    vertices=lambda value:base+F.pad(value,(1,1,1,1))[...,None]*e
    counts=_counts();trace=[];current=base
    with torch.no_grad():initial=float(objective(base)[0]);counts['objective_evaluations']+=1
    if not math.isfinite(initial):raise ValueError('nonfinite incoming complete objective')
    final=initial;stop='time_budget'
    while True:
        _sync(base)
        if clock()-start>=seconds:break
        variable=c.detach().requires_grad_(True);current=vertices(variable)
        total,parts=objective(current);counts['objective_evaluations']+=1
        value=float(total.detach())
        if not math.isfinite(value):stop='nonfinite_current_objective';counts['failed_trials']+=1;break
        gradient=torch.autograd.grad(total,variable)[0];counts['gradient_steps']+=1
        if not bool(torch.isfinite(gradient).all()):stop='nonfinite_gradient';counts['failed_trials']+=1;break
        if not bool((gradient!=0).any()):stop='zero_computed_gradient';break
        try:
            counts['metric_refreshes']+=1
            metric=DataAwareMetric(objective,current.detach(),direction,geometry=geometry,work_counts=counts)
        except (ValueError,RuntimeError) as error:
            stop='metric_failure';counts['failed_trials']+=1
            trace.append(dict(step=counts['gradient_steps']-1,error=f'{type(error).__name__}: {error}',accepted=False));break
        descent,solve=pcg(metric.apply,-gradient,metric.precondition,rtol=.1,max_steps=20)
        counts['pcg_iterations']+=solve['iterations'];counts['pcg_matvecs']+=solve['matvecs']
        counts['preconditioner_calls']+=solve['preconditioner_calls']
        slope=float((gradient*descent).sum())
        detail=dict(step=counts['gradient_steps']-1,total=value,**{k:float(v.detach()) for k,v in parts.items()},
            pcg=solve,metric_gamma=metric.gamma,directional_derivative=slope if math.isfinite(slope) else None,trials=[])
        if not bool(torch.isfinite(descent).all()) or not math.isfinite(slope) or slope>=0:
            stop='nonfinite_or_nondescent_direction';counts['failed_trials']+=1
            detail['accepted']=False;trace.append(detail);break
        with torch.no_grad():
            current=vertices(c);q=q1_corner_determinants(current)*normalize;slack=q-floor
            if not bool((slack>0).all()):raise RuntimeError('current accepted contracted floor invalid')
            change=single_direction_corner_change(base,F.pad(descent,(1,1,1,1)),e)*normalize
            declining=change<0
            amax=float((slack[declining]/(-change[declining])).min()) if bool(declining.any()) else math.inf
            alpha=min(1.,.99*amax);detail.update(alpha_max=amax if math.isfinite(amax) else None,initial_alpha=alpha)
            if not math.isfinite(alpha) or alpha<=0:
                stop='numerical_stagnation';detail['accepted']=False;trace.append(detail);break
            accepted=False
            for backtrack in range(13):
                next_c=c+alpha*descent;candidate=vertices(next_c)
                trial_q=q1_corner_determinants(candidate)*normalize
                valid=bool(torch.isfinite(candidate).all() and torch.isfinite(trial_q).all() and (trial_q>floor).all())
                candidate_value=None;rhs=value+1e-4*alpha*slope
                if valid:
                    candidate_value=float(objective(candidate)[0]);counts['objective_evaluations']+=1;counts['trial_evaluations']+=1
                    valid=math.isfinite(candidate_value)
                if not valid:counts['failed_trials']+=1
                success=valid and candidate_value<=rhs and candidate_value<value
                detail['trials'].append(dict(alpha=alpha,total=candidate_value if candidate_value is None or math.isfinite(candidate_value) else None,
                    armijo_rhs=rhs,geometry_valid=bool((trial_q>floor).all()),accepted=success))
                if success:
                    c=next_c.detach();current=candidate.detach();final=candidate_value
                    counts['accepted_steps']+=1;accepted=True;break
                if backtrack<12:alpha*=.5;counts['backtracks']+=1
            detail['accepted']=accepted;trace.append(detail)
            if not accepted:stop='line_search_exhausted';break
    current=vertices(c)
    return _finish(current,initial,final,trace,counts,stop,start,seconds,clock,base,floor,normalize)


def optimize_timed_adam_fiber(anchor,objective,*,learning_rate,direction=(1.,0.),seconds=2.,minimum_jacobian=.001,clock=time.perf_counter):
    """Existing final analytic latent Adam and best-trial selector, with a wall clock."""
    _sync(anchor);start=clock()
    base,e,normalize,floor=_setup(anchor,direction,seconds,minimum_jacobian)
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('positive original final Adam rate required')
    layer=FrozenAnchorCoordinatedUpdate(base,direction=direction,mode='analytic',minimum_jacobian=minimum_jacobian,theta=.95)
    coefficients=torch.nn.Parameter(base.new_zeros((1,base.shape[1]-2,base.shape[2]-2)))
    optimizer=torch.optim.Adam([coefficients],lr=learning_rate)
    counts=_counts();trace=[]
    with torch.no_grad():initial=float(objective(base)[0]);counts['objective_evaluations']+=1
    if not math.isfinite(initial):raise ValueError('nonfinite incoming complete objective')
    best_map=base;best=initial;stop='time_budget'
    result=layer(F.pad(coefficients,(1,1,1,1)),validate=False)
    total,parts=objective(result.vertices);counts['objective_evaluations']+=1
    while True:
        _sync(base)
        if clock()-start>=seconds:break
        value=float(total.detach())
        if not math.isfinite(value):stop='nonfinite_current_objective';counts['failed_trials']+=1;break
        optimizer.zero_grad(set_to_none=True);total.backward();counts['gradient_steps']+=1
        if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
            stop='nonfinite_gradient';counts['failed_trials']+=1;break
        optimizer.step()
        try:
            result=layer(F.pad(coefficients,(1,1,1,1)),validate=False)
        except (ValueError,RuntimeError) as error:
            stop='analytic_trial_failure';counts['failed_trials']+=1
            trace.append(dict(step=counts['gradient_steps']-1,error=f'{type(error).__name__}: {error}',accepted=False));break
        total,parts=objective(result.vertices);counts['objective_evaluations']+=1;counts['trial_evaluations']+=1
        value=float(total.detach())
        q=q1_corner_determinants(result.vertices.detach())*normalize
        # Preserve old Adam's actual Q>eta acceptance. Its analytic contraction
        # can reach Qfloor exactly; imposing a new strict contraction rejects
        # normal bound-active control trials and changes the control algorithm.
        legal=bool(torch.isfinite(result.vertices).all() and torch.isfinite(q).all() and (q>minimum_jacobian).all())
        if not math.isfinite(value) or not legal:
            stop='nonfinite_or_infeasible_trial';counts['failed_trials']+=1;break
        improved=value<best
        if improved:best=value;best_map=result.vertices.detach().clone();counts['accepted_steps']+=1
        trace.append(dict(step=counts['gradient_steps']-1,total=value,**{k:float(v.detach()) for k,v in parts.items()},
            accepted=improved,scale=float(result.scale.detach().min()),gauge=float(result.gauge.detach().max()),
            minimum_contracted_slack=float((q-floor).min())))
    return _finish(best_map,initial,best,trace,counts,stop,start,seconds,clock,base,floor,normalize,
        original_adam_floor=minimum_jacobian)
