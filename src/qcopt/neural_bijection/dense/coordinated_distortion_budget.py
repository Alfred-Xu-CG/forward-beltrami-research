"""Physical scalar-fiber descent under a fixed actual P1 ARAP budget.

The current-rotation quadratic is an upper bound, not the nonlinear ARAP Hessian.
Its scalar multiplier is found by a bounded approximate search, not an exact KKT
oracle. Every accepted map passes actual geometry, ARAP and strict data descent.
This instance optimizer is not differentiated through and certifies no global
stationarity or anatomical accuracy. Global injectivity also assumes a legal
incoming boundary; this core preserves that actual boundary without repair.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable

import torch

from .coordinated_stiffness import DirichletGalerkinStiffness,dst1_orthonormal
from .coordinated_update import single_direction_corner_change
from .digital_q1 import q1_corner_determinants


def _dst2(value):
    return dst1_orthonormal(dst1_orthonormal(value,-1),-2)


class DistortionBudgetMetric:
    """Simultaneous exact DST spectra of unweighted K and M=P'P/fine_side²."""
    def __init__(self,fine_side:int,coefficient_side:int,*,device='cpu',dtype=torch.float64):
        self.stiffness=DirichletGalerkinStiffness(fine_side,coefficient_side,weight=1.,device=device,dtype=dtype)
        t=self.stiffness.refinement;n=coefficient_side-1
        theta=torch.arange(1,n,device=device,dtype=dtype)*(math.pi/n)
        mass=((2*t*t+1)+(t*t-1)*theta.cos())/(3*t)
        self.mass_eigenvalues=mass[:,None]*mass[None,:]/(fine_side*fine_side)
        if not bool(torch.isfinite(self.mass_eigenvalues).all() and (self.mass_eigenvalues>0).all()):
            raise ValueError('nonpositive or nonfinite physical mass spectrum')

    def prolong(self,value):
        return self.stiffness.prolong(value)

    def apply_mass(self,value):
        self.stiffness._check(value)
        return _dst2(_dst2(value)*self.mass_eigenvalues)

    def solve_mass(self,value):
        self.stiffness._check(value)
        return _dst2(_dst2(value)/self.mass_eigenvalues)


@torch.no_grad()
def solve_budget_direction(gradient,distortion_gradient,metric,*,remaining_budget,
                           physical_rms_step,max_bracket_probes=32,max_bisections=32):
    """Solve one spectral QCQP approximately; return direction and truthful status.

    Production uses the fixed32 positive probes3*2**k and32bisections. Smaller
    explicit caps are supported for tiny failure tests, never an adaptive retry.
    Feasible upper endpoints still require an actual negative data derivative.
    """
    metric.stiffness._check(gradient);metric.stiffness._check(distortion_gradient)
    if (gradient.ndim!=3 or gradient.shape[0]!=1 or gradient.shape!=distortion_gradient.shape
            or not math.isfinite(remaining_budget) or remaining_budget<0
            or not math.isfinite(physical_rms_step) or physical_rms_step<=0
            or isinstance(max_bracket_probes,bool) or not isinstance(max_bracket_probes,int)
            or not 1<=max_bracket_probes<=32 or isinstance(max_bisections,bool)
            or not isinstance(max_bisections,int) or not 0<=max_bisections<=32):
        raise ValueError('batch-one scalar gradients, nonnegative slack and valid fixed solve budgets required')
    zero=torch.zeros_like(gradient)
    record=dict(numerical_failure=False,stop_reason=None,tau=None,multiplier=None,
        bracket_lower=None,bracket_upper=None,constraint_residual=None,cap_active=False,
        dual_evaluations=0,dual_bracket_probes=0,dual_bisections=0,
        directional_derivative=None,proposal_physical_rms=None,
        approximate_root=True,exact_kkt_claim=False)
    def stop(reason,*,failure=False,direction=None):
        record.update(stop_reason=reason,numerical_failure=failure)
        return zero if direction is None else direction,record
    if not bool(torch.isfinite(gradient).all() and torch.isfinite(distortion_gradient).all()):
        return stop('nonfinite_gradient',failure=True)
    k=metric.stiffness.eigenvalues;m=metric.mass_eigenvalues
    if not bool(torch.isfinite(k).all() and (k>0).all() and torch.isfinite(m).all() and (m>0).all()):
        return stop('nonfinite_or_nonpositive_spectrum',failure=True)
    if not bool((gradient!=0).any()):return stop('zero_data_gradient')
    if remaining_budget==0 and not bool((distortion_gradient!=0).any()):return stop('zero_ellipsoid')
    ghat,rhat=_dst2(gradient),_dst2(distortion_gradient)
    norm2=float((ghat.square()/m).sum())
    if not math.isfinite(norm2) or norm2<=0:return stop('nonfinite_or_zero_metric_norm',failure=True)
    tau=physical_rms_step/math.sqrt(norm2);record['tau']=tau
    if not math.isfinite(tau) or tau<=0:return stop('nonfinite_or_zero_tau',failure=True)
    def evaluate(mu):
        record['dual_evaluations']+=1
        d=-(ghat+mu*rhat)/(m/tau+mu*k)
        psi=float((rhat*d+.5*k*d.square()).sum())-remaining_budget
        finite=bool(torch.isfinite(d).all()) and math.isfinite(psi)
        return d,psi,finite
    chosen,psi,finite=evaluate(0.)
    if not finite:return stop('nonfinite_dual_evaluation',failure=True)
    lower=upper=0.
    if psi>0:
        record['cap_active']=True
        bracketed=False
        for probe in range(max_bracket_probes):
            upper=3.*(2**probe);record['dual_bracket_probes']+=1
            candidate,value,finite=evaluate(upper)
            if not finite:return stop('nonfinite_dual_evaluation',failure=True)
            if value<=0:
                chosen,psi=candidate,value;bracketed=True;break
            lower=upper
        if not bracketed:
            record.update(bracket_lower=lower,bracket_upper=None,constraint_residual=value)
            return stop('dual_bracket_failure',failure=True)
        for _ in range(max_bisections):
            mid=(lower+upper)/2
            if mid==lower or mid==upper:break
            record['dual_bisections']+=1
            candidate,value,finite=evaluate(mid)
            if not finite:return stop('nonfinite_dual_evaluation',failure=True)
            if value<=0:upper=mid;chosen,psi=candidate,value
            else:lower=mid
    direction=_dst2(chosen)
    record.update(multiplier=upper,bracket_lower=lower,bracket_upper=upper,constraint_residual=psi)
    if not bool(torch.isfinite(direction).all()):return stop('nonfinite_direction',failure=True)
    slope=float((gradient*direction).sum())
    rms=math.sqrt(float((m*chosen.square()).sum()))
    if not math.isfinite(slope) or not math.isfinite(rms):return stop('nonfinite_direction',failure=True)
    record.update(directional_derivative=slope,proposal_physical_rms=rms)
    if not bool((direction!=0).any()):return stop('zero_computed_direction',direction=direction)
    if slope>=0:return stop('finite_nondescent_direction',direction=direction)
    return direction,record


@dataclass
class DistortionBudgetFiberResult:
    vertices:torch.Tensor
    coefficients:torch.Tensor
    initial_objective:float | None
    final_objective:float | None
    initial_distortion:float | None
    final_distortion:float | None
    distortion_budget:float
    trace:list[dict]
    counts:dict[str,int]
    stop_reason:str
    minimum_contracted_slack:float
    numerical_failure:bool


def _counts():
    return dict(gradient_steps=0,data_vjps=0,distortion_vjps=0,objective_evaluations=0,
        trial_attempts=0,trial_evaluations=0,accepted_steps=0,backtracks=0,rejected_trials=0,
        dual_evaluations=0,dual_bracket_probes=0,dual_bisections=0,numerical_failures=0)


def _scalar_parts(parts):
    result={}
    for key,value in parts.items():
        if isinstance(value,torch.Tensor):
            if value.numel()!=1:continue
            value=float(value.detach())
        if isinstance(value,(int,float)):
            result[key]=value if math.isfinite(value) else None
    return result


def optimize_distortion_budget_fiber(anchor:torch.Tensor,objective_parts:Callable,*,coefficient_side:int,
        direction,distortion_budget:float,physical_rms_step:float,maximum_gradients:int=30,
        minimum_jacobian=.001,theta=.95,fraction=.99,armijo=1e-4,max_backtracks:int=12):
    """Fixed-anchor RAW fiber; ARAP cap and strict own-D Armijo acceptance.

    objective_parts(Y) returns (D,parts), with differentiable parts['strain']
    equal to the original UNWEIGHTED fine P1-ac ARAP. D is assembled directly
    without ARAP. The caller sets B once from the original incumbent and must
    preserve B through all stages. Early legal stops do not certify stationarity.
    """
    if (not isinstance(anchor,torch.Tensor) or anchor.dtype!=torch.float64 or anchor.ndim!=4
            or anchor.shape[0]!=1 or anchor.shape[-1]!=2 or anchor.shape[1]!=anchor.shape[2]
            or anchor.shape[1]<3 or not bool(torch.isfinite(anchor).all())
            or tuple(direction) not in ((1.,0.),(0.,1.))
            or isinstance(distortion_budget,bool) or not math.isfinite(distortion_budget) or distortion_budget<0
            or isinstance(physical_rms_step,bool) or not math.isfinite(physical_rms_step) or physical_rms_step<=0
            or not 0<minimum_jacobian<1 or not 0<theta<1 or not 0<fraction<1 or not 0<armijo<1
            or isinstance(maximum_gradients,bool) or not isinstance(maximum_gradients,int) or maximum_gradients<1
            or isinstance(max_backtracks,bool) or not isinstance(max_backtracks,int) or not 0<=max_backtracks<=12):
        raise ValueError('finite batch-one square float64 map, one axis, nonnegative fixed budget and valid search settings required')
    base=anchor.detach().clone();side=base.shape[1];e=base.new_tensor(direction);budget=float(distortion_budget)
    metric=DistortionBudgetMetric(side,coefficient_side,device=base.device,dtype=base.dtype)
    normalize=float((side-1)**2);q0=q1_corner_determinants(base)*normalize
    if not bool(torch.isfinite(q0).all() and (q0>minimum_jacobian).all()):
        raise ValueError('finite strict configured anchor corner floor required')
    floor=minimum_jacobian+(1-theta)*(q0-minimum_jacobian)
    c=base.new_zeros((1,coefficient_side-2,coefficient_side-2))
    vertices=lambda value:base+metric.prolong(value)[...,None]*e
    def geometry(value):
        q=q1_corner_determinants(value)*normalize
        finite=bool(torch.isfinite(value).all() and torch.isfinite(q).all())
        perimeter=all(torch.equal(a,b) for a,b in ((value[:,0],base[:,0]),(value[:,-1],base[:,-1]),
            (value[:,:,0],base[:,:,0]),(value[:,:,-1],base[:,:,-1])))
        return finite,finite and perimeter and bool((q>floor).all()),q
    counts=_counts();trace=[];initial_d=initial_r=final_d=final_r=None
    numerical_failure=False;stop='gradient_budget';current=base
    def evaluate(value,*,trial=False):
        counts['objective_evaluations']+=1
        if trial:counts['trial_evaluations']+=1
        data,parts=objective_parts(value)
        if (not isinstance(data,torch.Tensor) or data.numel()!=1 or not isinstance(parts,dict)
                or not isinstance(parts.get('strain'),torch.Tensor) or parts['strain'].numel()!=1):
            raise ValueError('callback requires scalar tensor D and differentiable scalar tensor parts[strain]')
        return data,parts['strain'],parts
    def fail(reason):
        nonlocal numerical_failure,stop
        numerical_failure=True;stop=reason;counts['numerical_failures']+=1
    for step in range(maximum_gradients):
        variable=c.detach().requires_grad_(True);current=vertices(variable)
        with torch.no_grad():
            finite,valid,q=geometry(current)
        if not finite or not valid:
            fail('invalid_current_geometry');break
        data,distortion,parts=evaluate(current)
        value=float(data.detach());rvalue=float(distortion.detach())
        if not math.isfinite(value) or not math.isfinite(rvalue):
            fail('nonfinite_current_objective_or_distortion');break
        if initial_d is None:initial_d,initial_r=value,rvalue
        final_d,final_r=value,rvalue
        if rvalue<0 or rvalue>budget:
            fail('current_distortion_outside_budget');break
        if not data.requires_grad or not distortion.requires_grad:
            raise ValueError('callback D and strain must remain connected for separate VJPs')
        gradient=torch.autograd.grad(data,variable,retain_graph=True)[0];counts['data_vjps']+=1
        rgradient=torch.autograd.grad(distortion,variable)[0];counts['distortion_vjps']+=1
        counts['gradient_steps']+=1
        detail=dict(step=step,objective=value,distortion=rvalue,distortion_budget=budget,
            remaining_budget=budget-rvalue,parts=_scalar_parts(parts),trials=[],accepted=False)
        descent,solve=solve_budget_direction(gradient.detach(),rgradient.detach(),metric,
            remaining_budget=budget-rvalue,physical_rms_step=physical_rms_step)
        detail['dual']=solve
        for key in ('dual_evaluations','dual_bracket_probes','dual_bisections'):counts[key]+=solve[key]
        if solve['numerical_failure']:
            fail(solve['stop_reason']);trace.append(detail);break
        if solve['stop_reason'] is not None:
            stop=solve['stop_reason'];trace.append(detail);break
        slope=solve['directional_derivative'];detail['directional_derivative']=slope
        with torch.no_grad():
            prolonged=metric.prolong(descent)
            change=single_direction_corner_change(base,prolonged,e)*normalize
            slack=q-floor
            if not bool(torch.isfinite(prolonged).all() and torch.isfinite(change).all() and torch.isfinite(slack).all()):
                fail('nonfinite_corner_direction');trace.append(detail);break
            declining=change<0
            amax=float((slack[declining]/(-change[declining])).min()) if bool(declining.any()) else math.inf
            alpha=fraction*min(1.,amax)
            detail.update(alpha_max=amax if math.isfinite(amax) else None,unbounded_corner_step=not bool(declining.any()),
                initial_alpha=alpha,minimum_contracted_slack=float(slack.min()))
            if not math.isfinite(alpha) or alpha<=0:
                fail('invalid_initial_step_length');trace.append(detail);break
            accepted=False
            for backtrack in range(max_backtracks+1):
                counts['trial_attempts']+=1
                next_c=c+alpha*descent;candidate=vertices(next_c)
                finite,valid,query_q=geometry(candidate)
                trial=dict(alpha=alpha,objective=None,distortion=None,geometry_valid=valid,
                    budget_valid=None,armijo_rhs=value+armijo*alpha*slope,armijo_valid=None,
                    strict_descent=None,accepted=False)
                if not finite:
                    trial['failure']='nonfinite_trial_geometry';detail['trials'].append(trial)
                    fail('nonfinite_trial_geometry');break
                if valid:
                    candidate_data,candidate_r,_=evaluate(candidate,trial=True)
                    trial_d,trial_r=float(candidate_data),float(candidate_r)
                    if not math.isfinite(trial_d) or not math.isfinite(trial_r):
                        trial['failure']='nonfinite_trial_objective_or_distortion'
                        detail['trials'].append(trial);fail('nonfinite_trial_objective_or_distortion');break
                    trial.update(objective=trial_d,distortion=trial_r,budget_valid=0<=trial_r<=budget,
                        armijo_valid=trial_d<=trial['armijo_rhs'],strict_descent=trial_d<value)
                    trial['accepted']=trial['budget_valid'] and trial['armijo_valid'] and trial['strict_descent']
                detail['trials'].append(trial)
                if trial['accepted']:
                    c=next_c.detach();current=candidate.detach();final_d,final_r=trial_d,trial_r
                    counts['accepted_steps']+=1;accepted=True
                    detail.update(accepted=True,accepted_alpha=alpha,accepted_objective=final_d,
                        accepted_distortion=final_r,accepted_displacement_rms=float((alpha*prolonged).square().mean().sqrt()))
                    break
                counts['rejected_trials']+=1
                if backtrack<max_backtracks:alpha*=.5;counts['backtracks']+=1
            trace.append(detail)
            if numerical_failure:break
            if not accepted:stop='line_search_exhausted';break
    with torch.no_grad():
        output=vertices(c);finite,valid,q=geometry(output)
        if not finite or not valid:
            if not numerical_failure:fail('invalid_output_geometry')
        if final_r is not None and not 0<=final_r<=budget:
            if not numerical_failure:fail('output_distortion_outside_budget')
        minimum_slack=float((q-floor).min()) if bool(torch.isfinite(q).all()) else float('nan')
    return DistortionBudgetFiberResult(output.detach(),c.detach(),initial_d,final_d,initial_r,final_r,
        budget,trace,counts,stop,minimum_slack,numerical_failure)
