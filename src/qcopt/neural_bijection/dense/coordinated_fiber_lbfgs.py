"""Bounded instance L-BFGS on one fixed physical scalar coordinate fiber.

This is NOT a differentiable decoder or a new map class. Constant anchor,
reference and direction; no AD through iterations, inverse, projection, repair,
resampling or geometric line search. Analytic fraction-to-boundary and actual
rounded-map checks preserve the EXISTING analytic stage's contracted set.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .coordinated_update import single_direction_corner_change
from .digital_q1 import q1_corner_determinants


@dataclass
class CoordinatedFiberLBFGSResult:
    vertices: torch.Tensor
    displacement: torch.Tensor
    best_vertices: torch.Tensor
    best_displacement: torch.Tensor
    initial_objective: float
    final_objective: float
    best_objective: float
    trace: list[dict]
    counts: dict[str, int]
    stop_reason: str
    calibration: dict[str, float]


def _direction(gradient, history, gamma):
    """Standard two-loop inverse-Hessian action, with explicit descent guard."""
    q=gradient.clone(); alphas=[]
    for s,y,rho in reversed(history):
        alpha=rho*torch.sum(s*q); alphas.append(alpha); q=q-alpha*y
    r=gamma*q
    for (s,y,rho),alpha in zip(history,reversed(alphas)):
        r=r+s*(alpha-rho*torch.sum(y*r))
    direction=-r
    fallback=not bool(torch.isfinite(direction).all() and torch.sum(direction*gradient)<0)
    if fallback:direction=-gradient
    return direction,fallback


def _curvature_pair(s,y,tolerance):
    sy=torch.sum(s*y); yy=torch.sum(y*y)
    norm_product=torch.linalg.vector_norm(s)*torch.linalg.vector_norm(y)
    valid=bool(torch.isfinite(sy) and torch.isfinite(yy) and torch.isfinite(norm_product)
               and norm_product>0 and yy>0
               and sy>0 and sy>tolerance*norm_product)
    if not valid:return None,None
    rho=1/sy;gamma=sy/yy
    if not bool(torch.isfinite(rho) and rho>0 and torch.isfinite(gamma) and gamma>0):
        return None,None
    return (s.detach().clone(),y.detach().clone(),rho),float(gamma)


def _parts(values):
    if not isinstance(values,dict):raise ValueError("objective parts must be a dict of scalar diagnostics")
    output={}
    for key,value in values.items():
        if isinstance(value,torch.Tensor):
            if value.numel()!=1:raise ValueError("objective parts must contain scalar diagnostics")
            output[key]=float(value.detach())
        elif isinstance(value,(float,int,bool)):output[key]=float(value)
        else:raise ValueError("objective parts must contain scalar diagnostics")
    return output


def _interior_rms(gradient):
    interior=gradient[:,1:-1,1:-1]
    magnitude=float(interior.abs().max())
    # Avoid squaring an extremely small/large finite gradient before scaling.
    return magnitude*float((interior/magnitude).square().mean().sqrt()) if magnitude else 0.


def solve_coordinated_fiber_lbfgs(anchor: torch.Tensor,
        objective: Callable[[torch.Tensor],tuple[torch.Tensor,dict]], *,
        initial_displacement_rms: float, reference: torch.Tensor|None=None,
        direction=(1.,0.), minimum_jacobian: float=.001, theta: float=.95,
        fraction_to_boundary: float=.99, max_gradient_evaluations: int=30,
        history_size: int=5, max_backtracks: int=6, armijo: float=1e-4,
        minimum_step: float=1e-12, gradient_tolerance: float=0.,
        curvature_tolerance: float=1e-10) -> CoordinatedFiberLBFGSResult:
    """Optimize boundary-zero physical u in Y=anchor+u*unit(direction).

    All four normalized corner changes A*u are linear at the constant anchor.
    The allowed stage set is theta*s0+A*u>=0, NOT merely q(Y)/qref>eta.
    Start u=0; each trial alpha=min(1,.99*alpha_max), with exact affine
    alpha_max over ALL negative A*d rows. Each actual rounded candidate must
    have finite corners, strict contracted slack and strict eta margin. A failed
    rounded geometry check STOPS, without geometry halving or projection.

    Complete objective Armijo performs at most max_backtracks+1 forwards (first
    trial plus max_backtracks halvings); only accepted finite losses get a VJP.
    Maximum gradient evaluations includes the initial gradient; there is no
    uncounted terminal trial. Thus 30 gradients permit at most 29 updates.
    Callback takes vertices in anchor dtype and returns scalar Tensor loss plus
    scalar diagnostic dict. All forwards/gradient attempts/rejections are counted.

    First H0 gamma calibrates interior RMS(-gamma*g0) to the EXPLICIT physical
    initial_displacement_rms. No assumed level/learning-rate formula. Later
    reliable curvature pairs use gamma=sTy/yTy, not repeated RMS normalization.
    Nonpositive/unreliable curvature is skipped; non-descent direction falls
    back to -g. No stationarity/global-convergence claim on early termination.
    Solver arithmetic is float64; output coordinates retain anchor dtype.
    """
    if (not isinstance(anchor,torch.Tensor) or anchor.ndim!=4 or anchor.shape[0]<1
            or anchor.shape[-1]!=2 or min(anchor.shape[1:3])<3
            or anchor.dtype not in (torch.float32,torch.float64)):
        raise ValueError("float32/64 anchor(B,R>=3,C>=3,2) required")
    if anchor.requires_grad:raise ValueError("anchor must be constant, not require gradients")
    if reference is not None and (not isinstance(reference,torch.Tensor) or reference.requires_grad
            or reference.ndim!=4 or reference.shape[1:]!=anchor.shape[1:]
            or reference.shape[0] not in (1,anchor.shape[0])
            or reference.dtype not in (torch.float32,torch.float64)):
        raise ValueError("reference must be constant float32/64 matching grid, batch1 or B")
    if isinstance(direction,torch.Tensor) and direction.requires_grad:
        raise ValueError("direction must be constant, not require gradients")
    for name,value,lower,upper in (("initial_displacement_rms",initial_displacement_rms,0,math.inf),
            ("minimum_jacobian",minimum_jacobian,-1e-300,math.inf),("theta",theta,0,1),
            ("fraction_to_boundary",fraction_to_boundary,0,1),("armijo",armijo,0,1),
            ("minimum_step",minimum_step,0,math.inf),("gradient_tolerance",gradient_tolerance,-1e-300,math.inf),
            ("curvature_tolerance",curvature_tolerance,-1e-300,math.inf)):
        if (isinstance(value,(bool,torch.Tensor)) or not isinstance(value,(float,int))
                or not math.isfinite(value) or not lower<value<upper):
            raise ValueError(f"invalid constant {name}")
        if name in ("minimum_jacobian","gradient_tolerance","curvature_tolerance") and value<0:
            raise ValueError(f"invalid nonnegative {name}")
    for name,value,minimum in (("max_gradient_evaluations",max_gradient_evaluations,1),
                              ("history_size",history_size,1),("max_backtracks",max_backtracks,0)):
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
            raise ValueError(f"invalid integer {name}")
    if not callable(objective):raise ValueError("objective must be callable")
    base=anchor.detach().clone().double()
    e=torch.as_tensor(direction,dtype=torch.float64,device=anchor.device)
    if e.shape!=(2,) or not bool(torch.isfinite(e).all() and e.abs().max()>0):
        raise ValueError("finite nonzero direction(2,) required")
    # Finite entries can have an overflowing unscaled squared norm. Normalize
    # by the maximum first, so a legitimate extreme direction never becomes0.
    e=e.detach().clone()/e.abs().max()
    e=e/torch.linalg.vector_norm(e)
    batch,rows,columns,_=base.shape
    if reference is None:
        yy,xx=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64,device=base.device),
            torch.linspace(0,1,columns,dtype=torch.float64,device=base.device),indexing="ij")
        ref=torch.stack((xx,yy),-1)[None]
    else:ref=reference.detach().clone().to(device=base.device,dtype=torch.float64)
    qref=q1_corner_determinants(ref); q0=q1_corner_determinants(base)
    s0=q0/qref-minimum_jacobian
    if not bool(torch.isfinite(base).all() and torch.isfinite(ref).all()
                and torch.isfinite(qref).all() and (qref>0).all()
                and torch.isfinite(s0).all() and (s0>0).all()):
        raise ValueError("finite anchor/reference and strictly positive all-four margins required")
    mask=torch.ones((batch,rows,columns),dtype=torch.float64,device=base.device)
    mask[:,0]=mask[:,-1]=0; mask[:,:,0]=mask[:,:,-1]=0
    counts=dict(objective_evaluations=0,gradient_evaluations=0,rejected_objective_evaluations=0,
        rounded_geometry_checks=0,geometry_rejections=0,anchor_geometry_checks=1,
        affine_direction_evaluations=0,accepted_steps=0,curvature_skips=0,descent_fallbacks=0,
        nonfinite_gradient_rejections=0,armijo_halvings=0)
    trace=[]; history=[]; calibration={"initial_displacement_rms":float(initial_displacement_rms)}
    u=torch.zeros_like(mask); contracted=theta*s0

    def evaluate(candidate_u):
        leaf=candidate_u.detach().requires_grad_()
        vertices=(base+(leaf*mask)[...,None]*e).to(anchor.dtype)
        with torch.no_grad():
            actual_q=q1_corner_determinants(vertices.double())
            margin=actual_q/qref-minimum_jacobian
            actual_contract=theta*s0+(actual_q-q0)/qref
            counts["rounded_geometry_checks"]+=1
            valid=bool(torch.isfinite(vertices).all() and torch.isfinite(margin).all()
                       and torch.isfinite(actual_contract).all() and (margin>0).all() and (actual_contract>0).all())
        geometry=dict(actual_margin_min=float(margin.amin()),actual_contracted_margin_min=float(actual_contract.amin()))
        if not valid:
            counts["geometry_rejections"]+=1
            return None,geometry
        counts["objective_evaluations"]+=1
        output=objective(vertices)
        if not isinstance(output,tuple) or len(output)!=2:
            raise ValueError("objective must return (scalar Tensor, parts dict)")
        loss,parts=output
        if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not loss.is_floating_point():
            raise ValueError("objective loss must be a scalar floating Tensor")
        return (leaf,vertices,loss,_parts(parts)),geometry

    def differentiate(evaluated):
        leaf,_,loss,_=evaluated
        if not loss.requires_grad:raise ValueError("objective loss must retain its vertex gradient graph")
        counts["gradient_evaluations"]+=1
        gradient,=torch.autograd.grad(loss,leaf)
        return gradient.detach()*mask

    evaluated,geometry=evaluate(u)
    # Input validation already establishes initial rounded geometry.
    if evaluated is None:raise RuntimeError("initial rounded anchor failed contracted margins")
    initial=float(evaluated[2].detach()); current_value=initial; current_vertices=evaluated[1].detach()
    best_value=initial; best_vertices=current_vertices.clone(); best_u=u.clone()
    trace.append(dict(kind="initial",accepted=math.isfinite(initial),objective=initial,parts=evaluated[3],**geometry))
    stop="gradient_budget"
    if not math.isfinite(initial):
        counts["rejected_objective_evaluations"]+=1; stop="nonfinite_initial_objective"; gradient=None
    else:
        gradient=differentiate(evaluated)
        if not bool(torch.isfinite(gradient).all()):
            counts["nonfinite_gradient_rejections"]+=1; stop="nonfinite_initial_gradient"; gradient=None
    gamma=1.
    if gradient is not None:
        rms=_interior_rms(gradient)
        gamma=initial_displacement_rms/rms if rms>0 else 1.
        nonzero=bool((gradient!=0).any())
        valid=math.isfinite(gamma) and gamma>0 and (rms>0 or not nonzero)
        calibrated=gamma*gradient if valid else torch.zeros_like(gradient)
        valid=valid and bool(torch.isfinite(calibrated).all())
        calibration.update(initial_gradient_interior_rms=rms,initial_h0_scale=float(gamma),
                           initial_calibration_valid=valid,
                           first_descent_interior_rms=_interior_rms(calibrated) if valid else float("nan"))
        if not valid:stop="nonfinite_initial_calibration";gradient=None
    while gradient is not None and counts["gradient_evaluations"]<max_gradient_evaluations:
        if float(gradient.abs().max())<=gradient_tolerance:stop="gradient_tolerance";break
        d,fallback=_direction(gradient,history,gamma)
        counts["descent_fallbacks"]+=int(fallback)
        slope=float(torch.sum(gradient*d))
        if not math.isfinite(slope) or slope>=0:stop="no_finite_descent";break
        ad=single_direction_corner_change(base,d,e)/qref
        counts["affine_direction_evaluations"]+=1
        if not bool(torch.isfinite(ad).all()):stop="nonfinite_affine_direction";break
        negative=ad<0
        denominator=torch.where(negative,-ad,torch.ones_like(ad))
        alpha_max=float(torch.where(negative,contracted/denominator,torch.full_like(ad,float("inf"))).amin())
        alpha=min(1.,fraction_to_boundary*alpha_max)
        accepted=False
        for backtrack in range(max_backtracks+1):
            if not math.isfinite(alpha) or alpha<minimum_step:stop="minimum_step";break
            trial_u=(u+alpha*d)*mask
            trial_evaluated,trial_geometry=evaluate(trial_u)
            event=dict(kind="trial",alpha=alpha,alpha_max=alpha_max,backtrack=backtrack,
                       gradient_evaluations_before=counts["gradient_evaluations"],directional_derivative=slope,
                       descent_fallback=fallback,h0_scale=float(gamma),**trial_geometry)
            if trial_evaluated is None:
                event.update(accepted=False,rejection="rounded_geometry");trace.append(event)
                stop="rounded_geometry_rejected";break  # NO geometric backtracking.
            value=float(trial_evaluated[2].detach())
            event.update(objective=value,parts=trial_evaluated[3])
            if math.isfinite(value) and value<=current_value+armijo*alpha*slope:
                new_gradient=differentiate(trial_evaluated)
                if not bool(torch.isfinite(new_gradient).all()):
                    counts["nonfinite_gradient_rejections"]+=1
                    event.update(accepted=False,rejection="nonfinite_gradient");trace.append(event)
                    stop="nonfinite_trial_gradient";break
                pair,new_gamma=_curvature_pair(trial_u-u,new_gradient-gradient,curvature_tolerance)
                if new_gamma is not None:
                    history.append(pair);history=history[-history_size:];gamma=new_gamma
                else:counts["curvature_skips"]+=1
                u=trial_u.detach();gradient=new_gradient;contracted=contracted+alpha*ad
                current_value=value;current_vertices=trial_evaluated[1].detach()
                counts["accepted_steps"]+=1
                if value<best_value:
                    best_value=value;best_vertices=current_vertices.clone();best_u=u.clone()
                event.update(accepted=True,gradient_evaluations_after=counts["gradient_evaluations"],
                             curvature_pair_accepted=new_gamma is not None,history_length=len(history))
                trace.append(event);accepted=True;break
            counts["rejected_objective_evaluations"]+=1
            event.update(accepted=False,rejection="nonfinite_objective" if not math.isfinite(value) else "armijo")
            trace.append(event)
            if backtrack<max_backtracks:alpha*=.5;counts["armijo_halvings"]+=1
            else:stop="objective_backtrack_limit"
        if not accepted:break
    return CoordinatedFiberLBFGSResult(current_vertices.clone(),u.clone(),best_vertices,best_u,
        initial,current_value,best_value,trace,counts,stop,calibration)
