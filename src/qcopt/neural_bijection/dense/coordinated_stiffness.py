"""Exact tensor Galerkin stiffness inverse and bounded physical-fiber descent.

Instance optimizer only: the inverse is a GLOBAL structured linear solve. It is
not a new topology decoder, an anatomical guarantee or differentiation through
an optimizer trajectory. Fine P1 area normalization and batch size one matter.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch.nn import functional as F

from .coordinated_update import interpolate_proposal, single_direction_corner_change
from .digital_q1 import q1_corner_determinants


def dst1_orthonormal(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Self-inverse orthonormal DST-I, through an odd extension of length2(n+1)."""
    if value.dtype not in (torch.float32,torch.float64) or value.shape[dim]<1:
        raise ValueError("nonempty floating transform axis required")
    x=value.movedim(dim,-1); n=x.shape[-1]
    zero=torch.zeros_like(x[...,:1])
    odd=torch.cat((zero,x,zero,-x.flip(-1)),dim=-1)
    result=-torch.fft.rfft(odd,dim=-1).imag[...,1:n+1]/math.sqrt(2*(n+1))
    return result.movedim(-1,dim)


class DirichletGalerkinStiffness:
    """H=weight*P.T Kfine P for zero-boundary bilinear RAW prolongation.

    Kfine is the scalar frozen-rotation Hessian of .5*mean_faces||J-R||² on
    a batch-one uniform unit square, with actual P1-ac faces. It is the unscaled
    five-point Dirichlet matrix (diag4, cardinal-1), not2K or a mass-lumped PDE.
    """
    def __init__(self,fine_side:int,coefficient_side:int,*,weight:float=3.,
                 device="cpu",dtype=torch.float64):
        if (any(isinstance(v,bool) or not isinstance(v,int) or v<3 for v in (fine_side,coefficient_side))
                or coefficient_side>fine_side or (fine_side-1)%(coefficient_side-1)
                or not math.isfinite(weight) or weight<=0
                or dtype not in (torch.float32,torch.float64)):
            raise ValueError("aligned integer-refinement square grids and positive finite weight required")
        self.fine_side,self.coefficient_side,self.weight=fine_side,coefficient_side,float(weight)
        self.refinement=(fine_side-1)//(coefficient_side-1)
        n=coefficient_side-1; r=self.refinement
        theta=torch.arange(1,n,device=device,dtype=dtype)*(math.pi/n)
        stiffness=(2-2*theta.cos())/r
        mass=((2*r*r+1)+(r*r-1)*theta.cos())/(3*r)
        self.eigenvalues=weight*(stiffness[:,None]*mass[None,:]+mass[:,None]*stiffness[None,:])
        if not bool(torch.isfinite(self.eigenvalues).all() and (self.eigenvalues>0).all()):
            raise ValueError("nonpositive or nonfinite spectral stiffness")

    def _check(self,value):
        n=self.coefficient_side-2
        if (value.shape[-2:]!=(n,n) or value.dtype!=self.eigenvalues.dtype
                or value.device!=self.eigenvalues.device):
            raise ValueError("coefficient interior shape/dtype/device must match stiffness")

    def solve(self,gradient):
        self._check(gradient)
        transformed=dst1_orthonormal(dst1_orthonormal(gradient,-1),-2)/self.eigenvalues
        return dst1_orthonormal(dst1_orthonormal(transformed,-1),-2)

    def apply(self,coefficients):
        self._check(coefficients)
        transformed=dst1_orthonormal(dst1_orthonormal(coefficients,-1),-2)*self.eigenvalues
        return dst1_orthonormal(dst1_orthonormal(transformed,-1),-2)

    def prolong(self,coefficients):
        self._check(coefficients)
        if coefficients.ndim!=3 or coefficients.shape[0]!=1:
            raise ValueError("batch-one scalar coefficient array required")
        return interpolate_proposal(F.pad(coefficients,(1,1,1,1)),(self.fine_side,self.fine_side))


@dataclass
class StiffnessFiberResult:
    vertices: torch.Tensor
    coefficients: torch.Tensor
    initial_objective: float
    final_objective: float
    trace: list[dict]
    counts: dict[str,int]
    stop_reason: str
    minimum_contracted_slack: float


def _parts(values):
    return {key:float(value.detach()) if isinstance(value,torch.Tensor) else float(value)
            for key,value in values.items()}


def optimize_stiffness_fiber(anchor:torch.Tensor,objective:Callable,*,coefficient_side:int,
                            direction=(1.,0.),strain_weight=3.,minimum_jacobian=.001,
                            theta=.95,fraction=.99,armijo=1e-4,max_backtracks=12,
                            maximum_gradients=30) -> StiffnessFiberResult:
    """Keep ORIGINAL anchor/floor fixed; accept only complete-objective Armijo.

    Q(Y0+(Pc)e) is affine in actual physical c. The contracted floor is
    eta+(1-theta)*(Q(Y0)-eta), NOT recomputed from current accepted iterates.
    Every trial is rebuilt from Y0 and c+a*d. No candidate chaining or repair.
    """
    if (anchor.ndim!=4 or anchor.shape[0]!=1 or anchor.shape[-1]!=2
            or anchor.shape[1]!=anchor.shape[2] or anchor.shape[1]<3
            or anchor.dtype!=torch.float64 or not bool(torch.isfinite(anchor).all())
            or not 0<minimum_jacobian<1 or not 0<theta<1 or not 0<fraction<1
            or not 0<armijo<1 or isinstance(maximum_gradients,bool) or maximum_gradients<1
            or isinstance(max_backtracks,bool) or max_backtracks<0):
        raise ValueError("finite batch-one square float64 geometry and valid search constants required")
    base=anchor.detach().clone(); side=base.shape[1]
    e=torch.as_tensor(direction,device=base.device,dtype=base.dtype)
    if e.shape!=(2,) or not bool(torch.isfinite(e).all()) or abs(float(e.square().sum())-1)>1e-12:
        raise ValueError("one unit scalar direction required")
    metric=DirichletGalerkinStiffness(side,coefficient_side,weight=strain_weight,
                                    device=base.device,dtype=base.dtype)
    normalize=float((side-1)**2)
    q0=q1_corner_determinants(base)*normalize
    if not bool((q0>minimum_jacobian).all()): raise ValueError("strict positive configured anchor floor required")
    floor=minimum_jacobian+(1-theta)*(q0-minimum_jacobian)
    c=base.new_zeros((1,coefficient_side-2,coefficient_side-2))
    def vertices(value): return base+metric.prolong(value)[...,None]*e
    def geometry(candidate):
        q=q1_corner_determinants(candidate)*normalize
        finite=bool(torch.isfinite(candidate).all() and torch.isfinite(q).all())
        return finite and bool((q>floor).all()),q
    counts=dict(gradient_steps=0,accepted_steps=0,objective_evaluations=0,
                trial_evaluations=0,backtracks=0,failed_trials=0)
    trace=[]; initial=None; current=base; final=float("nan"); stop="gradient_budget"
    for step in range(maximum_gradients):
        variable=c.detach().requires_grad_(True)
        current=vertices(variable)
        total,parts=objective(current); counts["objective_evaluations"]+=1
        value=float(total.detach())
        if initial is None: initial=value
        if not math.isfinite(value):
            stop="nonfinite_current_objective"; counts["failed_trials"]+=1; break
        gradient=torch.autograd.grad(total,variable)[0]; counts["gradient_steps"]+=1
        final=value
        if not bool(torch.isfinite(gradient).all()):
            stop="nonfinite_gradient"; counts["failed_trials"]+=1; break
        if not bool((gradient!=0).any()): stop="zero_computed_gradient"; break
        descent=-metric.solve(gradient)
        slope=float((gradient*descent).sum())
        if not bool(torch.isfinite(descent).all()) or not math.isfinite(slope) or slope>=0:
            stop="nonfinite_or_nondescent_direction"; counts["failed_trials"]+=1; break
        with torch.no_grad():
            feasible,q=geometry(current)
            if not feasible:
                stop="rounded_current_contracted_failure"; counts["failed_trials"]+=1; break
            slack=q-floor
            change=single_direction_corner_change(base,metric.prolong(descent),e)*normalize
            declining=change<0
            amax=float((slack[declining]/(-change[declining])).min()) if bool(declining.any()) else math.inf
            alpha=min(1.,fraction*amax)
            detail=dict(step=step,total=value,**_parts(parts),directional_derivative=slope,
                gradient_rms=float(gradient.square().mean().sqrt()),
                proposed_displacement_rms=float(metric.prolong(descent).square().mean().sqrt()),
                alpha_max=amax if math.isfinite(amax) else None,initial_alpha=alpha,
                minimum_contracted_slack=float(slack.min()),trials=[])
            accepted=False
            for backtrack in range(max_backtracks+1):
                next_c=c+alpha*descent
                candidate=vertices(next_c)
                valid,query_q=geometry(candidate)
                candidate_value=None; armijo_rhs=value+armijo*alpha*slope
                if valid:
                    candidate_total,candidate_parts=objective(candidate)
                    counts["objective_evaluations"]+=1; counts["trial_evaluations"]+=1
                    candidate_value=float(candidate_total)
                    valid=math.isfinite(candidate_value)
                if not valid: counts["failed_trials"]+=1
                success=valid and candidate_value<=armijo_rhs
                detail["trials"].append(dict(alpha=alpha,total=candidate_value if candidate_value is None or math.isfinite(candidate_value) else None,
                    armijo_rhs=armijo_rhs,geometry_valid=bool((query_q>floor).all()),accepted=success))
                if success:
                    c=next_c.detach(); current=candidate.detach(); final=candidate_value
                    counts["accepted_steps"]+=1; accepted=True
                    detail.update(accepted_alpha=alpha,accepted_total=final,
                        accepted_displacement_rms=float((alpha*metric.prolong(descent)).square().mean().sqrt()))
                    break
                if backtrack<max_backtracks:
                    alpha*=.5; counts["backtracks"]+=1
            detail["accepted"]=accepted; trace.append(detail)
            if not accepted: stop="line_search_exhausted"; break
    # Always reconstruct the accepted coefficients. A failed trial is never exported.
    with torch.no_grad():
        output=vertices(c); valid,q=geometry(output)
        if not valid: raise RuntimeError("accepted fiber output failed actual contracted corner check")
        if not (torch.equal(output[:,0],base[:,0]) and torch.equal(output[:,-1],base[:,-1])
                and torch.equal(output[:,:,0],base[:,:,0]) and torch.equal(output[:,:,-1],base[:,:,-1])):
            raise RuntimeError("actual boundary changed")
    return StiffnessFiberResult(output.detach(),c.detach(),float(initial),final,trace,counts,stop,float((q-floor).min()))
