"""Isolated exact nested-prior quadrature benchmark, not registration timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from qcopt.neural_bijection.dense.coordinated_nested_priors import ExactNestedP1Priors
from qcopt.neural_bijection.dense.coordinated_nested_refinement import FrozenNestedP1Refinement
from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from tools.coordinated_real_case import corner_symmetric_dirichlet
from tools.digital_mind_safe_optimize import strain_penalty
from tools.coordinated_dense_operator_benchmark import normalized_corners


def identity(side,device,dtype=torch.float64):
    axis=torch.arange(side,dtype=torch.float32,device=device)/(side-1)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    return torch.stack((x,y),-1)[None].to(dtype)


def legal_anchor(side,device):
    reference=identity(side,device)
    x,y=reference[...,0],reference[...,1]
    window=torch.sin(torch.pi*x).square()*torch.sin(torch.pi*y).square()
    window[:,0]=window[:,-1]=0;window[:,:,0]=window[:,:,-1]=0
    # Earlier diagnostic anchor (.01,-.007) plus the .002 x proposal. The
    # analytical rank-one field is simple; actual sampled corners are checked.
    return reference+window[...,None]*reference.new_tensor((.012,-.007))


def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def _allocated(device):
    return torch.cuda.memory_allocated(device) if device.type=="cuda" else None


def build(method,side,fine_side,diagonal,device,dtype):
    if method=="explicit_fine":
        materializer=FrozenNestedP1Refinement(side,fine_side,diagonal=diagonal,dtype=dtype,device=device)
        def functional(vertices):
            fine=materializer(vertices)
            return strain_penalty(fine),corner_symmetric_dirichlet(fine)
        return functional,materializer.resident_bytes
    module=ExactNestedP1Priors(side,fine_side,diagonal=diagonal,dtype=dtype,device=device)
    return module,module.resident_bytes


def compare(anchor,fine_side,diagonal):
    old,_=build("explicit_fine",anchor.shape[1],fine_side,diagonal,anchor.device,anchor.dtype)
    new,_=build("coarse_quadrature",anchor.shape[1],fine_side,diagonal,anchor.device,anchor.dtype)
    y=anchor.detach().clone().requires_grad_()
    full=old(y);small=new(y)
    full_g,=torch.autograd.grad(3*full[0]+.0001*full[1],y)
    small_g,=torch.autograd.grad(3*small[0]+.0001*small[1],y)
    errors={}
    tiny=torch.finfo(anchor.dtype).tiny
    for name,a,b in zip(("strain","shape"),full,small):
        errors[name]=dict(explicit_fine_value=float(a.detach()),coarse_value=float(b.detach()),
            maximum_absolute_value_error=float((a-b).abs()),
            relative_value_error=float((a-b).abs()/a.abs().clamp_min(tiny)))
    relative=float(torch.linalg.vector_norm(full_g-small_g)/torch.linalg.vector_norm(full_g).clamp_min(tiny))
    absolute=float((full_g-small_g).abs().max())
    if (relative>1e-8 or any(e["relative_value_error"]>1e-8 for e in errors.values()) or
            not bool(torch.isfinite(full_g).all() and torch.isfinite(small_g).all())):
        raise RuntimeError("fine/coarse weighted prior value or fullY gradient disagreement")
    return dict(prior_value_errors=errors,weighted_gradient_relative_l2_error=relative,
                weighted_gradient_maximum_absolute_error=absolute,
                reference_weighted_gradient_rms=float(full_g.square().mean().sqrt()))


def measure(anchor,fine_side,diagonal,method,args):
    if args.warmup<3 or args.repeats<10:raise ValueError("warm3/repeat10 minimum")
    device=anchor.device
    _sync(device);before=_allocated(device);tick=time.perf_counter()
    functional,constant_bytes=build(method,anchor.shape[1],fine_side,diagonal,device,anchor.dtype)
    _sync(device);setup=time.perf_counter()-tick;after=_allocated(device)
    forward_times=[];backward_times=[];complete_times=[];copies=[];residents=[];peaks=[]
    for repeat in range(args.warmup+args.repeats):
        _sync(device);tick=time.perf_counter();y=anchor.detach().clone().requires_grad_();_sync(device)
        copy_time=time.perf_counter()-tick;resident=_allocated(device)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        tick=time.perf_counter();strain,shape=functional(y);score=3*strain+.0001*shape;_sync(device)
        forward=time.perf_counter()-tick
        tick=time.perf_counter();gradient,=torch.autograd.grad(score,y);_sync(device)
        backward=time.perf_counter()-tick
        peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
        if not bool(torch.isfinite(score) and torch.isfinite(gradient).all()):
            raise RuntimeError("nonfinite weighted prior or fullY VJP")
        if repeat>=args.warmup:
            forward_times.append(forward);backward_times.append(backward);complete_times.append(forward+backward)
            copies.append(copy_time);residents.append(resident);peaks.append(peak)
        score_value=float(score.detach());del y,strain,shape,score,gradient
    del functional
    _sync(device);cleaned=_allocated(device)
    if cleaned!=before:raise RuntimeError("prior benchmark retained CUDA allocation after cleanup")
    return dict(constructor_seconds=setup,constant_buffer_bytes=constant_bytes,
        constructor_resident_increment_bytes=None if before is None else after-before,
        resident_before_constructor_bytes=before,resident_after_cleanup_bytes=cleaned,
        input_leaf_copy_seconds=copies,forward_seconds=forward_times,vjp_seconds=backward_times,
        forward_plus_vjp_seconds=complete_times,median_forward_seconds=statistics.median(forward_times),
        median_vjp_seconds=statistics.median(backward_times),median_forward_plus_vjp_seconds=statistics.median(complete_times),
        resident_allocated_bytes=residents,peak_total_allocated_bytes=peaks,
        peak_increment_over_resident_bytes=[p-r for p,r in zip(peaks,residents)] if device.type=="cuda" else None,
        weighted_prior_value=score_value,
        reference_construction="old fine identity float32->geometry dtype constructed INSIDE every strain forward" if method=="explicit_fine"
            else "coarse identity float32->geometry dtype/counts constructed ONCE, reported setup/cache")


def benchmark_case(side,fine_side,diagonal,args):
    device=torch.device(args.device);anchor=legal_anchor(side,device)
    coarse_qmin=normalized_corners(anchor,identity(side,device))
    if coarse_qmin<=.001:raise RuntimeError("actual legal diagnostic coarse anchor check failed")
    result=dict(status="success",coarse_side=side,fine_side=fine_side,diagonal=diagonal,
        factor=(fine_side-1)//(side-1),initial_coarse_normalized_qmin=coarse_qmin,
        **compare(anchor,fine_side,diagonal))
    for method in ("explicit_fine","coarse_quadrature"):
        result[method]=measure(anchor,fine_side,diagonal,method,args)
    # This output check is intentionally OUTSIDE either timed prior path.
    materializer=FrozenNestedP1Refinement(side,fine_side,diagonal=diagonal,dtype=anchor.dtype,device=device)
    with torch.no_grad():
        fine=materializer(anchor);reference=identity(fine_side,device)
        qref=q1_corner_determinants(reference)
        _sync(device);tick=time.perf_counter()
        margin=(q1_corner_determinants(fine)/qref-.001).amin()
        if not bool(torch.isfinite(fine).all() and torch.isfinite(margin) and margin>0):
            raise RuntimeError("fresh actual rounded fine eta margin failed")
        _sync(device);check_seconds=time.perf_counter()-tick
        result.update(fresh_actual_fine_corner_check_seconds=check_seconds,
            fresh_actual_normalized_fine_qmin=normalized_corners(fine,reference),
            fresh_fine_configured_margin=float(margin))
    return result


def float32_rounding_diagnostics(device):
    """Persist thin-cell float32 discrepancy, without declaring bitwise identity."""
    rows=[]
    for diagonal in ("ac","bd"):
        y=torch.tensor([[[[0.,0.],[1.,0.]],[[0.,1.],[1.,.0012]]]],dtype=torch.float32,device=device,requires_grad=True)
        new,_=build("coarse_quadrature",2,5,diagonal,device,y.dtype)
        # The application cached refiner intentionally requires >=3 controls;
        # the original exact-refinement function also supports this ONE quad.
        fine=refine_p1_vertices(y,4,diagonal)
        full=(strain_penalty(fine),corner_symmetric_dirichlet(fine));small=new(y)
        for name,a,b in zip(("strain","shape"),full,small):
            ga,=torch.autograd.grad(a,y,retain_graph=True);gb,=torch.autograd.grad(b,y,retain_graph=True)
            rows.append(dict(diagonal=diagonal,prior=name,dtype="float32",coarse_side=2,fine_side=5,
                refinement="untimed original one-quad exact P1 refinement, not cached application refiner",
                absolute_value_error=float((a-b).abs()),relative_value_error=float((a-b).abs()/a.abs().clamp_min(torch.finfo(y.dtype).tiny)),
                maximum_absolute_gradient_error=float((ga-gb).abs().max()),
                relative_gradient_l2_error=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(ga).clamp_min(torch.finfo(y.dtype).tiny)),
                finite=bool(torch.isfinite(a) and torch.isfinite(b) and torch.isfinite(ga).all() and torch.isfinite(gb).all())))
    return rows


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.warmup<3 or args.repeats<10:raise ValueError("warm3/repeat10 minimum")
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    payload=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},results=[],
        device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        scope="priors ONLY; batch1float64 samecoarseY/oldcachedexactP1materialization and actualoldfine strain/shape vs coarse quadrature; score3strain+1e-4shape fullYVJP; no image/objective optimization/neural encoder/anatomy conclusion",
        precision="dyadic source identity float32 THENcast unchanged; exact-arithmetic functional, NOT bitwise roundedfine equivalence; f32thinerrors explicitly retained",
        timing="constructor/refquerycache and source-reference/count setup separate; old per-forward reference creation INCLUDED; warm3repeat10; fresh actualfine coordinate check OUTSIDE priors timing and reported separately",
        method_order="within eachcase explicit_fine THEN coarse_quadrature, sameprocess; NOT randomized/ABBA",
        fresh_check_timing="one actualfine check percase outside repeated prior timers, NOT a warm-repeat cost estimate; cold/scheduling outliers retained",
        memory="CUDAallocated resident/completeforward+VJPpeak, notreserved; inputleafcopies separate; no saved-tensorhooks; all measured module/caches returnallocatedbaseline aftercleanup",
        anchor="synthetic x+.012window,y-.007window (includes earlier.002xraw), window sin²(pi*x)sin²(pi*y), boundaryfixed; NOT realtrajectory or target",
        caveat="no whole-instance speedup from this isolated benchmark; otherGPU CPUlaunch interference possible; sameprocess comparison notabsolute crossjob timing")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for fine_side in args.fine_sides:
        for side in args.coarse_sides:
            for diagonal in args.diagonals:
                try:result=benchmark_case(side,fine_side,diagonal,args)
                except Exception as error:
                    result=dict(status="failure",coarse_side=side,fine_side=fine_side,diagonal=diagonal,
                        error_type=type(error).__name__,error=str(error))
                payload["results"].append(result);args.output.write_text(json.dumps(payload,indent=2)+"\n")
                print(json.dumps(result),flush=True)
    payload["float32_thin_rounding_diagnostics"]=float32_rounding_diagnostics(device)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--fine-sides",nargs="+",type=int,default=[257,1025])
    p.add_argument("--coarse-sides",nargs="+",type=int,default=[17,65,257])
    p.add_argument("--diagonals",nargs="+",choices=("ac","bd"),default=["ac","bd"])
    p.add_argument("--threads",type=int,default=2);p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10)
    run(p.parse_args())


if __name__=="__main__":main()
