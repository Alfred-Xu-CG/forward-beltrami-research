"""Candidate-only engineering benchmark: BOTH Y/z gradients, same default checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import weakref

import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update
from qcopt.neural_bijection.dense.coordinated_explicit_vjp import explicit_coordinated_candidate
from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from tools.coordinated_dense_operator_benchmark import normalized_corners


def retained_storage(function,vertices,proposal):
    """Count only hook handles still retained by the CANDIDATE graph.

    Temporary discarded diagnostic graphs are not counted. Hooks are used in
    an untimed pass; measurements without hooks follow separately.
    """
    class Handle:
        # Saved outputs (e.g. reciprocal) can point back to their own grad_fn.
        # Retain storage/values, NOT that graph link, or the probe creates cycles.
        def __init__(self,tensor):self.tensor=tensor.detach()
    handles=[]
    def pack(tensor):
        handle=Handle(tensor);handles.append(weakref.ref(handle));return handle
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda handle:handle.tensor):
        candidate=function(vertices,proposal)
    retained=[handle() for handle in handles if handle() is not None]
    unique={}
    logical=0
    for handle in retained:
        tensor=handle.tensor;storage=tensor.untyped_storage()
        unique[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
        logical+=tensor.numel()*tensor.element_size()
    result=dict(retained_saved_tensor_count=len(retained),retained_saved_logical_bytes=logical,
        retained_saved_unique_storage_bytes=sum(unique.values()),retained_saved_unique_storage_count=len(unique))
    del candidate,retained
    return result


def benchmark_pair(anchor,reference,mode,amplitude,args):
    if args.warmup<3 or args.repeats<10:raise ValueError("at least3 warmups/10 repeats required")
    device=anchor.device;batch,rows,columns,_=anchor.shape
    yy,xx=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64,device=device),
        torch.linspace(0,1,columns,dtype=torch.float64,device=device),indexing="ij")
    field=amplitude*torch.sin(torch.pi*xx).square()*torch.sin(torch.pi*yy).square()
    field[0]=field[-1]=0;field[:,0]=field[:,-1]=0
    raw=field[None].expand(batch,-1,-1).clone()
    layer=CoordinatedQ1Update((1.,0.),mode=mode,minimum_jacobian=args.minimum_jacobian,theta=args.theta).to(device)
    functions=dict(existing=lambda y,z:layer(y,z,reference=reference,validate=True).vertices,
        explicit=lambda y,z:explicit_coordinated_candidate(y,z,reference=reference,mode=mode,
            minimum_jacobian=args.minimum_jacobian,theta=args.theta,validate=True,
            backward_backend=getattr(args,"backward_backend","autograd")))
    generator=torch.Generator(device=device);generator.manual_seed(args.seed)
    upstream=torch.randn(anchor.shape,generator=generator,device=device,dtype=torch.float64)/anchor.numel()
    sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    # Reference/explicit correctness and active-tie characterization are untimed.
    y=anchor.detach().clone().requires_grad_();z=raw.detach().clone().requires_grad_()
    ordinary=functions["existing"](y,z);manual=functions["explicit"](y,z)
    og=torch.autograd.grad(ordinary,(y,z),upstream)
    mg=torch.autograd.grad(manual,(y,z),upstream)
    difference=float((ordinary-manual).abs().max())
    yerror=float((og[0]-mg[0]).abs().max());zerror=float((og[1]-mg[1]).abs().max())
    tiny=torch.finfo(torch.float64).tiny
    yrelative=float(torch.linalg.vector_norm(og[0]-mg[0])/torch.linalg.vector_norm(og[0]).clamp_min(tiny))
    zrelative=float(torch.linalg.vector_norm(og[1]-mg[1])/torch.linalg.vector_norm(og[1]).clamp_min(tiny))
    yrms=float(og[0].square().mean().sqrt());zrms=float(og[1].square().mean().sqrt())
    if difference!=0 or max(yerror,zerror)>2e-11 or max(yrelative,zrelative)>1e-10:
        raise RuntimeError("candidate/fullY+z VJP equality failed")
    del y,z,ordinary,manual,og,mg
    with torch.no_grad():
        diagnostics=layer(anchor,raw,reference=reference)
        slack,delta,_=layer._constraints(anchor,layer._mask(raw),reference)
        ratios=(-delta).clamp_min(0)/slack
        ties=(ratios==ratios.amax(1)[:,None]).sum(1).cpu().tolist()
        gauge=diagnostics.gauge.cpu().tolist();scale=diagnostics.scale.cpu().tolist()
    del diagnostics,slack,delta,ratios
    results={}
    for method,function in functions.items():
        sync()
        probe_baseline=torch.cuda.memory_allocated(device) if device.type=="cuda" else None
        y=anchor.detach().clone().requires_grad_();z=raw.detach().clone().requires_grad_()
        storage=retained_storage(function,y,z)
        del y,z
        sync()
        probe_after=torch.cuda.memory_allocated(device) if device.type=="cuda" else None
        if probe_after!=probe_baseline:raise RuntimeError("storage probe retained CUDA allocation after counting")
        storage.update(storage_probe_resident_before_bytes=probe_baseline,storage_probe_resident_after_bytes=probe_after)
        forwards=[];vjps=[];residents=[];peaks=[]
        for repeat in range(args.warmup+args.repeats):
            if repeat:del last
            y=anchor.detach().clone().requires_grad_();z=raw.detach().clone().requires_grad_()
            sync()
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(device);resident=torch.cuda.memory_allocated(device)
            else:resident=None
            tick=time.perf_counter();candidate=function(y,z);sync();forward=time.perf_counter()-tick
            tick=time.perf_counter();gradients=torch.autograd.grad(candidate,(y,z),upstream);sync();vjp=time.perf_counter()-tick
            peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
            if not bool(torch.isfinite(candidate).all() and all(torch.isfinite(g).all() for g in gradients)):
                raise RuntimeError("nonfinite candidate/gradient")
            if repeat>=args.warmup:
                forwards.append(forward);vjps.append(vjp);residents.append(resident);peaks.append(peak)
            last=candidate.detach()
            del y,z,candidate,gradients
        qmin=normalized_corners(last,reference)
        if qmin<=args.minimum_jacobian:raise RuntimeError("fresh actual-corner recomputation failed")
        results[method]=dict(**storage,forward_seconds=forwards,vjp_seconds=vjps,
            median_forward_seconds=statistics.median(forwards),median_vjp_seconds=statistics.median(vjps),
            resident_allocated_bytes=residents,peak_total_allocated_bytes=peaks,
            peak_increment_over_resident_bytes=[p-b for p,b in zip(peaks,residents)] if peaks[0] is not None else None,
            actual_normalized_qmin=qmin)
        del last
    return dict(batch=batch,side=rows,mode=mode,amplitude=amplitude,gauge=gauge,scale=scale,
        exact_max_tie_counts=ties,candidate_max_difference=difference,current_y_gradient_max_difference=yerror,
        proposal_gradient_max_difference=zerror,current_y_gradient_relative_l2_error=yrelative,
        proposal_gradient_relative_l2_error=zrelative,reference_current_y_gradient_rms=yrms,
        reference_proposal_gradient_rms=zrms,upstream_rms=float(upstream.square().mean().sqrt()),**results)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    with np.load(args.anchor) as archive:
        base=torch.from_numpy(archive["vertices"].copy()).double()
        reference=torch.from_numpy(archive["boundary_reference"].copy()).double()
        diagonal=str(archive["interpolation"].item()).removeprefix("p1_")
    if base.shape[0]!=1 or base.shape[1]!=base.shape[2] or diagonal not in ("ac","bd"):
        raise ValueError("declared batch1 square P1 anchor required")
    results=[]
    for side in args.sizes:
        if side<base.shape[1] or (side-1)%(base.shape[1]-1):raise ValueError("exactinteger refinement ONLY")
        factor=(side-1)//(base.shape[1]-1)
        one=refine_p1_vertices(base,factor,diagonal).to(device)
        ref=refine_p1_vertices(reference,factor,diagonal).to(device)
        for batch in ([1,4] if side in args.batch4_sizes else [1]):
            anchor=one.expand(batch,-1,-1,-1).clone()
            for mode,amplitude in (("radial",.02),("analytic",.02),("analytic",.0001)):
                result=benchmark_pair(anchor,ref,mode,amplitude,args)
                if mode=="analytic" and ((amplitude==.02 and not all(s<1 for s in result["scale"]))
                        or (amplitude==.0001 and not all(s==1 for s in result["scale"]))):
                    raise RuntimeError("declared active/inactive analytic diagnostic branch not realized")
                results.append(result);print(json.dumps(result),flush=True)
            del anchor
        del one,ref
    payload=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},results=results,
        valid_memory_measurement=True,
        device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        scope="batch1 plus selectedbatch4,geometryfloat64,sameinteger-refinedrealanchor,bothY/zrequiresgrad,sameupstream,candidateonly,validate=True BOTHincludingactualroundedmargin; noimage/optimizer/auxdiagnostic-gradient benchmark",
        storage="retained candidate-graph hook handles measured separately; unique storage includes saved inputY/z, logical tensorbytes maycountviews/duplicates; liveCUDAallocated baseline/peak notreservedcache",
        caveat="explicit first-order only; exactties ALLprocessed; autograd backend chunks use host-sync keep.any, torch_manual does not; forward exact-tie compaction still synchronizes; manyties increase saved rowtable and chunk work; concurrentGPU6 jobs mayaffectCPUlaunch timing; noabsolute crossjob runtimeclaim; newmodule notappintegrated")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sizes",type=int,nargs="+",default=[257,513,1025]);p.add_argument("--batch4-sizes",type=int,nargs="+",default=[257])
    p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=2)
    p.add_argument("--warmup",type=int,default=3);p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--seed",type=int,default=20261001);p.add_argument("--minimum-jacobian",type=float,default=.001)
    p.add_argument("--theta",type=float,default=.95)
    p.add_argument("--backward-backend",choices=("autograd","torch_manual"),default="autograd")
    run(p.parse_args())


if __name__=="__main__":main()
