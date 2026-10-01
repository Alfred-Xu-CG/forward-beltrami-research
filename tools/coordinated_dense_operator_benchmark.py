"""Operator-only forward/VJP scaling on one exactly integer-refined real P1 map.

No image objective, optimizer, landmarks, field teacher or exact-sign certificate
is timed. All inputs represent the same .02-unit smooth x proposal. Existing
F1/F2 logits are calibrated by their IDENTITY infinitesimal transfer, without
inverting tanh to pretend that finite proposals are attainable on a fine grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import statistics

import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries
from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update
from qcopt.neural_bijection.dense.digital_q1 import AdaptiveSoftRadialQ1Relaxation, StaggeredPatchQ1Layer
from tools.coordinated_control_capacity import decode, f2_coverage


def normalized_corners(v, reference):
    """Independent NumPy generic triangle determinants; excluded from timing."""
    def determinants(z):
        z=z.detach().cpu().numpy().astype(np.float64)
        a,b,c,d=z[:,:-1,:-1],z[:,:-1,1:],z[:,1:,1:],z[:,1:,:-1]
        det=lambda p,q,r:(q-p)[...,0]*(r-p)[...,1]-(q-p)[...,1]*(r-p)[...,0]
        return np.stack([det(*t) for t in ((a,b,d),(a,b,c),(d,b,c),(a,c,d))],-1)
    return float((determinants(v)/determinants(reference)).min())


def physical_proposal(side, amplitude, device):
    axis=torch.linspace(0,1,side,device=device,dtype=torch.float64)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    u=amplitude*torch.sin(torch.pi*x).square()*torch.sin(torch.pi*y).square()
    u[0]=u[-1]=0;u[:,0]=u[:,-1]=0
    return u[None]


def operator(method,side,args,device):
    if method in ("radial","analytic"):
        layer=CoordinatedQ1Update((1.,0.),mode=method,minimum_jacobian=args.minimum_jacobian,theta=.95)
    elif method=="f1":
        layer=AdaptiveSoftRadialQ1Relaxation(side,raw_span=8.,safety_fraction=.75,minimum_jacobian=args.minimum_jacobian)
    else:
        layer=StaggeredPatchQ1Layer(side,args.patch_cells,proposal_mode="fixed_h",raw_span=.5,
            safety_fraction=.75,minimum_jacobian=args.minimum_jacobian,accepted_gain=args.f2_accepted_gain)
    layer=layer.to(device=device,dtype=torch.float64)
    coverage=f2_coverage(layer,side,torch.float64) if method=="f2" else None
    return layer,coverage


def evaluate(layer,method,anchor,coefficients,coverage,args):
    # decode's F2 nominal deliberately excludes gain; compensate ONCE here.
    coefficients=coefficients/args.f2_accepted_gain if method=="f2" else coefficients
    return decode(layer,method,anchor,coefficients,coverage)


def identity_calibration(args):
    side=17;axis=torch.linspace(0,1,side,dtype=torch.float64)
    y,x=torch.meshgrid(axis,axis,indexing="ij");reference=torch.stack((x,y),-1)[None]
    u=physical_proposal(side,1.,"cpu");result={}
    for method in args.methods:
        layer,coverage=operator(method,side,args,"cpu")
        channels=1 if method in ("radial","analytic") else 2
        tangent=torch.zeros(1,channels,side-2,side-2,dtype=torch.float64)
        tangent[:,0]=u[:,1:-1,1:-1]
        zero=torch.zeros_like(tangent)
        function=lambda c:evaluate(layer,method,reference,c,coverage,args)
        _,jvp=torch.autograd.functional.jvp(function,zero,tangent)
        expected=torch.stack((u,torch.zeros_like(u)),-1)
        error=float((jvp-expected).abs().max())
        if error>1e-12:raise RuntimeError(f"{method} identity physical JVP failed: {error}")
        result[method]=dict(side=side,identity_jvp_max_error=error,
            finite_amplitude_not_identity_jvp="measured separately on actual current geometry")
    return result


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    with np.load(args.anchor) as archive:
        base=torch.from_numpy(archive["vertices"].copy()).double()
        reference=torch.from_numpy(archive["boundary_reference"].copy()).double()
        declared=str(archive["interpolation"].item())
    if declared!="p1_"+args.diagonal or base.shape[0]!=1 or base.shape[1]!=base.shape[2]:
        raise ValueError("declare batch1 square P1 anchor with the SAME stored diagonal")
    base_side=base.shape[1]
    if base.shape!=reference.shape or args.repeats<1 or args.warmup<0 or args.amplitude<=0:
        raise ValueError("matching reference and positive bounded benchmark parameters required")
    if normalized_corners(base,reference)<=args.minimum_jacobian:
        raise ValueError("actual base anchor must strictly exceed the declared margin")
    calibration=identity_calibration(args)
    sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    rows=[]
    for side in args.sizes:
        if side<base_side or (side-1)%(base_side-1):
            raise ValueError("only exact uniform integer refinements of the SAME input map")
        factor=(side-1)//(base_side-1)
        tick=time.perf_counter()
        anchor=refine_p1_vertices(base,factor,args.diagonal).to(device)
        ref=refine_p1_vertices(reference,factor,args.diagonal).to(device)
        sync();refinement_seconds=time.perf_counter()-tick
        anchor_qmin=normalized_corners(anchor,ref)
        if anchor_qmin<=args.minimum_jacobian:raise RuntimeError("rounded refined anchor lost required margin")
        query=torch.from_numpy(np.random.default_rng(args.seed).uniform(0,1,(1,257,2))).double()
        difference=float((p1_map_at_queries(anchor.cpu(),query,args.diagonal)-p1_map_at_queries(base,query,args.diagonal)).abs().max())
        if difference>2e-12:raise RuntimeError("rounded refined P1 function disagrees at diagnostic queries")
        u=physical_proposal(side,args.amplitude,device)
        generator=torch.Generator(device=device);generator.manual_seed(args.seed)
        upstream=torch.randn(anchor.shape,generator=generator,device=device,dtype=torch.float64)/anchor.numel()
        requested_rms=float(u.square().mean().sqrt())
        for method in args.methods:
            sync();tick=time.perf_counter();layer,coverage=operator(method,side,args,device);sync()
            cold_setup_seconds=time.perf_counter()-tick
            channels=1 if method in ("radial","analytic") else 2
            values=torch.zeros(1,channels,side-2,side-2,device=device,dtype=torch.float64)
            values[:,0]=u[:,1:-1,1:-1]
            forwards=[];backwards=[];baseline=[];peaks=[]
            for repeat in range(args.warmup+args.repeats):
                if repeat:
                    del last  # Prior output is not an operator input/resident buffer.
                coefficients=values.detach().clone().requires_grad_()
                sync()
                if device.type=="cuda":
                    torch.cuda.reset_peak_memory_stats(device);resident=torch.cuda.memory_allocated(device)
                else:resident=None
                tick=time.perf_counter();candidate=evaluate(layer,method,anchor,coefficients,coverage,args);sync()
                forward_seconds=time.perf_counter()-tick
                tick=time.perf_counter();gradient,=torch.autograd.grad(candidate,coefficients,grad_outputs=upstream);sync()
                backward_seconds=time.perf_counter()-tick
                if not bool(torch.isfinite(candidate).all() and torch.isfinite(gradient).all()):raise RuntimeError("nonfinite operator/VJP")
                if repeat>=args.warmup:
                    forwards.append(forward_seconds);backwards.append(backward_seconds);baseline.append(resident)
                    peaks.append(torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
                last=candidate.detach();gradient_rms=float(gradient.square().mean().sqrt())
                del candidate,gradient,coefficients
            actual_qmin=normalized_corners(last,ref)
            if actual_qmin<=args.minimum_jacobian:raise RuntimeError("rounded finite-proposal output failed margins")
            motion=last-anchor
            row=dict(side=side,refinement_factor=factor,method=method,vertices=side**2,constraints=4*(side-1)**2,
                proposal_parameters=values.numel(),anchor_normalized_qmin=anchor_qmin,output_normalized_qmin=actual_qmin,
                requested_peak_unit=args.amplitude,requested_rms_unit=requested_rms,
                actual_motion_rms_unit=float(motion.square().sum(-1).mean().sqrt()),
                finite_amplitude_rms_ratio=float(motion.square().sum(-1).mean().sqrt())/requested_rms,
                actual_motion_peak_unit=float(motion.square().sum(-1).sqrt().max()),
                gradient_rms=gradient_rms,median_forward_seconds=statistics.median(forwards),
                median_vjp_seconds=statistics.median(backwards),forward_seconds=forwards,vjp_seconds=backwards,
                cold_layer_setup_seconds=cold_setup_seconds,refinement_seconds=refinement_seconds,
                refined_function_query_max_difference=difference,resident_allocated_bytes=baseline,
                peak_total_allocated_bytes=peaks,peak_increment_over_resident_bytes=[p-b for p,b in zip(peaks,baseline)] if peaks[0] is not None else None)
            rows.append(row);print(json.dumps(row),flush=True)
            del layer,coverage,values,last,motion
        del anchor,ref,u,upstream
    payload=dict(question=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        identity_calibration=calibration,results=rows,
        scope="proposal-only VJP, fixed detached anchor; batch1 geometry float64; setup/refinement/corner checks excluded from operator timing",
        memory="CUDA live allocated baseline plus graph peak; reserved caching allocator memory is not allocated memory",
        caution="F1/F2 two-component vs coordinated scalar proposal; identity tangent calibration is not equal finite motion; no image runtime or registration superiority claim")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sizes",type=int,nargs="+",default=[257,513,1025])
    p.add_argument("--methods",nargs="+",choices=("radial","analytic","f1","f2"),default=["radial","analytic","f1","f2"])
    p.add_argument("--diagonal",choices=("ac","bd"),default="ac")
    p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=2)
    p.add_argument("--amplitude",type=float,default=.02);p.add_argument("--minimum-jacobian",type=float,default=.001)
    p.add_argument("--patch-cells",type=int,default=8);p.add_argument("--f2-accepted-gain",type=float,default=.75)
    p.add_argument("--repeats",type=int,default=3);p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--seed",type=int,default=20261001)
    run(p.parse_args())


if __name__=="__main__":main()
