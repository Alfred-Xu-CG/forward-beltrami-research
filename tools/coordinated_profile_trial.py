"""Split one fixed-stage real trial, with an explicit optional geometry cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import numpy as np

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update,interpolate_proposal
from tools.coordinated_real_case import Evidence,load_registration_evidence,load_image_matches


def profile_one_trial(anchor,evidence,*,level=33,amplitude=.002,warmup=10,repeats=10,decoder_kind="original"):
    if anchor.requires_grad:raise ValueError("fixed stage anchor must not require gradients")
    if anchor.shape[0]!=1 or anchor.dtype!=torch.float64 or level<3 or level>min(anchor.shape[1:3]):
        raise ValueError("batch1 float64 anchor and fitting coefficient level required")
    if warmup<10 or repeats<10:raise ValueError("at least10 warmups/10 timed trials required")
    device=anchor.device
    sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    if decoder_kind not in ("original","stage_cache"):
        raise ValueError("declare original or constant-anchor stage_cache")
    sync();setup_tick=time.perf_counter()
    if decoder_kind=="original":
        layer=CoordinatedQ1Update((1.,0.),mode="analytic",minimum_jacobian=.001,theta=.95).to(device)
    else:
        from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
        layer=FrozenAnchorCoordinatedUpdate(anchor,direction=(1.,0.),mode="analytic",minimum_jacobian=.001,theta=.95)
    sync();decoder_setup_seconds=time.perf_counter()-setup_tick
    axis=torch.linspace(0,1,level,dtype=anchor.dtype,device=device)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    raw=(amplitude*torch.sin(torch.pi*x).square()*torch.sin(torch.pi*y).square())[None,None,1:-1,1:-1]
    def decoder(coefficients):
        coarse=F.pad(coefficients,(1,1,1,1))
        proposal=interpolate_proposal(coarse[:,0],tuple(anchor.shape[1:3]))
        # EXACT current application's path: omitted reference constructs unitgrid,
        # and validate=False still recomputes actual candidate corner margins.
        return layer(anchor,proposal,validate=False) if decoder_kind=="original" else layer(proposal,validate=False)
    times={name:[] for name in ("decoder_forward_seconds","evidence_forward_seconds","combined_vjp_seconds",
        "separate_evidence_vjp_seconds","separate_decoder_vjp_seconds")}
    residents=[];peaks=[];difference=0.
    for repeat in range(warmup+repeats):
        coefficients=raw.detach().clone().requires_grad_()
        sync()
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device);baseline=torch.cuda.memory_allocated(device)
        else:baseline=None
        tick=time.perf_counter();result=decoder(coefficients);sync();d=time.perf_counter()-tick
        tick=time.perf_counter();total,parts=evidence(result.vertices);sync();e=time.perf_counter()-tick
        tick=time.perf_counter();gradient,=torch.autograd.grad(total,coefficients);sync();v=time.perf_counter()-tick
        if not bool(torch.isfinite(gradient).all()):raise RuntimeError("nonfinite combined trial gradient")
        if repeat>=warmup:
            times["decoder_forward_seconds"].append(d);times["evidence_forward_seconds"].append(e)
            times["combined_vjp_seconds"].append(v);residents.append(baseline)
            peaks.append(torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
        objective=float(total.detach());objective_parts={k:float(value.detach()) for k,value in parts.items()}
        margin=float(result.normalized_margin_min.detach().min());scale=float(result.scale.detach().min())
        gauge=float(result.gauge.detach().max());gradient_rms=float(gradient.square().mean().sqrt())
        del result,total,parts,coefficients
        # A SECOND graph separates chain-rule components, outside primary timed
        # trial and peak accounting. Backward stops at candidate, then decoder.
        coefficients=raw.detach().clone().requires_grad_()
        result=decoder(coefficients);total,parts=evidence(result.vertices);sync()
        tick=time.perf_counter();candidate_gradient,=torch.autograd.grad(total,result.vertices);sync()
        ev=time.perf_counter()-tick
        tick=time.perf_counter();split_gradient,=torch.autograd.grad(result.vertices,coefficients,candidate_gradient);sync()
        dv=time.perf_counter()-tick
        difference=max(difference,float((gradient-split_gradient).abs().max()))
        if repeat>=warmup:
            times["separate_evidence_vjp_seconds"].append(ev)
            times["separate_decoder_vjp_seconds"].append(dv)
        del coefficients,result,total,parts,candidate_gradient,split_gradient,gradient
    medians={"median_"+key:statistics.median(value) for key,value in times.items()}
    combined=medians["median_decoder_forward_seconds"]+medians["median_evidence_forward_seconds"]+medians["median_combined_vjp_seconds"]
    geometry_share=(medians["median_decoder_forward_seconds"]+medians["median_separate_decoder_vjp_seconds"])/combined
    return dict(**times,**medians,geometry_cost_share_estimate=geometry_share,
        decoder_kind=decoder_kind,decoder_setup_seconds=decoder_setup_seconds,
        decoder_constant_bytes=layer.resident_constant_bytes if decoder_kind=="stage_cache" else 0,
        share_caution="separately synchronized chain-rule split is diagnostic, not perfectly additive to one fused backward",
        anchor_requires_grad=False,coefficient_parameters=raw.numel(),finite_gradient=True,gradient_rms=gradient_rms,
        chain_rule_gradient_max_difference=difference,objective=objective,objective_parts=objective_parts,
        actual_normalized_margin=margin,scale=scale,gauge=gauge,resident_allocated_bytes=residents,
        peak_total_allocated_bytes=peaks,peak_increment_over_resident_bytes=[p-b for p,b in zip(peaks,residents)] if peaks[0] is not None else None)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    report=json.loads(args.report.read_text());config=report["configuration"]
    with np.load(args.anchor) as archive:
        anchor=torch.from_numpy(archive["vertices"].copy()).double().to(device)
        a=np.asarray(archive["post_affine_matrix"],dtype=np.float32)
        b=np.asarray(archive["post_affine_offset"],dtype=np.float32)
        interpolation=str(archive["interpolation"].item())
    if config["loss"]!="mind" or interpolation not in ("p1_ac","p1_bd"):
        raise ValueError("requested profile uses MIND and actual declared P1 map")
    fixed_path,moving_path=Path(config["fixed"]),Path(config["moving"])
    tick=time.perf_counter()
    fixed,moving,mask,metadata=load_registration_evidence(fixed_path,moving_path,512,
        preprocessing="raw_inverted",device=device,dtype=torch.float32)
    matches,match_metadata=load_image_matches(Path(config["matches"]),a,b,fixed_path=fixed_path,
        moving_path=moving_path,image_side=512,device=device,dtype=torch.float64,robust_scale=8.)
    evidence=Evidence(fixed,moving,torch.from_numpy(a).double().to(device),torch.from_numpy(b).double().to(device),
        "mind",3.,1.,.0001,fixed_mask=mask,interpolation=interpolation,matches=matches,match_weight=.1)
    evidence.prepare_fixed_p1_sampling(*anchor.shape[1:3],dtype=anchor.dtype,device=device)
    if device.type=="cuda":torch.cuda.synchronize(device)
    setup=time.perf_counter()-tick
    result=profile_one_trial(anchor,evidence,level=args.level,amplitude=args.amplitude,warmup=args.warmup,repeats=args.repeats,
        decoder_kind=getattr(args,"decoder_kind","original"))
    payload=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        anchor_source_configuration=config,actual_objective=dict(loss="mind",strain_weight=3.,shape_weight=.0001,
            match_weight=.1,match_robust_scale=8.,geometry_dtype="float64",evidence_dtype="float32",p1_sampling="frozen",
            image_side=512,preprocessing="raw_inverted",interpolation=interpolation),
        setup_seconds=setup,fixed_p1_cache_bytes=sum(t.numel()*t.element_size() for t in evidence.fixed_p1_evaluator.buffers()),
        matches=match_metadata,preprocessing=metadata,profile=result,
        device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        scope="fixed detached real stage anchor, one scalar33 coefficient xdirection analytic trial; no optimizer step/image pyramid/anchor-gradient claim; current app decoder unchanged")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload),flush=True)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("anchor","report","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--level",type=int,default=33);p.add_argument("--amplitude",type=float,default=.002)
    p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=2)
    p.add_argument("--warmup",type=int,default=10);p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--decoder-kind",choices=("original","stage_cache"),default="original")
    run(p.parse_args())


if __name__=="__main__":main()
