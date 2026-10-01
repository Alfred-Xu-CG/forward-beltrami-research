"""Actual nested-control trial profile with the unchanged fine-grid Evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import numpy as np

from qcopt.neural_bijection.dense.coordinated_nested_refinement import FrozenNestedP1Refinement
from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from qcopt.neural_bijection.dense.coordinated_update import interpolate_proposal
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from tools.coordinated_real_case import Evidence,load_registration_evidence,load_image_matches,require_nested_fine_margin
from tools.coordinated_dense_operator_benchmark import normalized_corners


def identity(side,device):
    y,x=torch.meshgrid(torch.linspace(0,1,side,device=device,dtype=torch.float64),
                       torch.linspace(0,1,side,device=device,dtype=torch.float64),indexing="ij")
    return torch.stack((x,y),-1)[None]


def diagnostic_anchor(side,*,device):
    """Rank-one smooth residual with |v dot grad window| <= .017*pi <1.

    This analytical field is sampled ONLY to prepare a tiny legal diagnostic
    anchor, not to reconstruct a real optimizer's coarse state. Its sampled
    four-corner signs are also checked independently before profiling.
    """
    reference=identity(side,device)
    x,y=reference[...,0],reference[...,1]
    window=torch.sin(torch.pi*x).square()*torch.sin(torch.pi*y).square()
    window[:,0]=window[:,-1]=0;window[:,:,0]=window[:,:,-1]=0
    return reference+window[...,None]*reference.new_tensor((.01,-.007))


def profile_evidence_terms(fine_map,evidence,*,warmup=3,repeats=10):
    """VJP ONLY of existing parts, wrt fresh independent fine-vertex leaves.

    Every probe builds the COMPLETE Evidence graph outside its VJP timer. There
    is no hand-written alternative forward loss and no individual forward-time
    attribution. Per-term times are not additive to a combined reverse pass.
    """
    if fine_map.requires_grad or fine_map.ndim!=4 or fine_map.dtype!=torch.float64:
        raise ValueError("constant float64 fine-map snapshot required")
    if warmup<3 or repeats<10:
        raise ValueError("at least3 warmups/10 repeats required")
    weights=dict(image=1.,strain=float(evidence.strain_weight),shape=float(evidence.shape_weight),
                 match=float(evidence.match_weight),oob=float(evidence.oob_weight))
    device=fine_map.device
    sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    allocated=lambda:torch.cuda.memory_allocated(device) if device.type=="cuda" else None
    # Independent fresh graphs for the complete gradient and EACH part; no
    # retain_graph and no reuse of a graph with released saved tensors.
    probe=fine_map.clone().requires_grad_();total,parts=evidence(probe)
    if not set(weights).issubset(parts):
        raise ValueError("per-term probe requires existing image/strain/shape/match/oob parts")
    reference_gradient,=torch.autograd.grad(total,probe)
    values={key:float(parts[key].detach()) for key in weights}
    objective=float(total.detach());del probe,total,parts
    weighted_sum=torch.zeros_like(reference_gradient)
    terms={};combined=None
    for key in (None,*weights):
        times=[];graph_residents=[];initial_residents=[];backward_peaks=[];forward_peaks=[]
        for repeat in range(warmup+repeats):
            probe=fine_map.detach().clone().requires_grad_()
            sync();initial=allocated()
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            total,parts=evidence(probe) # ALL graph construction is outside timer
            sync();resident=allocated()
            forward_peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            tick=time.perf_counter()
            gradient,=torch.autograd.grad(total if key is None else parts[key],probe)
            sync();elapsed=time.perf_counter()-tick
            peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"nonfinite existing Evidence {key} fine-vertex gradient")
            if repeat>=warmup:
                times.append(elapsed);initial_residents.append(initial);graph_residents.append(resident)
                backward_peaks.append(peak);forward_peaks.append(forward_peak)
            if repeat==warmup+repeats-1:
                gradient_rms=float(gradient.square().mean().sqrt())
                if key is not None:
                    weighted_sum.add_(gradient,alpha=weights[key])
            del probe,total,parts,gradient
        row=dict(vjp_seconds=times,median_vjp_seconds=statistics.median(times),
            finite_gradient=True,unweighted_gradient_rms=gradient_rms,
            allocated_before_complete_graph_bytes=initial_residents,
            complete_graph_resident_before_vjp_bytes=graph_residents,
            complete_graph_build_peak_bytes=forward_peaks,
            vjp_peak_total_allocated_bytes=backward_peaks,
            vjp_peak_increment_over_graph_resident_bytes=[p-r for p,r in zip(backward_peaks,graph_residents)]
                if device.type=="cuda" else None)
        if key is None:combined=row
        else:terms[key]=dict(**row,declared_objective_weight=weights[key],part_value=values[key])
    absolute=float((weighted_sum-reference_gradient).abs().max())
    relative=float(torch.linalg.vector_norm(weighted_sum-reference_gradient)/
        torch.linalg.vector_norm(reference_gradient).clamp_min(torch.finfo(reference_gradient.dtype).tiny))
    if relative>1e-9:raise RuntimeError("weighted existing-parts sum disagrees with complete fine gradient")
    prior_time=sum(terms[key]["median_vjp_seconds"] for key in ("strain","shape"))
    return dict(gradient_variable="fine_vertices",fine_shape=list(fine_map.shape),weights=weights,
        objective=objective,terms=terms,combined=combined,
        weighted_sum_gradient_maximum_absolute_error=absolute,weighted_sum_gradient_relative_l2_error=relative,
        reference_complete_gradient_rms=float(reference_gradient.square().mean().sqrt()),
        sum_prior_vjp_seconds=prior_time,
        prior_separate_vjp_ratio_to_complete=prior_time/combined["median_vjp_seconds"],
        timing_scope="fresh COMPLETE Evidence graph perterm built outside VJP timer; unweighted existing parts gradients wrt fine vertices, NOT coefficients; perterm times NOT additive or isolated forward times",
        memory_scope="separate probe graph+backward peaks; all unused parts remain constructed; resident includes reference gradient and weighted-sum correctness buffers; NOT full application peak")


def profile_nested_trial(anchor,evidence,*,fine_side=257,nested=True,
                         coefficient_level=None,amplitude=.002,warmup=3,repeats=10):
    """Constant-anchor scalar stage, NEVER a trainable-anchor decoder benchmark.

    Nested trials use their actual coarse vertex table plus exact cached P1
    materialization. Fixed trials use a fine anchor and bilinear raw proposals.
    These are DIFFERENT parameterizations, not equal candidate functions. All
    losses, priors and frozen matches are evaluated on the SAME fine vertex grid.
    """
    if anchor.requires_grad:
        raise ValueError("constant stage anchor required")
    if anchor.ndim!=4 or anchor.shape[0]!=1 or anchor.shape[-1]!=2 or anchor.shape[1]!=anchor.shape[2] or anchor.dtype!=torch.float64:
        raise ValueError("square batch1 float64 stage anchor required")
    side=anchor.shape[1];level=side if coefficient_level is None else coefficient_level
    if warmup<3 or repeats<10 or level<3 or level>side:
        raise ValueError("fitting coefficient level>=3 and warm3/repeat10 minimum required")
    if not nested and side!=fine_side:
        raise ValueError("fixed-control anchor must have final fine size")
    if evidence.interpolation not in ("p1_ac","p1_bd") or evidence.fixed_p1_evaluator is None:
        raise ValueError("declared P1 fine Evidence with frozen query cache required")
    if (evidence.fixed_p1_evaluator.rows,evidence.fixed_p1_evaluator.columns)!=(fine_side,fine_side):
        raise ValueError("Evidence must retain final fine-grid functional")
    device=anchor.device
    sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    allocated=lambda:torch.cuda.memory_allocated(device) if device.type=="cuda" else None
    fine_reference=identity(fine_side,device)
    reference_corners=q1_corner_determinants(fine_reference)
    coarse_reference=identity(side,device)
    initial_coarse_qmin=normalized_corners(anchor,coarse_reference)
    if initial_coarse_qmin<=.001:
        raise ValueError("diagnostic coarse anchor failed independent actual margin")
    sync();construction_baseline=allocated();tick=time.perf_counter()
    layer=FrozenAnchorCoordinatedUpdate(anchor,direction=(1.,0.),mode="analytic",minimum_jacobian=.001,theta=.95)
    sync();geometry_construction=time.perf_counter()-tick;geometry_after=allocated()
    tick=time.perf_counter()
    materializer=FrozenNestedP1Refinement(side,fine_side,diagonal=evidence.interpolation[-2:],
        dtype=anchor.dtype,device=device) if nested else None
    sync();refinement_construction=time.perf_counter()-tick;refinement_after=allocated()
    yy,xx=torch.meshgrid(torch.linspace(0,1,level,dtype=anchor.dtype,device=device),
                         torch.linspace(0,1,level,dtype=anchor.dtype,device=device),indexing="ij")
    raw=(amplitude*torch.sin(torch.pi*xx).square()*torch.sin(torch.pi*yy).square())[None,None,1:-1,1:-1]

    def decoder(coefficients):
        proposal=interpolate_proposal(F.pad(coefficients,(1,1,1,1))[:,0],(side,side))
        # App's trusted fixed-stage API STILL computes actual coarse output slack.
        return layer(proposal,validate=False)

    def materialize(vertices):
        return materializer(vertices) if materializer is not None else vertices

    def fine_check(vertices,result):
        if nested:
            return require_nested_fine_margin(vertices,reference_corners,.001,"profile trial")
        return result.normalized_margin_min.amin()

    def whole(coefficients):
        result=decoder(coefficients)
        fine=materialize(result.vertices)
        margin=fine_check(fine,result)
        total,parts=evidence(fine)
        if not bool(torch.isfinite(fine).all() and torch.isfinite(total) and margin>0):
            raise RuntimeError("actual fine candidate/objective or configured margin failed")
        return result,fine,margin,total,parts

    times={key:[] for key in ("full_forward_seconds","full_vjp_seconds","full_forward_plus_vjp_seconds",
        "decoder_forward_seconds","materialize_forward_seconds","fine_corner_check_seconds","evidence_forward_seconds",
        "separate_evidence_vjp_seconds","separate_materialize_vjp_seconds","separate_decoder_vjp_seconds")}
    residents=[];peaks=[];absolute_error=0.;relative_error=0.
    initial_fine=materialize(anchor)
    initial_fine_qmin=normalized_corners(initial_fine,fine_reference)
    if initial_fine_qmin<=.001:raise RuntimeError("rounded initial fine anchor failed independent margin")
    del initial_fine
    for repeat in range(warmup+repeats):
        coefficients=raw.detach().clone().requires_grad_()
        sync();baseline=allocated()
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();result,fine,margin,total,parts=whole(coefficients);sync()
        forward=time.perf_counter()-start
        tick=time.perf_counter();gradient,=torch.autograd.grad(total,coefficients);sync()
        backward=time.perf_counter()-tick;complete=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
        if not bool(torch.isfinite(gradient).all()):raise RuntimeError("nonfinite whole fine-objective gradient")
        if repeat>=warmup:
            times["full_forward_seconds"].append(forward);times["full_vjp_seconds"].append(backward)
            times["full_forward_plus_vjp_seconds"].append(complete);residents.append(baseline);peaks.append(peak)
        objective=float(total.detach());objective_parts={k:float(v.detach()) for k,v in parts.items()}
        actual_margin=float(margin.detach());coarse_margin=float(result.normalized_margin_min.detach().min())
        scale=float(result.scale.detach().min());gauge=float(result.gauge.detach().max())
        gradient_rms=float(gradient.square().mean().sqrt())
        if repeat==warmup+repeats-1:last_fine=fine.detach()
        del result,fine,margin,total,parts,coefficients
        # Independent graph/chain split is OUTSIDE primary whole-trial peak/time.
        coefficients=raw.detach().clone().requires_grad_()
        tick=time.perf_counter();result=decoder(coefficients);sync();d=time.perf_counter()-tick
        tick=time.perf_counter();fine=materialize(result.vertices);sync();m=time.perf_counter()-tick
        tick=time.perf_counter();margin=fine_check(fine,result);sync();c=time.perf_counter()-tick
        tick=time.perf_counter();total,parts=evidence(fine);sync();e=time.perf_counter()-tick
        tick=time.perf_counter();fine_gradient,=torch.autograd.grad(total,fine);sync();ev=time.perf_counter()-tick
        if fine is result.vertices:
            coarse_gradient=fine_gradient;mv=0.
        else:
            tick=time.perf_counter();coarse_gradient,=torch.autograd.grad(fine,result.vertices,fine_gradient);sync()
            mv=time.perf_counter()-tick
        tick=time.perf_counter();split_gradient,=torch.autograd.grad(result.vertices,coefficients,coarse_gradient);sync()
        dv=time.perf_counter()-tick
        absolute_error=max(absolute_error,float((gradient-split_gradient).abs().max()))
        relative_error=max(relative_error,float(torch.linalg.vector_norm(gradient-split_gradient)/
            torch.linalg.vector_norm(gradient).clamp_min(torch.finfo(gradient.dtype).tiny)))
        if repeat>=warmup:
            for key,value in zip(("decoder_forward_seconds","materialize_forward_seconds","fine_corner_check_seconds",
                "evidence_forward_seconds","separate_evidence_vjp_seconds","separate_materialize_vjp_seconds",
                "separate_decoder_vjp_seconds"),(d,m,c,e,ev,mv,dv)):
                times[key].append(value)
        del coefficients,result,fine,margin,total,parts,fine_gradient,coarse_gradient,split_gradient,gradient
    if relative_error>1e-9:raise RuntimeError("split full-fine chain-rule gradient disagrees")
    fresh_qmin=normalized_corners(last_fine,fine_reference)
    boundary=all(torch.equal(last_fine[:,edge],fine_reference[:,edge]) for edge in (0,-1))
    boundary=boundary and all(torch.equal(last_fine[:,:,edge],fine_reference[:,:,edge]) for edge in (0,-1))
    if fresh_qmin<=.001 or not boundary:raise RuntimeError("independent final fine margin/boundary failure")
    medians={"median_"+key:statistics.median(values) for key,values in times.items()}
    complete=medians["median_full_forward_plus_vjp_seconds"]
    evidence_share=(medians["median_evidence_forward_seconds"]+medians["median_separate_evidence_vjp_seconds"])/complete
    return dict(status="success",**times,**medians,nested=nested,control_side=side,fine_side=fine_side,
        coefficient_level=level,coefficient_parameters=raw.numel(),raw_physical_amplitude=amplitude,
        geometry_cache_construction_seconds=geometry_construction,geometry_constant_bytes=layer.resident_constant_bytes,
        materializer_construction_seconds=refinement_construction,
        materializer_constant_bytes=materializer.resident_bytes if materializer is not None else 0,
        geometry_cache_resident_increment_bytes=None if construction_baseline is None else geometry_after-construction_baseline,
        materializer_resident_increment_bytes=None if geometry_after is None else refinement_after-geometry_after,
        evidence_cost_share_estimate=evidence_share,
        share_caution="separate synchronized component/backprop split is diagnostic, not perfectly additive to complete trial",
        anchor_source="synthetic legal smooth residual, NOT optimizer trajectory",
        anchor_requires_grad=False,initial_coarse_normalized_qmin=initial_coarse_qmin,
        initial_fine_normalized_qmin=initial_fine_qmin,fresh_actual_normalized_qmin=fresh_qmin,
        actual_fine_margin=actual_margin,coarse_margin=coarse_margin,fine_margin_check_enabled=nested,
        fixed_boundary_equal=boundary,finite_gradient=True,gradient_rms=gradient_rms,
        chain_rule_gradient_max_difference=absolute_error,chain_rule_gradient_relative_l2_error=relative_error,
        objective=objective,objective_parts=objective_parts,scale=scale,gauge=gauge,
        resident_allocated_bytes=residents,peak_total_allocated_bytes=peaks,
        peak_increment_over_resident_bytes=[p-b for p,b in zip(peaks,residents)] if device.type=="cuda" else None)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    config=json.loads(args.report.read_text())["configuration"]
    if (config["loss"]!="mind" or config["strain_weight"]!=3. or config["shape_weight"]!=.0001 or
            config["match_weight"]!=.1 or config["image_precision"]!="float32" or config["precision"]!="float64" or
            config["interpolation"] not in ("p1_ac","p1_bd")):
        raise ValueError("declare SAME calibrated mixed MIND/point.1/strain3/shape1e-4 P1 configuration")
    with np.load(args.affine_archive) as archive:
        a=np.asarray(archive["post_affine_matrix"],dtype=np.float32)
        b=np.asarray(archive["post_affine_offset"],dtype=np.float32)
    fixed_path,moving_path=Path(config["fixed"]),Path(config["moving"])
    tick=time.perf_counter()
    fixed,moving,mask,metadata=load_registration_evidence(fixed_path,moving_path,512,
        preprocessing=config["preprocessing"],device=device,dtype=torch.float32)
    matches,match_metadata=load_image_matches(Path(config["matches"]),a,b,fixed_path=fixed_path,moving_path=moving_path,
        image_side=512,device=device,dtype=torch.float64,robust_scale=config["match_robust_scale"])
    evidence=Evidence(fixed,moving,torch.from_numpy(a).double().to(device),torch.from_numpy(b).double().to(device),
        config["loss"],config["strain_weight"],config["oob_weight"],config["shape_weight"],fixed_mask=mask,
        interpolation=config["interpolation"],matches=matches,match_weight=config["match_weight"])
    evidence.prepare_fixed_p1_sampling(257,257,dtype=torch.float64,device=device)
    if device.type=="cuda":torch.cuda.synchronize(device)
    payload=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        evidence_source_configuration=config,evidence_setup_seconds=time.perf_counter()-tick,
        fixed_pixel_query_cache_bytes=sum(b.numel()*b.element_size() for b in evidence.fixed_p1_evaluator.buffers()),
        matches=match_metadata,preprocessing=metadata,results=[],
        scope="fixed detached SYNTHETIC legal stage anchors with actual real ratkidney frozen512 Evidence; analytic stage_cache, actual nestedcoarse17/65/257; unchanged fine257 priors/pointterm; fixed257 coefficient33 is different parameterization/anchor function, NOT equal-candidate comparison; no optimizer/anatomy result",
        memory="live allocated resident/peak, not reserved; only current coarse cache/refiner plus FULL512Evidence, not all application pyramid caches/optimizer history",
        checks="cache validate=False as app retains actual coarse slack; nested fine actual q/eta/finite guard plus independent untimed NumPy corner/boundary check; no assumed legality of real fine subsets",
        device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU")
    per_term=getattr(args,"per_term",False)
    cases=[] if per_term else [(side,True,side) for side in (17,65,257)]+[(257,False,33)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if per_term:
        # SAME fixed257 syntheticanchor/33 physicalraw proposal as the preserved
        # earlier diagnostic. No rerun of its four whole-trial profiles.
        anchor=diagnostic_anchor(257,device=device)
        layer=FrozenAnchorCoordinatedUpdate(anchor,direction=(1.,0.),mode="analytic",minimum_jacobian=.001,theta=.95)
        yy,xx=torch.meshgrid(torch.linspace(0,1,33,dtype=torch.float64,device=device),
                             torch.linspace(0,1,33,dtype=torch.float64,device=device),indexing="ij")
        coefficients=(args.amplitude*torch.sin(torch.pi*xx).square()*torch.sin(torch.pi*yy).square())[None]
        coefficients[:,0]=coefficients[:,-1]=0;coefficients[:,:,0]=coefficients[:,:,-1]=0
        candidate=layer(interpolate_proposal(coefficients,(257,257)),validate=False)
        fine=candidate.vertices.detach()
        if not bool(torch.isfinite(fine).all() and (candidate.normalized_margin_min>0).all()):
            raise RuntimeError("per-term snapshot failed actual configured margins")
        del anchor,layer,candidate,coefficients,yy,xx
        payload["per_term_probe"]=profile_evidence_terms(fine,evidence,warmup=args.warmup,repeats=args.repeats)
        payload["scope"]+="; optional fixed257 coeff33 perterm ONLY, allparts graph outside backward timers, no wholeprofile rerun"
        args.output.write_text(json.dumps(payload,indent=2)+"\n")
        print(json.dumps(payload["per_term_probe"]),flush=True)
    for side,nested,level in cases:
        try:
            anchor=diagnostic_anchor(side,device=device)
            result=profile_nested_trial(anchor,evidence,fine_side=257,nested=nested,coefficient_level=level,
                amplitude=args.amplitude,warmup=args.warmup,repeats=args.repeats)
            del anchor
        except Exception as error:
            result=dict(status="failure",control_side=side,nested=nested,coefficient_level=level,
                error_type=type(error).__name__,error=str(error))
        payload["results"].append(result)
        args.output.write_text(json.dumps(payload,indent=2)+"\n")
        print(json.dumps(result),flush=True)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("report","affine-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=2)
    p.add_argument("--amplitude",type=float,default=.002);p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--per-term",action="store_true",help="ONLY fixed257/coeff33 per-term Evidence VJPs; default leaves original profile unchanged")
    run(p.parse_args())


if __name__=="__main__":main()
