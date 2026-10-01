"""Paired center/four-quarter image quadrature with the original Adam chart.

Only final x/y suffixes run. Both arms reuse one saved incoming prefix, not a
terminal fiber map. This is an instance-objective ablation, not a new decoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from qcopt.neural_bijection.dense.coordinated_update import interpolate_proposal
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants, validate_q1_map
from tools.coordinated_fiber_suffix import (
    _load_reused_prefix, _make_evidence, _physical_initial_rms, _plain_configuration,
    _save_map, prepare_configuration, reconstruct_transition,
)
from tools.coordinated_real_case import require_nested_fine_margin
from tools.digital_q1_dhr_distill import identity_vertices


def _synchronize(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def _record(total,parts):
    return dict(total=float(total),**{key:float(value) for key,value in parts.items()})


def _check_map(vertices,reference,corners,minimum,context):
    if not validate_q1_map(vertices,reference)["valid"]:
        raise RuntimeError(f"{context}: invalid actual map")
    require_nested_fine_margin(vertices,corners,minimum,context)


def run_arm(incoming,candidates,objective,trial_objective,reference,config,*,layer_factory=None):
    """The original final-stage latent Adam loop, with explicit call scopes."""
    layer_factory=FrozenAnchorCoordinatedUpdate if layer_factory is None else layer_factory
    device,dtype=incoming.device,incoming.dtype
    corners=q1_corner_determinants(reference.double())
    _check_map(incoming,reference,corners,config.minimum_jacobian,"incoming prefix")
    counts={key:0 for key in ("prefix_selection_objective_evaluations",
        "trial_anchor_objective_evaluations","original_anchor_objective_evaluations",
        "trial_objective_evaluations","acceptance_objective_evaluations",
        "stage_selection_objective_evaluations","backward_attempts","gradient_steps","failed_trials")}
    def evaluate(function,vertices,kind):
        counts[kind+"_objective_evaluations"]+=1
        return function(vertices)
    prefix_records=[]
    best,best_loss,selected=None,float("inf"),None
    _synchronize(device); setup_tick=time.perf_counter()
    for name,vertices in candidates.items():
        _check_map(vertices,reference,corners,config.minimum_jacobian,"candidate "+name)
        _synchronize(device); tick=time.perf_counter()
        with torch.no_grad():total,parts=evaluate(objective,vertices,"prefix_selection")
        _synchronize(device)
        value=float(total)
        if not np.isfinite(value):raise RuntimeError("nonfinite prefix candidate objective")
        prefix_records.append(dict(candidate=name,seconds=time.perf_counter()-tick,**_record(total,parts)))
        if value<best_loss:
            best,best_loss=vertices.detach().clone(),value
            selected=dict(kind="prefix",candidate=name,stage=None)
    if best is None:raise ValueError("nonempty common prefix candidate family required")
    _synchronize(device); candidate_setup_seconds=time.perf_counter()-setup_tick
    current=incoming.detach().clone()
    incoming_equal=torch.equal(current,incoming)
    stages,trace,failures=[],[],[]
    forwards,backwards=[],[]
    physical_lr=_physical_initial_rms(config)
    _synchronize(device); tick=time.perf_counter()
    for direction in ((1.,0.),(0.,1.)):
        stage_tick=time.perf_counter()
        anchor=current.detach()
        coefficients=torch.nn.Parameter(torch.zeros(1,1,config.grid_side-2,config.grid_side-2,
            device=device,dtype=dtype))
        optimizer=torch.optim.Adam([coefficients],lr=physical_lr)
        layer=layer_factory(anchor,reference=reference,direction=direction,mode="analytic",
            minimum_jacobian=config.minimum_jacobian,theta=.95)
        with torch.no_grad():
            anchor_loss=float(evaluate(trial_objective,anchor,"trial_anchor")[0])
            original_anchor_loss=float(evaluate(objective,anchor,"original_anchor")[0])
        if not np.isfinite([anchor_loss,original_anchor_loss]).all():
            raise RuntimeError("nonfinite original stage anchor objective")
        stage_best,stage_best_loss=anchor,anchor_loss
        initial_coefficient_max_abs=float(coefficients.detach().abs().max())
        stage_gradients=0
        for step in range(config.inner_steps+1):
            optimizer.zero_grad(set_to_none=True)
            _synchronize(device); forward_tick=time.perf_counter()
            padded=F.pad(coefficients,(1,1,1,1))
            proposal=interpolate_proposal(padded[:,0],(config.grid_side,config.grid_side))
            result=layer(proposal,validate=False)
            candidate=result.vertices
            # Inspect actual rounded coordinates, not the layer's reported slack.
            margin=(q1_corner_determinants(candidate.detach().double())/corners-config.minimum_jacobian).amin()
            total,parts=evaluate(trial_objective,candidate,"trial")
            _synchronize(device); forwards.append(time.perf_counter()-forward_tick)
            value=float(total.detach())
            diagnostics=dict(scale=float(result.scale.detach().min()),gauge=float(result.gauge.detach().max()),
                actual_margin=float(margin),reported_margin=float(result.normalized_margin_min.detach().min()),
                raw_coefficient_rms=float(coefficients.detach().square().mean().sqrt()))
            legal=bool(torch.isfinite(candidate).all()) and np.isfinite(float(margin)) and float(margin)>0
            if not np.isfinite(value) or not legal:
                counts["failed_trials"]+=1
                failures.append(dict(direction=direction,step=step,
                    reason="nonfinite objective/coordinates or rounded margin",**diagnostics))
                break
            if value<stage_best_loss:
                stage_best,stage_best_loss=candidate.detach().clone(),value
            trace.append(dict(direction=direction,step=step,**_record(total.detach(),
                {key:part.detach() for key,part in parts.items()}),**diagnostics))
            if step<config.inner_steps:
                counts["backward_attempts"]+=1
                backward_tick=time.perf_counter();total.backward();_synchronize(device)
                backwards.append(time.perf_counter()-backward_tick)
                if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                    counts["failed_trials"]+=1
                    failures.append(dict(direction=direction,step=step,reason="missing/nonfinite gradient"))
                    break
                optimizer.step();counts["gradient_steps"]+=1;stage_gradients+=1
        with torch.no_grad():proposed_full=float(evaluate(objective,stage_best,"acceptance")[0])
        rejected=not np.isfinite(proposed_full) or proposed_full>original_anchor_loss
        current=anchor if rejected else stage_best
        _check_map(current,reference,corners,config.minimum_jacobian,"accepted x/y stage")
        with torch.no_grad():accepted_full=float(evaluate(objective,current,"stage_selection")[0])
        if not np.isfinite(accepted_full):raise RuntimeError("nonfinite accepted complete objective")
        if accepted_full<best_loss:
            best,best_loss=current.detach().clone(),accepted_full
            selected=dict(kind="accepted_stage",candidate=None,stage=len(stages))
        _synchronize(device)
        stages.append(dict(direction=direction,physical_lr=physical_lr,
            initial_coefficient_max_abs=initial_coefficient_max_abs,gradient_steps=stage_gradients,
            anchor_reduced_total=anchor_loss,anchor_total=original_anchor_loss,
            proposed_reduced_total=stage_best_loss,proposed_original_total=proposed_full,
            rounded_full_stage_fallback=rejected,accepted_full_total=accepted_full,
            seconds=time.perf_counter()-stage_tick))
    _synchronize(device)
    suffix_seconds=time.perf_counter()-tick
    counts["complete_objective_evaluations"]=sum(value for key,value in counts.items()
        if key.endswith("_objective_evaluations"))
    _check_map(best,reference,corners,config.minimum_jacobian,"selected output")
    return best,current,dict(selected=selected,selected_total=best_loss,counts=counts,
        stages=stages,trace=trace,failures=failures,prefix_candidates=prefix_records,
        incoming_prefix_bitwise_equal=incoming_equal,prefix_selection_seconds=candidate_setup_seconds,
        first_complete_objective_seconds=prefix_records[0]["seconds"],suffix_seconds=suffix_seconds,
        first_trial_forward_seconds=forwards[0] if forwards else None,
        subsequent_trial_forward_mean_seconds=float(np.mean(forwards[1:])) if len(forwards)>1 else None,
        forward_seconds=sum(forwards),backward_seconds=sum(backwards),
        solver_status="failed_trial_valid_best_retained" if failures else "budget_reached",
        chart="FrozenAnchorCoordinatedUpdate analytic theta=.95; fixed boundary; zero interior scalar Adam per x/y stage",
        budget="inner_steps Adam gradients plus last evaluated trial per direction; actual counts reported")


def run(args,*,production=True):
    if args.output.suffix!=".json":raise ValueError("paired output must be .json")
    stem=args.output.with_suffix("")
    outputs={name:stem.with_name(stem.name+"_"+name+".npz") for name in ("centers","quarters")}
    prefix_path=stem.with_name(stem.name+"_prefix.npz")
    for path in [args.output,prefix_path,*outputs.values(),*[p.with_suffix(".json") for p in outputs.values()]]:
        if path.exists():raise FileExistsError(path)
    cold_tick=time.perf_counter()
    previous=json.loads(args.reuse_prefix_report.read_text(encoding="utf-8"))
    grid_side=1025 if production else previous["final_control_side"]
    baseline,baseline_path,source_prefix,capture,payload,_=_load_reused_prefix(args.reuse_prefix_report,grid_side)
    config=prepare_configuration(baseline,outputs["centers"],grid_side,production=production)
    if (config.loss!="mind" or getattr(config,"mind_order","transport")!="transport" or
            getattr(config,"geometry_backend","existing")!="stage_cache"):
        raise ValueError("quadrature comparison requires MIND transport and original stage_cache analytic chart")
    if production and (config.grid_side!=1025 or config.lr_calibration!="edge" or config.learning_rate!=.004):
        raise ValueError("production quadrature uses1025 controls and original .004 edge learning rate")
    if isinstance(config.threads,bool) or not isinstance(config.threads,int) or config.threads<=0:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(config.threads)
    device=torch.device(config.device)
    dtype={"float32":torch.float32,"float64":torch.float64}[config.precision]
    _synchronize(device); loaded_seconds=time.perf_counter()-cold_tick
    evidence_tick=time.perf_counter()
    base,matrix,offset,evidence_metadata=_make_evidence(config,device,dtype)
    reference=identity_vertices(grid_side,device=device).to(dtype)
    for key,actual in (("post_affine_matrix",matrix),("post_affine_offset",offset),
            ("boundary_reference",reference.cpu().numpy())):
        if not np.array_equal(payload[key],actual):raise ValueError("saved prefix "+key+" differs from evidence")
    incoming,coarse,transition=reconstruct_transition(capture.prefix,config.levels[-2],
        device=device,dtype=dtype,diagonal=config.interpolation[-2:])
    corners=q1_corner_determinants(reference.double())
    _check_map(incoming,reference,corners,config.minimum_jacobian,"shared incoming prefix")
    candidates=dict(initial=reference,stored_e1_best_prefix=capture.best_prefix.to(device=device,dtype=dtype),
        incoming_prefix=incoming)
    center_trial=base.coarse_nested_evidence(grid_side,grid_side,dtype=dtype,device=device)
    _synchronize(device); common_setup_seconds=time.perf_counter()-evidence_tick
    from tools.coordinated_pixel_quadrature import FourQuarterMindEvidence
    quarter_tick=time.perf_counter()
    quarters=FourQuarterMindEvidence(base,grid_side,dtype=dtype,device=device)
    quarter_trial=FourQuarterMindEvidence(center_trial,grid_side,dtype=dtype,device=device)
    _synchronize(device); quarter_setup_seconds=time.perf_counter()-quarter_tick
    objectives=dict(centers=base,quarters=quarters)
    trials=dict(centers=center_trial,quarters=quarter_trial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    prefix_certificate=_save_map(prefix_path,incoming,reference,matrix,offset,config.interpolation,
        coarse_vertices=coarse.cpu().numpy())
    arms,maps={},{}
    for name in ("centers","quarters"):
        _synchronize(device)
        resident=torch.cuda.memory_allocated(device) if device.type=="cuda" else None
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        best,terminal,record=run_arm(incoming,candidates,objectives[name],trials[name],reference,config)
        _synchronize(device)
        peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
        save_tick=time.perf_counter()
        certificate=_save_map(outputs[name],best,reference,matrix,offset,config.interpolation,
            terminal_vertices=terminal.cpu().numpy())
        maps[name]=best.detach().cpu().clone()
        record.update(output=str(outputs[name]),saved_binary_certificate=certificate,
            objective_name="E1_centers" if name=="centers" else "E4_four_quarters",
            image_quadrature=("one original pixel center, original mask denominator D" if name=="centers" else
                "four quarter queries of ORIGINAL fixed/moving descriptors; original pixel mask repeated, denominator4D; ORIGINAL center OOB unchanged"),
            export_certificate_seconds=time.perf_counter()-save_tick,
            resident_before_arm_bytes=resident,peak_allocated_bytes=peak,
            incremental_peak_bytes=None if peak is None else peak-resident,
            moving_descriptor_interpolations_per_objective=1 if name=="centers" else 4)
        arms[name]=record
        del best,terminal
    cross={}
    _synchronize(device); cross_tick=time.perf_counter()
    for map_name,vertices_cpu in maps.items():
        vertices=vertices_cpu.to(device=device,dtype=dtype)
        cross[map_name]={}
        for objective_name,objective in objectives.items():
            with torch.no_grad():total,parts=objective(vertices)
            if not np.isfinite(float(total)):raise RuntimeError("nonfinite cross objective")
            cross[map_name][objective_name]=_record(total,parts)
    _synchronize(device); cross_seconds=time.perf_counter()-cross_tick
    for name,arm in arms.items():
        arm["cross_objective_evaluations"]=2
        arm["moving_descriptor_interpolation_equivalents"]=arm["moving_descriptor_interpolations_per_objective"]*(
            arm["counts"]["complete_objective_evaluations"]+2)
        arm["final"]=cross[name][name]
        arm["configuration"]=_plain_configuration(config)
        arm["configuration"]["output"]=str(outputs[name])
        outputs[name].with_suffix(".json").write_text(json.dumps(arm,indent=2)+"\n",encoding="utf-8")
    report=dict(source_report=str(args.reuse_prefix_report),source_baseline=str(baseline_path),
        source_prefix=str(source_prefix),shared_prefix_output=str(prefix_path),
        baseline_executed_this_run=False,shared_incoming_prefix_bitwise_equal=all(
            arm["incoming_prefix_bitwise_equal"] for arm in arms.values()),**transition,
        prefix_certificate=prefix_certificate,configuration=_plain_configuration(config),
        candidate_prefix_family=list(candidates),arms=arms,cross_objectives=cross,
        prefix_load_seconds=loaded_seconds,common_evidence_setup_seconds=common_setup_seconds,
        quarter_evidence_setup_seconds=quarter_setup_seconds,cross_objective_seconds=cross_seconds,
        quarter_cache_resident_bytes=quarters.resident_constant_bytes+quarter_trial.resident_constant_bytes,
        quarter_cache_metadata=dict(selection=quarters.metadata,trial=quarter_trial.metadata),
        fixed_descriptor_interpolations_setup=8,quarter_wrapper_instances=2,
        complete_objective_evaluations=sum(arm["counts"]["complete_objective_evaluations"] for arm in arms.values())+4,
        moving_descriptor_interpolation_equivalents=sum(arm["moving_descriptor_interpolation_equivalents"] for arm in arms.values()),
        image_quadrature="E1 one center query; E4 four quarter queries with inherited original pixel mask and denominator4*sum(mask); original center OOB unchanged",
        selection="each arm minimizes its OWN complete objective over identical initial/storedE1best/incoming prefix family and accepted x/y stages; no cross-objective raw-total ranking",
        timing_scope="sequential centers then quarters; input load, shared evidence setup, quarter query/fixed-descriptor setup, per-arm three-candidate selection, suffix, export/certificate and cross-evaluation reported separately. First complete objective is that arm's first setup candidate (not a fresh process); first trial is already objective-warm. Subsequent forward mean excludes first trial; suffix includes Adam/cache construction, all anchor/trial/acceptance/selection and actual-map checks. No historical baseline speed comparison.",
        memory_scope="CUDA allocated bytes only; both objective and trial caches plus common prefix candidates resident before each arm; peak resets per arm, records resident and incremental peak; CPU peak unavailable",
        scope="fixed-prefix instance image-quadrature ablation; no claim of new image information or neural generalization",
        landmarks_used=False,**evidence_metadata)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-prefix-report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    result=run(parser.parse_args())
    print(json.dumps(result))


if __name__=="__main__":main()
