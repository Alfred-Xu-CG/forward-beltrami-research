"""Diagnostic: replace only the last x/y Adam stages by physical-fiber L-BFGS.

By default the baseline is rerun once; --reuse-prefix-report instead loads its
saved incoming prefix without rerunning it. This is an instance optimizer comparison, not a
new neural decoder. No anatomical evaluation data enter this executable.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants, validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.coordinated_real_case import (
    Evidence, load_image_matches, load_registration_evidence, optimize,
    require_nested_fine_margin,
)
from tools.digital_q1_dhr_distill import identity_vertices


def prepare_configuration(report, output, grid_side, *, production=True):
    """Preserve the supplied objective; explicitly replace only the final size."""
    config=copy.deepcopy(report["configuration"])
    requirements=dict(method="analytic",control_hierarchy="nested_p1",
        nested_evaluation="coarse_exact",p1_sampling="frozen",output_selection="best_full")
    for name,value in requirements.items():
        if config.get(name)!=value:
            raise ValueError(f"source configuration requires {name}={value}")
    if (config.get("coordinate_mode","alternating")!="alternating" or
            config.get("fine_patch_cells",0)!=0 or config.get("proposal_filter_steps",0)!=0 or
            config.get("fine_patch_backend","ordinary")!="ordinary" or
            config.get("cycles")!=1 or config.get("interpolation") not in ("p1_ac","p1_bd")):
        raise ValueError("one alternating global P1 cycle without patches or filtering required")
    levels=config.get("levels",[])
    if (len(levels)<2 or isinstance(grid_side,bool) or not isinstance(grid_side,int) or
            grid_side<=levels[-2] or (grid_side-1)%(levels[-2]-1)):
        raise ValueError("final control size must refine the penultimate control mesh")
    if production and (grid_side not in (257,1025) or levels[:-1]!=[17,33,65,129] or
            config.get("image_side")!=1024 or config.get("image_levels")!=[64,128,256,512,1024] or
            config.get("inner_steps")!=30):
        raise ValueError("production requires 257/1025 controls, native1024 continuation and30 Adam steps")
    if config.get("lr_calibration") not in ("edge","physical"):
        raise ValueError("declared edge/physical learning-rate calibration required")
    if not np.isfinite(config.get("learning_rate",np.nan)) or config["learning_rate"]<=0:
        raise ValueError("finite positive learning_rate required")
    config["grid_side"]=grid_side
    config["levels"]=levels[:-1]+[grid_side]
    config["output"]=Path(output)
    for key in ("fixed","moving","affine","matches"):
        if config.get(key) is not None:config[key]=Path(config[key])
    return argparse.Namespace(**config)


def _plain_configuration(config):
    return {key:str(value) if isinstance(value,Path) else value for key,value in vars(config).items()}


def _physical_initial_rms(config):
    return (config.learning_rate*(config.levels[0]-1)/(config.levels[-1]-1)
            if config.lr_calibration=="edge" else config.learning_rate)


def reconstruct_transition(snapshot, prefix_side, *, device, dtype, diagonal):
    """Use the baseline transition kernel on its exactly retained old nodes.

    The callback materializes with a frozen evaluator, whose summation order
    can differ from refine_p1_vertices. Non-old-node snapshot equality is not
    required to reconstruct the baseline's actual next-stage anchor.
    """
    fine_side=snapshot.shape[1]
    stride=(fine_side-1)//(prefix_side-1)
    old_nodes=snapshot[:,::stride,::stride]
    coarse=old_nodes.to(device=device,dtype=dtype).clone()
    if not torch.equal(coarse.detach().cpu(),old_nodes):
        raise RuntimeError("prefix old-node transfer changed stored values")
    current=refine_p1_vertices(coarse,stride,diagonal).detach()
    if not torch.equal(current[:,::stride,::stride].detach().cpu(),old_nodes):
        raise RuntimeError("baseline transition failed exact old-node retention")
    actual=current.detach().cpu()
    metadata=dict(shared_transition_from_exact_oldnodes=True,
        shared_transition_scope="baseline transition reconstructed from exact copied old nodes using the same refine_p1_vertices function, dtype and device; callback fine materialization may use different summation",
        snapshot_bitwise_equal=torch.equal(actual,snapshot),
        frozen_materialization_max_difference=float((actual.double()-snapshot.double()).abs().max()))
    return current,coarse,metadata


class PrefixCapture:
    """Capture fine vertex tables only up to the requested shared prefix."""
    def __init__(self, prefix_side, final_side):
        self.prefix_side,self.final_side=prefix_side,final_side
        self.prefix=None
        self.best_prefix=None
        self.best_prefix_objective=float("inf")
        self.best_prefix_stage=None
        self.prefix_elapsed=None
        self.last_elapsed=None
        self.callback_seconds=0.
        self.prefix_callback_seconds=0.
        self.stages=[]

    def __call__(self, vertices, stage, elapsed):
        tick=time.perf_counter()
        if vertices.shape!=(1,self.final_side,self.final_side,2):
            raise ValueError("callback must provide the materialized fine map")
        self.stages.append(dict(stage))
        self.last_elapsed=elapsed
        if stage["control_side"]<=self.prefix_side:
            value=stage["accepted_full_total"]
            if not np.isfinite(value):raise ValueError("nonfinite accepted prefix objective")
            if value<self.best_prefix_objective:
                self.best_prefix=vertices.detach().cpu().clone()
                self.best_prefix_objective=value
                self.best_prefix_stage=len(self.stages)-1
            if stage["control_side"]==self.prefix_side and tuple(stage["direction"])==(0.,1.):
                self.prefix=vertices.detach().cpu().clone()
                self.prefix_elapsed=elapsed
        duration=time.perf_counter()-tick
        self.callback_seconds+=duration
        if stage["control_side"]<=self.prefix_side:self.prefix_callback_seconds+=duration


def _make_evidence(config, device, dtype):
    image_precision=getattr(config,"image_precision","same")
    image_dtype=dtype if image_precision=="same" else {"float32":torch.float32,"float64":torch.float64}[image_precision]
    with np.load(config.affine,allow_pickle=False) as archive:
        matrix=np.asarray(archive["post_affine_matrix"],dtype=np.float32)
        offset=np.asarray(archive["post_affine_offset"],dtype=np.float32)
    if (matrix.shape!=(2,2) or offset.shape!=(2,) or not np.isfinite(matrix).all() or
            not np.isfinite(offset).all() or np.linalg.det(matrix.astype(np.float64))<=0):
        raise ValueError("finite positive supplied affine required")
    fixed,moving,mask,preprocessing=load_registration_evidence(
        config.fixed,config.moving,config.image_side,
        preprocessing=getattr(config,"preprocessing","raw_inverted"),device=device,dtype=image_dtype)
    matches,match_metadata=None,None
    match_weight=getattr(config,"match_weight",0.)
    if match_weight:
        matches,match_metadata=load_image_matches(config.matches,matrix,offset,
            fixed_path=config.fixed,moving_path=config.moving,image_side=config.image_side,
            device=device,dtype=dtype,robust_scale=getattr(config,"match_robust_scale",8.))
    evidence=Evidence(fixed,moving,torch.from_numpy(matrix).to(device=device,dtype=dtype),
        torch.from_numpy(offset).to(device=device,dtype=dtype),config.loss,
        config.strain_weight,config.oob_weight,getattr(config,"shape_weight",0.),fixed_mask=mask,
        interpolation=config.interpolation,matches=matches,match_weight=match_weight,
        strain_model=getattr(config,"strain_model","displacement_gradient"),
        mind_order=getattr(config,"mind_order","transport"))
    evidence.prepare_fixed_p1_sampling(config.grid_side,config.grid_side,dtype=dtype,device=device)
    return evidence,matrix,offset,dict(image_preprocessing=preprocessing,image_match_evidence=match_metadata)


def _save_map(path, vertices, reference, matrix, offset, interpolation, **extra):
    np.savez(path,vertices=vertices.detach().cpu().numpy(),
        boundary_reference=reference.detach().cpu().numpy(),post_affine_matrix=matrix,
        post_affine_offset=offset,interpolation=np.asarray(interpolation),**extra)
    certificate=certify_q1_binary_map(path)
    if not certificate["valid"]:raise RuntimeError(f"saved map failed exact certificate: {path}")
    return certificate


def _baseline_suffix_counts(report, final_side, inner_steps):
    trace=[row for row in report["trace"] if row["level"]==final_side]
    failures=[row for row in report["failures"] if row["level"]==final_side]
    bad_gradients=sum(row["reason"]=="missing/nonfinite gradient" for row in failures)
    trial_failures=sum(row["reason"]!="missing/nonfinite gradient" for row in failures)
    return dict(trial_objective_evaluations=len(trace)+trial_failures,
        successful_gradient_steps=sum(row["step"]<inner_steps for row in trace)-bad_gradients,
        backward_attempts=sum(row["step"]<inner_steps for row in trace),
        failed_trials=len(failures),failures=failures,
        scope="trial counts; per-stage anchor/acceptance/full-objective checks and final export evaluation are additional")


def _stop_status(reason):
    if reason.startswith(("nonfinite","tangent_")) or reason in ("rounded_geometry_rejected","no_finite_descent"):
        return "solver_failure_valid_best_retained"
    if reason in ("minimum_step","objective_backtrack_limit"):
        return "early_numerical_stop_valid_best_retained"
    if reason in ("gradient_budget","gradient_tolerance"):
        return "budget_or_tolerance_reached"
    return "other_stop_see_reason"


def _load_reused_prefix(path, grid_side):
    """Load the comparison's sibling artifacts, never its terminal fiber map."""
    previous=json.loads(path.read_text(encoding="utf-8"))
    def artifact(key):
        recorded=Path(previous[key])
        sibling=path.parent/recorded.name
        return sibling if sibling.exists() else recorded
    baseline_path=artifact("baseline_output")
    prefix_path=artifact("prefix_output")
    baseline=json.loads(baseline_path.with_suffix(".json").read_text(encoding="utf-8"))
    configuration=baseline["configuration"]
    if configuration["grid_side"]!=grid_side:
        raise ValueError("reused prefix requires the same final control size")
    if configuration!=previous["actual_configuration"]:
        raise ValueError("saved baseline and comparison configurations differ")
    capture=PrefixCapture(configuration["levels"][-2],grid_side)
    with np.load(prefix_path,allow_pickle=False) as saved:
        payload={key:np.array(saved[key],copy=True) for key in saved.files}
    expected=(1,grid_side,grid_side,2)
    precision={"float32":np.float32,"float64":np.float64}[configuration["precision"]]
    for key in ("vertices","best_prefix_vertices","boundary_reference"):
        if payload[key].shape!=expected or payload[key].dtype!=precision or not np.isfinite(payload[key]).all():
            raise ValueError(f"invalid saved prefix {key}")
    if (int(payload["prefix_control_side"])!=capture.prefix_side or
            str(payload["interpolation"])!=configuration["interpolation"]):
        raise ValueError("saved prefix control/interpolation mismatch")
    stride=(grid_side-1)//(capture.prefix_side-1)
    if not np.array_equal(payload["coarse_vertices"],payload["vertices"][:,::stride,::stride]):
        raise ValueError("saved prefix old nodes differ from coarse vertices")
    capture.prefix=torch.from_numpy(payload["vertices"])
    capture.best_prefix=torch.from_numpy(payload["best_prefix_vertices"])
    capture.best_prefix_objective=float(payload["best_prefix_objective"])
    best_value=float(baseline["initial"]["total"])
    for index,stage in enumerate(baseline["stages"]):
        if stage["control_side"]<=capture.prefix_side and stage["accepted_full_total"]<best_value:
            best_value=stage["accepted_full_total"]
            capture.best_prefix_stage=index
    if not np.isfinite(capture.best_prefix_objective) or capture.best_prefix_objective!=best_value:
        raise ValueError("saved best-prefix objective disagrees with baseline prefix selection")
    prefix_stages=[row for row in baseline["stages"] if row["control_side"]==capture.prefix_side]
    if not prefix_stages or tuple(prefix_stages[-1]["direction"])!=(0.,1.):
        raise ValueError("baseline lacks incoming final-stage y prefix")
    capture.prefix_elapsed=previous["baseline_prefix_elapsed_seconds"]
    capture.last_elapsed=capture.prefix_elapsed+previous["baseline_suffix_callback_elapsed_seconds"]
    capture.callback_seconds=previous["baseline_callback_seconds"]
    capture.prefix_callback_seconds=previous["baseline_prefix_callback_seconds"]
    return baseline,baseline_path,prefix_path,capture,payload,previous


def run(args, *, production=True, solver=None):
    """Run a fresh baseline or reuse its saved prefix, then only the suffix."""
    if args.output.suffix!=".json":raise ValueError("comparison output must be a .json path")
    stem=args.output.with_suffix("")
    baseline_path=stem.with_name(stem.name+"_baseline.npz")
    prefix_path=stem.with_name(stem.name+"_prefix.npz")
    direct_path=stem.with_name(stem.name+"_fiber.npz")
    paths=[args.output,baseline_path,baseline_path.with_suffix(".json"),prefix_path,
        direct_path,direct_path.with_suffix(".json")]
    for path in paths:
        if path.exists():raise FileExistsError(path)
    reuse=getattr(args,"reuse_prefix_report",None)
    if reuse:
        if getattr(args,"report",None) is not None:raise ValueError("choose report or reuse-prefix-report")
        load_tick=time.perf_counter()
        baseline,baseline_path,reused_prefix_path,capture,payload,previous=_load_reused_prefix(reuse,args.grid_side)
        reused_prefix_load_seconds=time.perf_counter()-load_tick
        source=baseline
    else:
        source=json.loads(args.report.read_text(encoding="utf-8"))
    config=prepare_configuration(source,baseline_path,args.grid_side,production=production)
    if reuse:
        if isinstance(config.threads,bool) or not isinstance(config.threads,int) or config.threads<=0:
            raise ValueError("replay threads must be a positive integer")
        torch.set_num_threads(config.threads)
    if solver is None:
        from qcopt.neural_bijection.dense.coordinated_fiber_lbfgs import solve_coordinated_fiber_lbfgs
        solver=solve_coordinated_fiber_lbfgs
    baseline_call_seconds=0.
    if not reuse:
        capture=PrefixCapture(config.levels[-2],config.grid_side)
        baseline_tick=time.perf_counter()
        baseline=optimize(config,accepted_stage_callback=capture)
        baseline_call_seconds=time.perf_counter()-baseline_tick
    if capture.prefix is None or capture.prefix_elapsed is None or capture.last_elapsed is None:
        raise RuntimeError("baseline did not supply its final penultimate-control y prefix")
    if not baseline["saved_binary_certificate"]["valid"]:
        raise RuntimeError("baseline output failed saved certificate")

    device=torch.device(config.device)
    dtype={"float32":torch.float32,"float64":torch.float64}[config.precision]
    synchronize=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    prepare_tick=time.perf_counter()
    evidence,matrix,offset,metadata=_make_evidence(config,device,dtype)
    reference=identity_vertices(config.grid_side,device=device).to(dtype)
    if reuse:
        for key,actual in (("post_affine_matrix",matrix),("post_affine_offset",offset),
                ("boundary_reference",reference.cpu().numpy())):
            if not np.array_equal(payload[key],actual):raise ValueError(f"saved prefix {key} differs from current evidence")
    reference_corners=q1_corner_determinants(reference.double())
    current,coarse,transition=reconstruct_transition(capture.prefix,config.levels[-2],
        device=device,dtype=dtype,diagonal=config.interpolation[-2:])
    require_nested_fine_margin(current,reference_corners,config.minimum_jacobian,"shared suffix prefix")
    if not validate_q1_map(current,reference)["valid"]:raise RuntimeError("invalid shared prefix")
    initial_objective=float(baseline["initial"]["total"])
    if capture.best_prefix_objective<initial_objective:
        best=capture.best_prefix.to(device=device,dtype=dtype).clone()
        best_objective=capture.best_prefix_objective
        selected=dict(kind="prefix",stage=capture.best_prefix_stage)
    else:
        best=reference.clone()
        best_objective=initial_objective
        selected=dict(kind="initial",stage=None)
    setup_evaluations=0
    reuse_metadata={}
    if reuse:
        # The archive already stores the initial-vs-prefix winner. Re-evaluate
        # both it and the initial map on the freshly reconstructed complete E.
        best=capture.best_prefix.to(device=device,dtype=dtype).clone()
        if not validate_q1_map(best,reference)["valid"]:raise RuntimeError("invalid saved best prefix")
        require_nested_fine_margin(best,reference_corners,config.minimum_jacobian,"saved best prefix")
        with torch.no_grad():
            best_objective=float(evidence(best)[0])
            refreshed_initial=float(evidence(reference)[0])
        if not np.isfinite([best_objective,refreshed_initial]).all():
            raise RuntimeError("nonfinite reused-prefix setup objective")
        setup_evaluations=2
        selected=dict(kind="prefix" if capture.best_prefix_stage is not None else "initial",
            stage=capture.best_prefix_stage)
        reuse_metadata=dict(reused_prefix_report=str(reuse),reused_prefix_source=str(reused_prefix_path),
            reused_prefix_load_seconds=reused_prefix_load_seconds,
            stored_best_prefix_objective=capture.best_prefix_objective,
            reevaluated_best_prefix_objective=best_objective,reevaluated_initial_objective=refreshed_initial,
            best_prefix_objective_difference=best_objective-capture.best_prefix_objective)
        if refreshed_initial<best_objective:
            best=reference.clone();best_objective=refreshed_initial
            selected=dict(kind="initial",stage=None)
    prefix_certificate=_save_map(prefix_path,current,reference,matrix,offset,config.interpolation,
        coarse_vertices=coarse.cpu().numpy(),best_prefix_vertices=best.cpu().numpy(),
        best_prefix_objective=np.asarray(best_objective),
        prefix_control_side=np.asarray(config.levels[-2]))
    synchronize()
    preparation_seconds=time.perf_counter()-prepare_tick
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    tick=time.perf_counter()
    records=[]
    for direction in ((1.,0.),(0.,1.)):
        stage_tick=time.perf_counter()
        rescue_options={"tangent_rescue":True} if getattr(args,"tangent_rescue",False) else {}
        result=solver(current,evidence,initial_displacement_rms=_physical_initial_rms(config),
            reference=reference,direction=direction,minimum_jacobian=config.minimum_jacobian,
            theta=.95,fraction_to_boundary=.99,max_gradient_evaluations=30,
            history_size=5,max_backtracks=6,**rescue_options)
        synchronize()
        stage_seconds=time.perf_counter()-stage_tick
        current=result.best_vertices.detach()
        if not validate_q1_map(current,reference)["valid"]:
            raise RuntimeError("fiber solver returned an invalid best map")
        require_nested_fine_margin(current,reference_corners,config.minimum_jacobian,"fiber accepted best")
        # Re-evaluate the identical complete objective for output selection.
        with torch.no_grad():value=float(evidence(current)[0])
        if not np.isfinite(value):raise RuntimeError("nonfinite fiber best-map objective")
        if value<best_objective:
            best=current.clone();best_objective=value
            selected=dict(kind="fiber",stage=len(records))
        records.append(dict(direction=direction,seconds=stage_seconds,
            initial_objective=result.initial_objective,final_objective=result.final_objective,
            best_objective=result.best_objective,selection_objective=value,
            counts=result.counts,stop_reason=result.stop_reason,calibration=result.calibration,
            solver_status=_stop_status(result.stop_reason),trace=result.trace))
        if _stop_status(result.stop_reason)=="solver_failure_valid_best_retained" or result.counts.get("geometry_rejections",0):break
    synchronize()
    suffix_seconds=time.perf_counter()-tick
    peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
    with torch.no_grad():final,parts=evidence(best)
    certificate=_save_map(direct_path,best,reference,matrix,offset,config.interpolation,
        terminal_vertices=current.cpu().numpy())
    pair_status=("solver_failure_valid_best_retained" if any(
        row["solver_status"]=="solver_failure_valid_best_retained" for row in records) else
        "early_numerical_stop_valid_best_retained" if any(
            row["solver_status"]=="early_numerical_stop_valid_best_retained" for row in records) else
        "budget_or_tolerance_reached" if len(records)==2 and all(
            row["solver_status"]=="budget_or_tolerance_reached" for row in records) else "other_stop_see_reasons")
    evaluation_counts=dict(solver_objective_evaluations=sum(row["counts"]["objective_evaluations"] for row in records),
        stage_selection_objective_evaluations=len(records),final_report_objective_evaluations=1)
    evaluation_counts["complete_objective_evaluations"]=sum(evaluation_counts.values())
    direct=dict(configuration=_plain_configuration(config),
        optimizer="physical scalar-fiber L-BFGS final x/y suffix; no solver differentiation",
        selected=selected,final=dict(total=float(final),**{k:float(v) for k,v in parts.items()}),
        source_prefix=str(prefix_path),**transition,
        prefix_preparation_seconds=preparation_seconds,optimize_seconds=suffix_seconds,
        peak_allocated_bytes=peak,stages=records,saved_binary_certificate=certificate,
        stop_reasons=[row["stop_reason"] for row in records],
        solver_status=pair_status,objective_evaluation_counts=evaluation_counts,
        initial_displacement_rms=_physical_initial_rms(config),
        tangent_rescue=getattr(args,"tangent_rescue",False),
        setup_complete_objective_evaluations=setup_evaluations,**reuse_metadata,
        budget="each coordinate: at most30 gradients including initial; at most6 halvings plus first trial per search; counts are actual",
        selection="minimum complete full-resolution objective over initial, accepted prefix and accepted fiber stages",
        landmarks_used=False,**metadata)
    direct["configuration"]["output"]=str(direct_path)
    direct_path.with_suffix(".json").write_text(json.dumps(direct,indent=2)+"\n",encoding="utf-8")
    actual=_plain_configuration(config)
    changes={key:dict(source=source["configuration"].get(key),actual=value)
        for key,value in actual.items() if source["configuration"].get(key)!=value}
    comparison=dict(source_report=str(baseline_path.with_suffix(".json") if reuse else args.report),source_configuration=source["configuration"],
        actual_configuration=actual,configuration_changes=changes,
        baseline_output=str(baseline_path),fiber_output=str(direct_path),prefix_output=str(prefix_path),
        prefix_control_side=config.levels[-2],final_control_side=config.grid_side,
        **transition,prefix_certificate=prefix_certificate,
        baseline_final=baseline["final"],fiber_final=direct["final"],
        baseline_suffix_counts=_baseline_suffix_counts(baseline,config.grid_side,config.inner_steps),
        fiber_suffix_counts={key:sum(row["counts"].get(key,0) for row in records)
            for key in sorted({key for row in records for key in row["counts"]})},
        fiber_solver_status=pair_status,fiber_stop_reasons=direct["stop_reasons"],
        fiber_objective_evaluation_counts=evaluation_counts,
        baseline_prefix_elapsed_seconds=capture.prefix_elapsed,
        baseline_suffix_callback_elapsed_seconds=capture.last_elapsed-capture.prefix_elapsed,
        baseline_complete_call_seconds=baseline_call_seconds,
        baseline_executed_this_run=not bool(reuse),
        historical_baseline_complete_call_seconds=((previous.get("historical_baseline_complete_call_seconds") or
            previous["baseline_complete_call_seconds"]) if reuse else None),
        baseline_counts_scope="historical baseline run" if reuse else "baseline executed this run",
        setup_complete_objective_evaluations=setup_evaluations,
        tangent_rescue=getattr(args,"tangent_rescue",False),**reuse_metadata,
        baseline_callback_seconds=capture.callback_seconds,
        baseline_prefix_callback_seconds=capture.prefix_callback_seconds,
        fiber_preparation_seconds=preparation_seconds,fiber_suffix_seconds=suffix_seconds,
        baseline_peak_allocated_bytes=baseline["peak_allocated_bytes"],fiber_peak_allocated_bytes=peak,
        baseline_certificate=baseline["saved_binary_certificate"],fiber_certificate=certificate,
        scope="diagnostic optimizer/parameterization comparison, not a neural decoder or generalization test; prefix shared only within this grid",
        timing_scope=("baseline prefix/suffix/callback times and peak memory are historical, baseline_complete_call_seconds is zero (no rerun); " if reuse else "")+"baseline suffix is last callback timestamp minus prefix timestamp, includes intervening callback/cloning overhead and final-grid refinement; direct suffix includes solver, actual-map checks and stage selection; fresh evidence/prefix export setup and final export/certificate are separately excluded; peaks reset after feature setup, baseline peak spans all controls",
        gradient_budget="baseline30 Adam gradients plus final trial per coordinate; fiber30 includes initial gradient and forward-only Armijo trials; no claim of equal work",
        landmarks_used=False)
    args.output.write_text(json.dumps(comparison,indent=2)+"\n",encoding="utf-8")
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report",type=Path)
    source.add_argument("--reuse-prefix-report",type=Path,
        help="Existing comparison JSON; load its sibling prefix and baseline artifacts without rerunning baseline")
    parser.add_argument("--tangent-rescue",action="store_true",help="Enable bounded core-solver tangent rescue")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--grid-side",type=int,choices=(257,1025),required=True)
    result=run(parser.parse_args())
    print(json.dumps(result))


if __name__=="__main__":main()
