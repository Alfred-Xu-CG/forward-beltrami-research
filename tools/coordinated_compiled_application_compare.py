"""Frozen HE->CC10 dispatch comparison, with observed cold cost.

Compare joint priors (eager/Inductor), or frozen points (existing/frozen with
Inductor priors in BOTH arms). Two cold runs precede three warm A/B/B/A groups.
Every call starts from identity
inside the unchanged optimizer and writes a fresh map/report. No labels, new
matches, warm-start map, graph profiler, or silent fallback are used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from tools import coordinated_application_profile as profile
from tools import coordinated_real_case as application


def schedule(comparison_kind="joint_priors"):
    if comparison_kind not in ("joint_priors","frozen_points"):
        raise ValueError("comparison_kind must be joint_priors or frozen_points")
    arms=("eager","inductor") if comparison_kind=="joint_priors" else ("existing","frozen")
    cold_names=("cold_eager","cold_compiled") if comparison_kind=="joint_priors" else ("cold_existing","cold_frozen")
    rows = [dict(name=name,backend=backend,phase="observed_cold",group=None,position=position)
            for position,(name,backend) in enumerate(zip(cold_names,arms))]
    for group in range(3):
        for position, backend in enumerate((arms[0],arms[1],arms[1],arms[0])):
            rows.append(dict(name=f"warm_g{group}_{position}_{backend}", backend=backend,
                             phase="warm_ABBA", group=group, position=position))
    for row in rows:
        row.update(comparison_kind=comparison_kind,
            joint_prior_backend=row["backend"] if comparison_kind=="joint_priors" else "inductor",
            match_p1_sampling="existing" if comparison_kind=="joint_priors" else row["backend"])
    return rows


@contextmanager
def observe_factory(records):
    """Observe factory construction only, never wrap the compiled callable."""
    from tools import coordinated_compiled_priors as priors
    original = priors.make_joint_priors
    def observed(compile_backend=None):
        started = time.perf_counter()
        row = dict(requested_backend=compile_backend)
        try:
            result = original(compile_backend)
            row["status"] = "returned"
            return result
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["seconds"] = time.perf_counter()-started
            records.append(row)
    priors.make_joint_priors = observed
    try:
        yield
    finally:
        priors.make_joint_priors = original


@contextmanager
def observe_point_preparation(records):
    """Count/setup-time only; never intercept mapped queries or point losses."""
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    original=ImageCorrespondences.prepare_fixed_p1_sampling
    def observed(self,rows,columns,diagonal="ac"):
        started=time.perf_counter()
        row=dict(rows=rows,columns=columns,diagonal=diagonal)
        try:
            result=original(self,rows,columns,diagonal)
            row.update(status="returned",metadata=result)
            return result
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["seconds"]=time.perf_counter()-started
            records.append(row)
    ImageCorrespondences.prepare_fixed_p1_sampling=observed
    try:
        yield
    finally:
        ImageCorrespondences.prepare_fixed_p1_sampling=original


def expected_budget(config):
    stages = config.cycles*len(config.levels)*2
    gradients = stages*config.inner_steps
    evaluations = gradients+stages
    return dict(gradient_steps=gradients, evaluations=evaluations,
                objective_evaluations=evaluations+2*stages+2, failed_trials=0)


def execute(config, plan):
    device = torch.device(config.device)
    row = dict(**plan, output=str(config.output), report=str(config.output.with_suffix(".json")),
               factory_calls=[], point_preparation_calls=[], status="running",
               requested_dispatch=dict(joint_prior_backend=config.joint_prior_backend,
                                       match_p1_sampling=config.match_p1_sampling))
    profile._sync(device)
    started = time.perf_counter()
    try:
        point_context=observe_point_preparation(row["point_preparation_calls"]) if plan["comparison_kind"]=="frozen_points" else nullcontext()
        with observe_factory(row["factory_calls"]), point_context:
            result = application.optimize(config)
        profile._sync(device)
        row["complete_call_seconds"] = time.perf_counter()-started
    except Exception as error:
        row.update(status="failed_no_fallback", error=f"{type(error).__name__}: {error}",
                   complete_call_seconds=time.perf_counter()-started)
        return row
    row["compiler_factory_seconds"] = sum(item["seconds"] for item in row["factory_calls"])
    for key in ("initial", "final", "gradient_steps", "evaluations", "objective_evaluations",
                "failed_trials", "failures", "optimize_seconds", "loading_seconds", "feature_seconds",
                "serialization_seconds", "certification_seconds", "end_to_end_seconds", "peak_allocated_bytes",
                "saved_binary_certificate", "joint_prior_backend", "match_p1_sampling",
                "image_match_evidence", "landmarks_used", "selected_stage"):
        row[key] = result.get(key)
    reasons = []
    for key, wanted in expected_budget(config).items():
        if row[key] != wanted:
            reasons.append(f"{key} expected {wanted}, got {row[key]}")
    if row["failures"] != []:
        reasons.append("nonempty or missing optimizer failures")
    if not isinstance(row["saved_binary_certificate"],dict) or row["saved_binary_certificate"].get("valid") is not True:
        reasons.append("missing or invalid saved binary certificate")
    if row["landmarks_used"] is not False:
        reasons.append("label-off evidence not confirmed")
    for key in ("joint_prior_backend","match_p1_sampling"):
        if row[key] != plan[key]:
            reasons.append(f"optimizer did not report requested {key}")
    wanted_factories = 1 if plan["joint_prior_backend"] == "inductor" else 0
    if len(row["factory_calls"]) != wanted_factories:
        reasons.append("unexpected number of joint-prior factories")
    if plan["comparison_kind"]=="frozen_points":
        wanted_preparations=1 if plan["match_p1_sampling"]=="frozen" else 0
        if len(row["point_preparation_calls"])!=wanted_preparations:
            reasons.append("unexpected number of frozen point preparations")
        metadata=(row["image_match_evidence"] or {}).get("fixed_p1_sampling")
        if wanted_preparations and (not isinstance(metadata,dict) or
                metadata.get("prepared") is not True or metadata.get("rows")!=config.grid_side or
                metadata.get("columns")!=config.grid_side or metadata.get("diagonal")!="ac"):
            reasons.append("missing or inconsistent frozen point sampling metadata")
    for key in ("initial","final"):
        if not isinstance(row[key],dict) or not row[key] or not all(
                isinstance(value,(float,int)) and math.isfinite(value) for value in row[key].values()):
            reasons.append(f"missing/nonfinite {key} objective parts")
    try:
        with np.load(config.output,allow_pickle=False) as saved:
            if saved["vertices"].shape != (1,config.grid_side,config.grid_side,2):
                reasons.append("saved map has wrong shape")
            if saved["vertices"].dtype != np.float64 or not np.isfinite(saved["vertices"]).all():
                reasons.append("saved map is not finite float64 geometry")
            if str(saved["interpolation"].item()) != "p1_ac":
                reasons.append("saved map interpolation is not p1_ac")
    except Exception as error:
        reasons.append(f"saved map read failed: {type(error).__name__}: {error}")
    row.update(status="complete" if not reasons else "invalid_or_incomplete_attempt", invalid_reasons=reasons)
    return row


def compare_runs(runs):
    comparisons = []
    for left, right in itertools.combinations(runs,2):
        comparison = profile._compare(left,right)
        comparison.update(left=left["name"],right=right["name"],
            same_backend=left["backend"]==right["backend"],
            both_warm=left["phase"]==right["phase"]=="warm_ABBA",
            initial_part_deltas={key:right["initial"][key]-value for key,value in left["initial"].items()})
        comparisons.append(comparison)
    return comparisons


def numerical_summary(comparisons):
    result = {}
    for name, same_backend in (("warm_same_backend_repeat_variation",True),
                               ("warm_cross_backend_differences",False)):
        rows = [row for row in comparisons if row["both_warm"] and row["same_backend"]==same_backend]
        result[name] = dict(pair_count=len(rows),
            max_map_abs_delta=max((row["map_max_abs_delta"] for row in rows),default=None),
            max_map_rms_delta=max((row["map_rms_delta"] for row in rows),default=None),
            max_abs_final_total_delta=max((abs(row["final_total_delta"]) for row in rows),default=None))
    return result


def warm_summary(runs):
    warm = [row for row in runs if row["phase"] == "warm_ABBA"]
    arms=[row["backend"] for row in runs if row["phase"]=="observed_cold"]
    baseline,candidate=arms
    medians = {backend: statistics.median(row["complete_call_seconds"] for row in warm if row["backend"]==backend)
               for backend in arms}
    groups = []
    for group in range(3):
        rows = [row for row in warm if row["group"]==group]
        means = {backend: statistics.mean(row["complete_call_seconds"] for row in rows if row["backend"]==backend)
                 for backend in arms}
        groups.append(dict(group=group,order=[row["backend"] for row in rows],
            complete_call_means=means,baseline_over_candidate=means[baseline]/means[candidate]))
        if baseline=="eager":
            groups[-1]["eager_over_compiled"]=groups[-1]["baseline_over_candidate"]
    cold = {row["backend"]:row["complete_call_seconds"] for row in runs if row["phase"]=="observed_cold"}
    result=dict(complete_call_medians=medians,baseline=baseline,candidate=candidate,
        baseline_over_candidate_median=medians[baseline]/medians[candidate],
        paired_ABBA_groups=groups,median_paired_ABBA_ratio=statistics.median(row["baseline_over_candidate"] for row in groups),
        cold_complete_call_seconds=cold,
        observed_cold_minus_warm_median_seconds={arm:cold[arm]-medians[arm] for arm in arms},
        cold_excess_scope="Observed cold-to-warm difference includes compiler/runtime/cache/optimizer effects; NOT pure compiler cost. Cold costs are excluded from warm medians but retained here and per run.")
    if baseline=="eager":
        result.update(eager_over_compiled_median=medians[baseline]/medians[candidate],
            cold_compiled_over_cold_eager=cold[candidate]/cold[baseline],
            observed_cold_compiled_minus_warm_compiled_median_seconds=cold[candidate]-medians[candidate])
    else:
        result.update(existing_over_frozen_median=medians[baseline]/medians[candidate],
            cold_frozen_over_cold_existing=cold[candidate]/cold[baseline],
            point_cold_scope="Both arms use Inductor priors. First existing cold run includes first prior compilation; second frozen cold run can reuse those caches. NOT a point-compiler comparison; inspect warm ABBA for point dispatch timing.")
    return result


def _json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {key:_json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(args, *, production=True, test_backend_label=None):
    if production and test_backend_label is not None:
        raise ValueError("test backend label is forbidden for production benchmark")
    if args.output.suffix != ".json":
        raise ValueError("comparison output must be a new .json file")
    comparison_kind=getattr(args,"comparison_kind","joint_priors")
    plans = schedule(comparison_kind)
    stem = args.output.with_suffix("")
    paths = {row["name"]:stem.with_name(stem.name+"_"+row["name"]+".npz") for row in plans}
    targets = [args.output,*paths.values(),*[path.with_suffix(".json") for path in paths.values()]]
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("output paths must be distinct")
    for path in targets:
        if path.exists():
            raise FileExistsError(path)
    config = profile.load_configuration(args.predictions,paths[plans[0]["name"]],production=production)
    device = torch.device(config.device)
    init_start = time.perf_counter()
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("source recipe requires CUDA, unavailable on this host")
        # Frozen application recipes use unindexed "cuda"; CUDA_VISIBLE_DEVICES
        # chooses the physical GPU, and its first visible device is index zero.
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.cuda.init()
    profile._sync(device)
    initialization_seconds = time.perf_counter()-init_start
    report = dict(status="running",comparison_kind=comparison_kind,
        dispatch_arms={plan["backend"]:dict(joint_prior_backend=plan["joint_prior_backend"],match_p1_sampling=plan["match_p1_sampling"]) for plan in plans},
        source_predictions=str(args.predictions),source_matches=str(config.matches),
        source_fixed=str(config.fixed),source_moving=str(config.moving),source_affine=str(config.affine),
        configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(config).items()},
        expected_budget=expected_budget(config),planned_runs=plans,runs=[],comparisons=[],
        annotations_read=False,matcher_executed=False,initialization="fresh identity inside optimize for EVERY attempt; no saved-map warm starts",
        torch_version=torch.__version__,device=str(device),
        cuda_device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        device_initialization_seconds_excluded=initialization_seconds,
        compiler_cache_environment={key:os.environ.get(key) for key in (
            "TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","TORCHINDUCTOR_FX_GRAPH_CACHE",
            "TORCHINDUCTOR_AUTOGRAD_CACHE","TORCH_COMPILE_DEBUG_DIR","CUDA_VISIBLE_DEVICES")},
        compiler_settings=dict(backend="inductor",fullgraph=True,dynamic=False,mode="default"),
        evidence_scope=test_backend_label or "actual application backend dispatch; numerical differences reported, not assumed bitwise identical",
        speed_evidence_eligible=False,
        numerical_equivalence_claimed=False,
        numerical_scope="No arbitrary GPU bitwise or tolerance gate. Actual map and objective differences must be assessed against same-backend repeat variation before interpreting a timing gain as equivalent optimization quality.",
        timing_scope="External synchronized complete-call timer starts BEFORE compiler factory and optimize; includes loading/features, initial requires_grad=False calls, trial requires_grad=True forward/backward, optimization, export and certification, plus minimal factory observer and point-preparation observer when selected. GPU context initialization, recipe validation, comparison and report persistence excluded. Each compiled-prior attempt creates a fresh factory; compiled code caches may be reused.",
        point_preparation_timing_scope="Per-call preparation seconds are host-call elapsed without an extra CUDA synchronization; setup work is included in synchronized complete-call timing. Observer counts preparation only, never query evaluation or energies.",
        cold_scope=("Observed cold eager then observed cold compiled BEFORE any compiled application warmup in this runner. " if comparison_kind=="joint_priors" else
                    "Both arms use Inductor joint priors. Observed-cold existing point sampling FIRST includes prior compilation; observed-cold frozen point sampling SECOND may reuse those caches. No hidden prior warmup. This is NOT a point-compiler comparison. ")+"Caches are neither cleared nor claimed pristine. Record explicit environment paths; null means framework default not resolved. No first-call-minus-baseline pure compilation claim.",
        order_scope="After two observed-cold calls, three warm A/B/B/A groups in declared dispatch-arm order; no additional untimed application warmups. Fresh map outputs for all 14 attempts.",
        interpretation="One fixed known-specimen image-only pair, not a held-out generalization or clinical-validity test. CUDA grid_sample backward may be nondeterministic; inspect all pairwise differences and same-backend repeat variation.",
        memory_scope="Optimizer-reported CUDA peak allocated bytes, reset inside optimize after features. Does not include all cold compiler/host memory.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def persist():
        args.output.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for plan in plans:
        attempt_config = copy.deepcopy(config)
        attempt_config.output = paths[plan["name"]]
        attempt_config.joint_prior_backend = plan["joint_prior_backend"]
        attempt_config.match_p1_sampling = plan["match_p1_sampling"]
        row = execute(attempt_config,plan)
        report["runs"].append(row)
        persist()
        if row["status"] != "complete":
            report.update(status="incomplete_comparison_no_speed_claim",stopped_at=plan["name"],
                          unattempted_runs=[item["name"] for item in plans[len(report["runs"]):]])
            try:
                report["comparisons"] = compare_runs([item for item in report["runs"] if item["status"]=="complete"])
                report["numerical_summary"] = numerical_summary(report["comparisons"])
            except Exception as error:
                report["comparison_error"] = f"{type(error).__name__}: {error}"
            persist()
            return report
    try:
        report["comparisons"] = compare_runs(report["runs"])
        report["numerical_summary"] = numerical_summary(report["comparisons"])
        if not all(row["affine_boundary_interpolation_equal"] and row["counters_equal"] for row in report["comparisons"]):
            report["status"] = "inconsistent_comparison_no_speed_claim"
        else:
            report.update(status="complete",warm_summary=warm_summary(report["runs"]),
                          speed_evidence_eligible=test_backend_label is None)
    except Exception as error:
        report.update(status="incomplete_comparison_no_speed_claim",comparison_error=f"{type(error).__name__}: {error}")
    persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True,help="original all20 predictions.json")
    parser.add_argument("--output",type=Path,required=True,help="new JSON stem; also creates 14 fresh NPZ/JSON pairs")
    parser.add_argument("--comparison-kind",choices=("joint_priors","frozen_points"),default="joint_priors",
                        help="frozen_points holds Inductor joint priors fixed in both arms")
    report = run(parser.parse_args())
    print(json.dumps(dict(status=report["status"],completed_attempts=len(report["runs"]),
                          warm_summary=report.get("warm_summary"))))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
