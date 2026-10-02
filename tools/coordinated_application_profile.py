"""Profile unchanged HE->CC10 analytic application, bracketed by normal runs.

Uses the existing all20 raw-match JSON/affine/canvases. No matcher, labels,
optimizer replacement, graph compilation, or production-source edit occurs.
CPU phase ranges are nested; they are NOT an additive CUDA time partition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools import coordinated_real_case as application
from tools.coordinated_lung_all20 import make_configuration


@contextmanager
def instrumentation():
    """Reversible process-local wrappers; each calls its original exactly once."""
    from qcopt.neural_bijection.dense import coordinated_arap,coordinated_stage_cache,coordinated_update
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
    from qcopt.neural_bijection.dense import digital_q1
    targets=[
        (application,"load_registration_evidence","application.image_load"),
        (application,"load_image_matches","application.frozen_match_load"),
        (application.Evidence,"__init__","application.evidence_setup"),
        (application.Evidence,"__call__","application.evidence"),
        (application,"self_similarity","application.feature_extraction"),
        (coordinated_stage_cache.FrozenAnchorCoordinatedUpdate,"__init__","application.decoder_setup"),
        (coordinated_stage_cache.FrozenAnchorCoordinatedUpdate,"__call__","application.decoder"),
        (FrozenP1Evaluator,"__init__","application.map_sampling_setup"),
        (FrozenP1Evaluator,"forward","application.map_sampling"),
        (torch.nn.functional,"grid_sample","application.raster_sampling"),
        (coordinated_arap,"p1_arap_energy","application.prior_arap"),
        (application,"corner_symmetric_dirichlet","application.prior_shape"),
        (ImageCorrespondences,"forward","application.prior_matches"),
        (application,"validate_q1_map","application.accepted_map_check"),
        (application,"certify_q1_binary_map","application.saved_binary_certificate"),
        (torch.autograd,"backward","application.backward"),
        (torch.optim.Adam,"step","application.adam_step"),
        (torch.optim.Adam,"zero_grad","application.adam_zero_grad"),
    ]
    targets.extend((owner,"q1_corner_determinants","application.corner_geometry")
        for owner in (application,coordinated_stage_cache,coordinated_update,digital_q1))
    originals=[]
    def decorate(original,label):
        @wraps(original)
        def observed(*args,**kwargs):
            with torch.profiler.record_function(label):
                return original(*args,**kwargs)
        return observed
    try:
        for owner,name,label in targets:
            original=getattr(owner,name)
            # Remember whether the attribute was inherited (e.g. Adam.zero_grad).
            own=name in vars(owner)
            originals.append((owner,name,original,own))
            setattr(owner,name,decorate(original,label))
        yield
    finally:
        for owner,name,original,own in reversed(originals):
            if own:setattr(owner,name,original)
            else:delattr(owner,name)


def _event_row(event):
    return dict(name=str(event.key),device=str(event.device_type).split(".")[-1],count=int(event.count),
        self_cpu_ms=float(event.self_cpu_time_total)/1000.,inclusive_cpu_ms=float(event.cpu_time_total)/1000.,
        self_cuda_ms=float(event.self_device_time_total)/1000.,inclusive_cuda_ms=float(event.device_time_total)/1000.)


def summarize_events(events,*,cuda_requested,top=25):
    """Use actual Kineto CUDA events for the total, not CPU-linked duplicates."""
    cpu=[];kernels=[];ranges=[];scalar_sync=[]
    cpu_total=0.
    for event in events:
        row=_event_row(event)
        cpu_total+=row["self_cpu_ms"]
        annotation=bool(getattr(event,"is_user_annotation",False))
        if row["device"]=="CPU":
            if row["name"].startswith("application."):
                ranges.append(row)
            elif not annotation:
                cpu.append(row)
                if any(token in row["name"] for token in ("_local_scalar_dense","aten::item",
                        "cudaDeviceSynchronize","cudaStreamSynchronize","cudaEventSynchronize")):
                    scalar_sync.append(row)
        elif row["device"]=="CUDA" and not annotation:
            kernels.append(row)
    return dict(self_cpu_total_ms=cpu_total,self_cpu_operator_total_ms=sum(row["self_cpu_ms"] for row in cpu),
        self_cuda_kernel_total_ms=sum(row["self_cuda_ms"] for row in kernels) if cuda_requested else None,
        cpu_operator_linked_self_cuda_total_ms=sum(row["self_cuda_ms"] for row in cpu) if cuda_requested else None,
        cuda_kernel_events_observed=sum(row["count"] for row in kernels),
        cuda_capture_status=("captured" if kernels else "requested_but_no_kernel_events") if cuda_requested else "not_requested",
        top_self_cpu_operators=sorted(cpu,key=lambda row:row["self_cpu_ms"],reverse=True)[:top],
        top_self_cuda_kernels=sorted(kernels,key=lambda row:row["self_cuda_ms"],reverse=True)[:top],
        top_cpu_operators_by_linked_self_cuda=sorted(cpu,key=lambda row:row["self_cuda_ms"],reverse=True)[:top] if cuda_requested else [],
        ranges=sorted(ranges,key=lambda row:row["inclusive_cpu_ms"],reverse=True),
        scalar_and_synchronization_events=sorted(scalar_sync,key=lambda row:row["self_cpu_ms"],reverse=True),
        cuda_total_scope="sum self-device durations of actual non-annotation CUDA events, including compute kernels, memory copies (memcpy), and memset, following Kineto device-event convention; existing kernel-named JSON keys cover all these device events, not compute-only time or FLOPs. CPU-linked device attribution is separately reported, NEVER added again; event duration sums need not equal GPU wall time with overlap",
        range_scope="nested inclusive CPU/linked-device ranges are non-additive; parent/child ranges overlap. Host range duration is not GPU phase duration. Operator self times and actual CUDA device events (compute, copies, memset) are reported separately; no exclusive GPU phase partition is inferred",
        cpu_total_scope="sum profiler event self CPU times including annotation self time; operator-only sum excludes annotations; multiple threads and instrumentation can differ from complete-call wall time")


def load_configuration(path,output,*,production=True):
    source=json.loads(path.read_text(encoding="utf-8"))
    if source.get("annotations_read") is not False or source.get("cohort_size")!=20:
        raise ValueError("original image-only all20 prediction report required")
    rows=[row for row in source["rows"] if row["name"]=="he_to_cc10"]
    if len(rows)!=1 or rows[0]["input_status"]!="ok" or rows[0]["raw_matches"]["status"]!="ok":
        raise ValueError("one valid frozen HE-to-CC10 input/raw-match record required")
    row=rows[0]
    recorded=row["methods"]["analytic"]["configuration"]
    pair={key:Path(row[key]) for key in ("fixed","moving","affine")}
    pair["name"]="he_to_cc10"
    settings=argparse.Namespace(output=output.parent,device=recorded["device"],threads=recorded["threads"])
    config=make_configuration(pair,"analytic",settings,production=production)
    config.output=output
    matches=Path(row["raw_matches"]["path"])
    config.matches=(path.parent/matches if not matches.is_absolute() else matches).resolve()
    for key,value in vars(config).items():
        if key=="output":continue
        old=recorded.get(key)
        same=Path(old).resolve()==value.resolve() if isinstance(value,Path) and isinstance(old,str) else old==value
        if not same:raise ValueError(f"recorded analytic recipe differs at {key}")
    raw=json.loads(config.matches.read_text(encoding="utf-8"))
    if (raw.get("global_geometric_ransac_used") is not False or
            raw.get("targets_manual_landmarks_or_dense_teacher_loaded") is not False):
        raise ValueError("original frozen raw-confidence match provenance required")
    return config


def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def _execute(config,kind):
    device=torch.device(config.device)
    _sync(device);tick=time.perf_counter()
    report=application.optimize(config)
    _sync(device);elapsed=time.perf_counter()-tick
    if not report["saved_binary_certificate"]["valid"]:
        raise RuntimeError("unchanged optimizer returned an invalid saved map")
    return dict(kind=kind,output=str(config.output),report=str(config.output.with_suffix(".json")),
        complete_call_seconds=elapsed,**{key:report[key] for key in ("initial","final","gradient_steps",
        "evaluations","objective_evaluations","failed_trials","optimize_seconds","loading_seconds","feature_seconds",
        "serialization_seconds","certification_seconds","saved_binary_certificate","peak_allocated_bytes")})


def _compare(left,right):
    with np.load(left["output"],allow_pickle=False) as a,np.load(right["output"],allow_pickle=False) as b:
        av,bv=a["vertices"],b["vertices"]
        delta=av.astype(np.float64)-bv.astype(np.float64)
        map_bits=np.array_equal(av,bv)
        affine_equal=all(np.array_equal(a[key],b[key]) for key in ("post_affine_matrix","post_affine_offset","boundary_reference","interpolation"))
    counter_keys=("gradient_steps","evaluations","objective_evaluations","failed_trials")
    return dict(map_bitwise_equal=map_bits,map_max_abs_delta=float(np.max(np.abs(delta))),
        map_rms_delta=float(np.sqrt(np.mean(delta*delta))),affine_boundary_interpolation_equal=affine_equal,
        initial_total_delta=right["initial"]["total"]-left["initial"]["total"],
        final_total_delta=right["final"]["total"]-left["final"]["total"],
        final_part_deltas={key:right["final"][key]-value for key,value in left["final"].items()},
        counters_equal=all(left[key]==right[key] for key in counter_keys),
        counter_deltas={key:right[key]-left[key] for key in counter_keys})


def run(args,*,production=True):
    if args.output.suffix!=".json":raise ValueError("compact profile output must be .json")
    stem=args.output.with_suffix("")
    kinds=("unprofiled_before","profiled","unprofiled_after")
    paths={kind:stem.with_name(stem.name+"_"+kind+".npz") for kind in kinds}
    trace=getattr(args,"trace",None)
    targets=[args.output,*paths.values(),*[path.with_suffix(".json") for path in paths.values()]]
    if trace is not None:targets.append(trace)
    if len({path.resolve() for path in targets})!=len(targets):raise ValueError("output paths must be distinct")
    for path in targets:
        if path.exists():raise FileExistsError(path)
    config=load_configuration(args.predictions,paths[kinds[0]],production=production)
    device=torch.device(config.device)
    cuda=device.type=="cuda"
    if cuda and not torch.cuda.is_available():raise RuntimeError("source recipe requires CUDA, unavailable on this host")
    activities=[torch.profiler.ProfilerActivity.CPU]
    if cuda:activities.append(torch.profiler.ProfilerActivity.CUDA)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status="running",source_predictions=str(args.predictions),source_matches=str(config.matches),
        source_fixed=str(config.fixed),source_moving=str(config.moving),source_affine=str(config.affine),
        matcher_executed=False,annotations_read=False,activities=["CPU","CUDA"] if cuda else ["CPU"],runs=[],
        torch_version=torch.__version__,cuda_device_name=torch.cuda.get_device_name(device) if cuda else None,
        configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(config).items()},
        interpretation="unchanged HE-to-CC10 analytic recipe; brackets are repeated instance runs, not held-out accuracy tests. CUDA grid-sample backward can be nondeterministic; report actual differences and unprofiled repeat variation rather than assuming bitwise equality",
        timing_scope="matcher excluded: same already-frozen raw JSON every run. Complete-call timing synchronizes before/after existing optimize, includes input/features/optimizer/export/certificate; profiler session additionally includes profiler start/finalization and wrapper setup. Before can contain cold first optimizer setup; after is warmer. No cold/warm speedup claim",
        memory_scope="existing optimizer CUDA allocated peak only, reset by optimizer after features; profiler CPU event-buffer memory and process RSS are not measured. Shapes, stacks and profiler allocation recording are disabled to bound profiling overhead")
    def persist():args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    persist()
    try:
        before=_execute(config,kinds[0]);report["runs"].append(before);persist()
    except Exception as error:
        report.update(status="unprofiled_before_failed",error=f"{type(error).__name__}: {error}")
        persist();raise
    config.output=paths["profiled"]
    session_tick=time.perf_counter()
    try:
        with instrumentation():
            with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as profiler:
                with torch.profiler.record_function("application.complete"):
                    profiled=_execute(config,"profiled")
        report["profiler_session_seconds"]=time.perf_counter()-session_tick
        report["runs"].append(profiled)
        report["profile"]=summarize_events(profiler.key_averages(),cuda_requested=cuda)
        if trace is not None:
            trace.parent.mkdir(parents=True,exist_ok=True)
            profiler.export_chrome_trace(str(trace));report["trace"]=str(trace)
        del profiler
    except Exception as error:
        report["profile_error"]=f"{type(error).__name__}: {error}"
        report["profiler_session_seconds"]=time.perf_counter()-session_tick
    persist()
    config.output=paths["unprofiled_after"]
    try:
        after=_execute(config,"unprofiled_after");report["runs"].append(after)
    except Exception as error:
        report.update(status="unprofiled_after_failed",error=f"{type(error).__name__}: {error}")
        persist();raise
    mean=(before["complete_call_seconds"]+after["complete_call_seconds"])/2
    report.update(unprofiled_bracket_mean_seconds=mean,
        unprofiled_after_over_before=after["complete_call_seconds"]/before["complete_call_seconds"],
        agreement=dict(before_vs_after=_compare(before,after)))
    if "profile_error" not in report:
        report["agreement"].update(before_vs_profile=_compare(before,profiled),after_vs_profile=_compare(after,profiled))
        report.update(profile_overhead_vs_bracket_mean=profiled["complete_call_seconds"]/mean,
            profile_overhead_vs_before=profiled["complete_call_seconds"]/before["complete_call_seconds"],
            profile_overhead_vs_after=profiled["complete_call_seconds"]/after["complete_call_seconds"],
            profiler_session_overhead_vs_bracket_mean=report["profiler_session_seconds"]/mean)
    report["status"]="complete" if "profile_error" not in report else "profile_failed_brackets_preserved"
    if report["status"]=="complete" and cuda and report["profile"]["cuda_capture_status"]!="captured":
        report["status"]="complete_but_cuda_kernel_trace_missing"
    persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True,help="Existing all20 predictions.json")
    parser.add_argument("--output",type=Path,required=True,help="New compact profile JSON")
    parser.add_argument("--trace",type=Path,help="Optional new Chrome trace JSON; can be large")
    report=run(parser.parse_args())
    print(json.dumps({key:report.get(key) for key in ("status","activities","profile_overhead_vs_bracket_mean","profiler_session_seconds","agreement")}))


if __name__=="__main__":main()
