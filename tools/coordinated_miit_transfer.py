"""Three metadata-selected MIIT prostate DEVELOPMENT pairs, not blind testing.

All inputs are images. No annotation argument, coordinate reader or scorer is
imported. Previously used serial sections belong to ONE physical sample.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools.digital_birl_pair_canvas import make_canvas
from tools.digital_superglue_direct_affine import extract as extract_affine
from tools.coordinated_image_matches import extract as extract_matches
from tools.coordinated_lung_all20 import (
    METHODS, make_configuration as original_configuration, _plain,
    _check_input, _validate_matches, _check_safe_export, _native_run,
)
from tools.coordinated_f2_floor_compare import _actual_ratio
from tools.coordinated_real_case import optimize


PAIRS=((2,3),(7,8),(10,11))  # MOVING section, FIXED section; metadata only.
SCALE_ASSUMPTION="unknown physical spacing; shared native-pixel development layout, not inferred equal physical scale"


def pairs(args):
    return [dict(name=f"miit_{moving}_to_{fixed}",moving_section=moving,fixed_section=fixed,
        source_moving=args.source_data/str(moving)/"images/image.tif",
        source_fixed=args.source_data/str(fixed)/"images/image.tif",
        moving=args.output/f"miit_{moving}_to_{fixed}_moving512.png",
        fixed=args.output/f"miit_{moving}_to_{fixed}_fixed512.png",
        affine=args.output/f"miit_{moving}_to_{fixed}_affine.npz",
        layout=args.output/f"miit_{moving}_to_{fixed}_layout.json") for moving,fixed in PAIRS]


def make_configuration(pair,method,args,*,production=True):
    config=original_configuration(pair,method,args,production=production)
    config.joint_prior_backend="eager"
    config.match_p1_sampling="existing"  # Frozen correspondence DATA, not accelerated point dispatch.
    config.image_weight=1.
    config.capture_prefix="none"
    config.f2_floor_safety_fraction=.95 if method=="f2" else 1.
    return config


def _finite_json(value):
    """Ordinary failed-report retention; nonfinite diagnostics cannot be success."""
    if isinstance(value,dict):
        cleaned={};bad=False
        for key,item in value.items():
            cleaned[key],invalid=_finite_json(item);bad|=invalid
        return cleaned,bad
    if isinstance(value,(list,tuple)):
        items=[_finite_json(item) for item in value]
        return [item[0] for item in items],any(item[1] for item in items)
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):return None,True
    return value,False


def run(args,*,production=True,canvas_maker=None,affine_extractor=None,
        match_extractor=None,optimizer=None,dhr_runner=None):
    initializer=getattr(args,"initializer","direct")
    if initializer not in ("direct","quarter_turns"):raise ValueError("initializer must be direct or quarter_turns")
    if args.device not in ("cpu","cuda") or isinstance(args.threads,bool) or not isinstance(args.threads,int) or args.threads<1:
        raise ValueError("cpu/cuda device and positive integer threads required")
    args=argparse.Namespace(**vars(args))
    for key in ("source_data","output"):setattr(args,key,Path(getattr(args,key)).resolve())
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("new/empty MIIT development output directory required")
    canvas_maker=make_canvas if canvas_maker is None else canvas_maker
    if affine_extractor is None:
        if initializer=="quarter_turns":
            from tools.coordinated_rotation_initializer import extract as affine_extractor
        else:affine_extractor=extract_affine
    match_extractor=extract_matches if match_extractor is None else match_extractor
    optimizer=optimize if optimizer is None else optimizer
    dhr_runner=_native_run if dhr_runner is None else dhr_runner
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    side=512 if production else 16
    cohort=pairs(args);rows=[];started=time.perf_counter()
    for pair in cohort:
        row=dict(name=pair["name"],moving_section=pair["moving_section"],fixed_section=pair["fixed_section"],
            fixed=pair["fixed"].name,moving=pair["moving"].name,affine=pair["affine"].name,layout=pair["layout"].name,
            source_fixed=str(pair["source_fixed"]),source_moving=str(pair["source_moving"]),
            map_direction="fixed_canvas_to_moving_canvas",input_status="pending",status="pending",
            affine_estimation=dict(status="pending",report=pair["name"]+"_affine.json"),
            raw_matches=dict(status="pending",path=pair["name"]+"_raw_matches.json"),methods={})
        for method in METHODS:
            if method=="dhr":
                folder=pair["name"]+"_dhr";final=folder+"/common_affine_dhr/Results_Final/"
                row["methods"][method]=dict(status="pending",output_directory=folder,report=folder+"/runtime.json",
                    configuration=folder+"/config.json",field=final+"displacement_field.mha",
                    postprocessing_params=final+"postprocessing_params.json",geometry_certified=False)
            else:
                config=make_configuration(pair,method,args,production=production)
                row["methods"][method]=dict(status="pending",output=config.output.name,
                    report=config.output.with_suffix(".json").name,configuration=_plain(vars(config)))
        rows.append(row)
    report=dict(protocol="frozen three MIIT metadata-selected moving-to-fixed DEVELOPMENT pairs",cohort_size=3,
        prediction_complete=False,annotations_read=False,rows=rows,source_data=str(args.source_data),
        started_utc=datetime.now(timezone.utc).isoformat(),
        cohort_scope="ONE previously used prostate sample; additional-organ/specimen development transfer, not blind independent validation",
        scale_assumption=SCALE_ASSUMPTION,initializer=initializer,
        initializer_scope="positive image-only SuperGlue similarity, identical for analytic/F2/native DHR; quarter_turns is a disclosed amended initializer protocol",
        correspondence_scope="frozen image-only raw correspondence DATA; existing point evaluator, frozen raster P1 queries",
        timing_scope="serial complete calls include setup/loading/checks; affine matcher model is rebuilt once per pair; no hidden warmup or equal-runtime claim",
        memory_scope="safe-method existing executable CUDA allocated peaks; native wrapper resets before call and synchronizes/readbacks after call. Allocated peaks include any live baseline allocation, not whole-process resident/RSS or other-process GPU memory")
    def persist():
        report["elapsed_seconds"]=time.perf_counter()-started
        (args.output/"predictions.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()  # ALL three pending directions before image or model calls.
    for pair,row in zip(cohort,rows,strict=True):
        tick=time.perf_counter()
        try:
            layout=canvas_maker(pair["source_moving"],pair["source_fixed"],pair["moving"],pair["fixed"],pair["layout"],side=side)
            layout["scale_assumption"]=SCALE_ASSUMPTION
            for role in ("moving","fixed"):
                layout[role]["canvas_mpp"]=None;layout[role]["original_mpp_xy"]=None
            pair["layout"].write_text(json.dumps(layout,indent=2)+"\n",encoding="utf-8")
            affine=affine_extractor([(pair["name"],pair["fixed"],pair["moving"],pair["affine"])],
                args.output/row["affine_estimation"]["report"],device_name=args.device,match_threshold=.3)
            if (affine.get("landmarks_initial_DHR_affine_full_DHR_field_loaded") is not False
                    or len(affine.get("rows",[]))!=1 or affine["rows"][0].get("name")!=pair["name"]):
                raise ValueError("single image-only affine report with correct direction required")
            source=affine["rows"][0]
            matrix,offset=_check_input(pair,source,side)
            row["affine_estimation"].update(status="ok",matrix=matrix.tolist(),offset=offset.tolist())
            row["affine_estimation"].update({key:source[key] for key in
                ("selected_quarter_turns","matcher_calls","model_build_and_load_seconds") if key in source})
            row["input_status"]="ok"
        except Exception as error:
            row.update(input_status="failed",input_error=f"{type(error).__name__}: {error}")
            row["affine_estimation"]["status"]="failed"
        row["input_preparation_seconds"]=time.perf_counter()-tick
        if row["input_status"]=="ok":
            tick=time.perf_counter()
            try:
                match_args=argparse.Namespace(fixed=pair["fixed"],moving=pair["moving"],affine=pair["affine"],
                    output=args.output/row["raw_matches"]["path"],image_side=side,device=args.device)
                raw=match_extractor(match_args)
                metadata=_validate_matches(match_args.output,pair,matrix,offset,side)
                row["raw_matches"].update(status="ok",validation=metadata,
                    **{key:raw.get(key) for key in ("raw_matches","setup_seconds","inference_seconds","peak_allocated_bytes")})
                clean,bad=_finite_json(row["raw_matches"]);row["raw_matches"]=clean
                if bad:raise ValueError("nonfinite raw matcher diagnostics")
            except Exception as error:
                row["raw_matches"].update(status="failed",error=f"{type(error).__name__}: {error}")
            row["raw_matches"]["complete_call_seconds"]=time.perf_counter()-tick
        else:row["raw_matches"].update(status="skipped",reason="invalid input/initializer")
        persist()
        for method in METHODS:
            record=row["methods"][method]
            if row["input_status"]!="ok" or (method!="dhr" and row["raw_matches"]["status"]!="ok"):
                record.update(status="skipped",reason="invalid input/initializer or raw correspondence failure");persist();continue
            tick=time.perf_counter()
            try:
                if method!="dhr":
                    config=make_configuration(pair,method,args,production=production)
                    result=optimizer(config)
                    clean,bad=_finite_json({key:result.get(key) for key in ("initial","final","gradient_steps","failed_trials",
                        "evaluations","objective_evaluations","saved_binary_certificate","peak_allocated_bytes","optimize_seconds",
                        "loading_seconds","feature_seconds","serialization_seconds","certification_seconds","end_to_end_seconds")})
                    record.update(clean)
                    if bad:raise ValueError("nonfinite optimizer diagnostics")
                    _check_safe_export(config.output,result,matrix,offset,config.grid_side)
                    ratio=_actual_ratio(config.output)
                    record.update(actual_minimum_corner_ratio=ratio,actual_strict_floor_valid=ratio>config.minimum_jacobian)
                    if not record["actual_strict_floor_valid"]:raise ValueError("strict actual eta floor required")
                    if method=="f2" and result.get("configuration",{}).get("f2_floor_safety_fraction")!=.95:
                        raise ValueError("explicit F2 floor reserve .95 must be recorded")
                    expected=config.cycles*len(config.levels)*config.inner_steps*(2 if method=="analytic" else 1)
                    record.update(status="ok",expected_gradient_steps=expected,
                        budget_complete=result["gradient_steps"]==expected and result["failed_trials"]==0)
                else:
                    if args.device=="cuda":
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    dhr_runner(argparse.Namespace(fixed=pair["fixed"],moving=pair["moving"],affine=pair["affine"],
                        output=args.output/record["output_directory"],device=args.device,threads=args.threads,image_side=side))
                    if args.device=="cuda":torch.cuda.synchronize()
                    record["wrapper_observed_peak_allocated_bytes"]=(torch.cuda.max_memory_allocated()
                        if args.device=="cuda" else None)
                    native=json.loads((args.output/record["report"]).read_text(encoding="utf-8"))
                    for key in ("field","configuration","postprocessing_params"):
                        if not (args.output/record[key]).is_file():raise FileNotFoundError(record[key])
                    if (not np.array_equal(np.asarray(native["post_affine_matrix"],dtype=np.float32),matrix)
                            or not np.array_equal(np.asarray(native["post_affine_offset"],dtype=np.float32),offset)):
                        raise ValueError("native DHR common affine mismatch")
                    clean,bad=_finite_json({key:native.get(key) for key in ("runtime_seconds","nonrigid_seconds","preprocessing_seconds")})
                    record.update(clean)
                    if bad:raise ValueError("nonfinite native diagnostics")
                    record["status"]="ok"
            except Exception as error:record.update(status="failed",error=f"{type(error).__name__}: {error}")
            record["complete_call_seconds"]=time.perf_counter()-tick;persist()
        row["status"]="ok" if all(item["status"]=="ok" for item in row["methods"].values()) else "partial_failure"
        persist()
    report.update(prediction_complete=True,method_success_counts={method:sum(row["methods"][method]["status"]=="ok" for row in rows) for method in METHODS},
        safe_method_incomplete_budget_counts={method:sum(row["methods"][method].get("budget_complete") is False for row in rows) for method in ("analytic","f2")})
    persist();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--initializer",choices=("direct","quarter_turns"),default="direct")
    print(json.dumps(run(parser.parse_args())["method_success_counts"]))


if __name__=="__main__":main()
