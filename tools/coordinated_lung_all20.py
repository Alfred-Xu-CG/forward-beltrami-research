"""Predict exactly20 ordered directions of ONE previously viewed lung specimen.

Images and saved image-only affines only. Extract existing raw-confidence
matches for the whole cohort first, then run the fixed analytic/F2/native-DHR
recipes. This program neither imports a scorer nor accepts annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from tools.coordinated_image_matches import extract
from tools.coordinated_real_case import load_image_matches,optimize


STAINS=("he","cc10","cd31","ki67","prospc")
METHODS=("analytic","f2","dhr")


def pairs(args):
    image=lambda stain:args.canvas/("cc10_fixed512.png" if stain=="he" else stain+"_moving512.png")
    return [dict(name=fixed+"_to_"+moving,fixed_stain=fixed,moving_stain=moving,
        fixed=image(fixed),moving=image(moving),affine=args.affines_from/(fixed+"_to_"+moving+"_affine.npz"))
        for fixed,moving in itertools.permutations(STAINS,2)]


def make_configuration(pair,method,args,*,production=True):
    if method not in ("analytic","f2"):raise ValueError("analytic or f2 configuration required")
    return argparse.Namespace(fixed=pair["fixed"],moving=pair["moving"],affine=pair["affine"],
        output=args.output/(pair["name"]+"_"+method+".npz"),method=method,
        loss="mind",mind_order="transport",grid_side=257 if production else 9,
        image_side=512 if production else 16,image_levels=[32,64,128,256,512] if production else [8,16],
        levels=[17,33,65,129,257] if production else [5,9],inner_steps=30 if production else 1,
        preprocessing="raw_inverted",continuation_scope="all_cycles",cycles=1 if method=="analytic" else 2,
        learning_rate=.004,lr_calibration="edge",patch_cells=8 if production else 2,f2_accepted_gain=1.,
        regional_cells=32,regional_min_level=3,strain_weight=3.,strain_model="p1_arap",oob_weight=1.,
        minimum_jacobian=.001,precision="float64",image_precision="float32",shape_weight=1e-4,
        interpolation="p1_ac",p1_sampling="frozen",geometry_backend="stage_cache" if method=="analytic" else "existing",
        coordinate_mode="alternating",joint_backend="cached_manual",proposal_filter_steps=0,
        proposal_filter_min_level=129,fine_patch_cells=0,fine_patch_backend="ordinary",
        control_hierarchy="fixed",nested_evaluation="full_fine",output_selection="best_full",
        matches=args.output/(pair["name"]+"_raw_matches.json"),match_weight=.1,match_robust_scale=8.,
        device=args.device,threads=args.threads)


def _plain(value):
    return {key:str(item) if isinstance(item,Path) else item for key,item in value.items()}


def _source_rows(args,expected):
    report=json.loads((args.affines_from/"affines.json").read_text(encoding="utf-8"))
    if report.get("landmarks_initial_DHR_affine_full_DHR_field_loaded") is not False:
        raise ValueError("affine source must explicitly exclude landmarks, initial DHR affine and dense fields")
    rows=report["rows"]
    if len(rows)!=20 or {row["name"] for row in rows}!={row["name"] for row in expected}:
        raise ValueError("affine source cohort must contain exactly all20 ordered directions")
    return {row["name"]:row for row in rows}


def _check_input(pair,source,side):
    if source["status"]!="ok":raise ValueError("source affine estimation was not successful")
    for key in ("fixed","moving"):
        with Image.open(pair[key]) as image:
            if image.size!=(side,side):raise ValueError(f"original square{side} canvas required")
        if key in source and Path(str(source[key]).replace("\\","/")).name!=pair[key].name:
            raise ValueError("source affine image basename differs from requested canvas")
    with np.load(pair["affine"],allow_pickle=False) as saved:
        matrix=np.array(saved["post_affine_matrix"],copy=True)
        offset=np.array(saved["post_affine_offset"],copy=True)
        if str(saved["affine_source"])!="direct_image_superglue":
            raise ValueError("positive image-only direct_image_superglue affine required")
    if (matrix.shape!=(2,2) or offset.shape!=(2,) or not np.isfinite(matrix).all() or
            not np.isfinite(offset).all() or np.linalg.det(matrix.astype(np.float64))<=0):
        raise ValueError("finite positive supplied image-only affine required")
    # Existing executables all interpret these archives as binary32 affines.
    if matrix.dtype!=np.float32 or offset.dtype!=np.float32:
        raise ValueError("existing common initializer must use original float32 affine storage")
    for key,value in (("matrix",matrix),("offset",offset)):
        if key in source and not np.array_equal(np.asarray(source[key],dtype=np.float32),value):
            raise ValueError("stored affine differs from original affine report")
    return matrix,offset


def _validate_matches(path,pair,matrix,offset,side):
    record=json.loads(path.read_text(encoding="utf-8"))
    if record.get("status")!="ok" or record.get("global_geometric_ransac_used") is not False:
        raise ValueError("successful raw-confidence matches without geometric RANSAC required")
    _,metadata=load_image_matches(path,matrix,offset,fixed_path=pair["fixed"],moving_path=pair["moving"],
        image_side=side,device="cpu",dtype=torch.float64,robust_scale=8.)
    return metadata


def _check_safe_export(path,report,matrix,offset,side):
    if report.get("landmarks_used") is not False or not report["saved_binary_certificate"]["valid"]:
        raise ValueError("image-only prediction with a valid saved binary certificate required")
    with np.load(path,allow_pickle=False) as saved:
        if str(saved["interpolation"])!="p1_ac" or saved["vertices"].shape!=(1,side,side,2):
            raise ValueError("actual declared P1ac output grid differs")
        if not np.array_equal(saved["post_affine_matrix"],matrix) or not np.array_equal(saved["post_affine_offset"],offset):
            raise ValueError("actual saved map did not retain common affine")


def _native_run(args):
    from tools.coordinated_dhr_common import run
    return run(args)


def run(args,*,production=True,match_extractor=None,optimizer=None,dhr_runner=None):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("new or empty all20 output directory required")
    if args.device not in ("cpu","cuda"):
        raise ValueError("device must be cpu or cuda; select GPU through CUDA_VISIBLE_DEVICES")
    if isinstance(args.threads,bool) or not isinstance(args.threads,int) or args.threads<=0:
        raise ValueError("threads must be a positive integer")
    args=argparse.Namespace(**vars(args))
    for key in ("canvas","affines_from","output"):setattr(args,key,getattr(args,key).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    match_extractor=extract if match_extractor is None else match_extractor
    optimizer=optimize if optimizer is None else optimizer
    dhr_runner=_native_run if dhr_runner is None else dhr_runner
    cohort=pairs(args)
    side=512 if production else 16
    started=time.perf_counter()
    rows=[]
    for pair in cohort:
        row=dict(**_plain(pair),input_status="pending",status="pending",
            raw_matches=dict(status="pending",path=pair["name"]+"_raw_matches.json"),methods={})
        for method in METHODS:
            if method=="dhr":
                output=args.output/(pair["name"]+"_dhr")
                final=output/"common_affine_dhr/Results_Final"
                relative=lambda path:path.relative_to(args.output).as_posix()
                row["methods"][method]=dict(status="pending",output_directory=relative(output),report=relative(output/"runtime.json"),
                    configuration=relative(output/"config.json"),field=relative(final/"displacement_field.mha"),
                    postprocessing_params=relative(final/"postprocessing_params.json"),geometry_certified=False)
            else:
                config=make_configuration(pair,method,args,production=production)
                row["methods"][method]=dict(status="pending",output=config.output.name,report=config.output.with_suffix(".json").name,
                    configuration=_plain(vars(config)))
        rows.append(row)
    report=dict(protocol="all20 ordered directions in ONE previously viewed lung-lesion3 specimen",
        cohort_size=20,stains=list(STAINS),prediction_complete=False,annotations_read=False,
        canvas=str(args.canvas),affines_from=str(args.affines_from),device=args.device,threads=args.threads,
        started_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        initialization="identity residual plus same original positive image-only affine for all methods; no neural warm start",
        match_recipe="existing raw confidence-weighted SuperPoint/SuperGlue extraction; no new RANSAC or invented confidence",
        timing_scope="each matcher call includes current executable's image/prewarp/model setup, inference, serialization and wrapper validation; inner setup/inference separately retained. Optimizer complete call includes its loading/features/optimization/export/certificate plus wrapper checks; inner timings are separately retained. DHR complete call includes import/native setup/preprocessing/optimization/export/checks; native timings separate. Sequential in one process; first calls may be cold, no equal-runtime claim.",
        memory_scope="per-executable reported CUDA allocated peak; matcher and optimizer reset internally after setup, so not whole-process/RSS peaks. DHR wrapper observed allocated peak resets before native call. CPU peaks unavailable.",
        cohort_scope="known-specimen direction robustness only; all20 labels previously viewed, not20 independent patients or1600 independent observations",
        native_scope="existing native DHR objective/preprocessing; separate application baseline, no topology repair or hard guarantee")
    report_path=args.output/"predictions.json"
    def persist():
        report["elapsed_seconds"]=time.perf_counter()-started
        report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    persist()  # Cohort and pending cases exist before ANY image/matcher call.
    try:
        sources=_source_rows(args,cohort)
        source_error=None
    except Exception as error:
        sources={};source_error=f"{type(error).__name__}: {error}"
    affines={}
    for pair,row in zip(cohort,rows,strict=True):
        try:
            if source_error:raise ValueError(source_error)
            matrix,offset=_check_input(pair,sources[pair["name"]],side)
            affines[pair["name"]]=(matrix,offset)
            row.update(input_status="ok",affine_determinant=float(np.linalg.det(matrix.astype(np.float64))))
        except Exception as error:
            row.update(input_status="failed",input_error=f"{type(error).__name__}: {error}")
    report["source_input_preparation_seconds"]=time.perf_counter()-started
    persist()
    # All20 raw-match attempts precede every registration method.
    for pair,row in zip(cohort,rows,strict=True):
        record=row["raw_matches"]
        if row["input_status"]!="ok":
            record.update(status="skipped",reason="invalid source inputs");continue
        tick=time.perf_counter()
        try:
            match_args=argparse.Namespace(fixed=pair["fixed"],moving=pair["moving"],affine=pair["affine"],
                output=args.output/record["path"],image_side=side,device=args.device)
            extracted=match_extractor(match_args)
            metadata=_validate_matches(match_args.output,pair,*affines[pair["name"]],side)
            record.update(status="ok",validation=metadata,**{key:extracted.get(key) for key in
                ("setup_seconds","inference_seconds","peak_allocated_bytes","raw_matches","occupied_quadrants_4x4")})
        except Exception as error:
            record.update(status="failed",error=f"{type(error).__name__}: {error}")
        record["complete_call_seconds"]=time.perf_counter()-tick
        persist()
    for pair,row in zip(cohort,rows,strict=True):
        for method in METHODS:
            record=row["methods"][method]
            if row["input_status"]!="ok" or (method!="dhr" and row["raw_matches"]["status"]!="ok"):
                record.update(status="skipped",reason="invalid inputs" if row["input_status"]!="ok" else "raw matcher failed")
                persist();continue
            tick=time.perf_counter()
            try:
                matrix,offset=affines[pair["name"]]
                if method!="dhr":
                    config=make_configuration(pair,method,args,production=production)
                    result=optimizer(config)
                    _check_safe_export(config.output,result,matrix,offset,config.grid_side)
                    expected=config.inner_steps*config.cycles*len(config.levels)*(2 if method=="analytic" else 1)
                    record.update(status="ok",expected_gradient_steps=expected,
                        budget_complete=result["gradient_steps"]==expected and result["failed_trials"]==0,
                        **{key:result.get(key) for key in ("initial","final","gradient_steps","evaluations","objective_evaluations",
                            "failed_trials","saved_binary_certificate","peak_allocated_bytes","optimize_seconds",
                            "loading_seconds","feature_seconds","serialization_seconds","certification_seconds","end_to_end_seconds")})
                else:
                    if args.device=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    dhr_args=argparse.Namespace(fixed=pair["fixed"],moving=pair["moving"],affine=pair["affine"],
                        output=args.output/record["output_directory"],device=args.device,threads=args.threads,image_side=side)
                    dhr_runner(dhr_args)
                    if args.device=="cuda":torch.cuda.synchronize()
                    native=json.loads((args.output/record["report"]).read_text(encoding="utf-8"))
                    for key in ("field","configuration","postprocessing_params"):
                        if not (args.output/record[key]).is_file():raise FileNotFoundError(record[key])
                    if (not np.array_equal(np.asarray(native["post_affine_matrix"],dtype=np.float32),matrix) or
                            not np.array_equal(np.asarray(native["post_affine_offset"],dtype=np.float32),offset)):
                        raise ValueError("native DHR did not retain common affine")
                    record.update(status="ok",wrapper_observed_peak_allocated_bytes=(
                        torch.cuda.max_memory_allocated() if args.device=="cuda" else None),
                        **{key:native.get(key) for key in ("runtime_seconds","nonrigid_seconds","preprocessing_seconds")})
            except Exception as error:
                record.update(status="failed",error=f"{type(error).__name__}: {error}")
            record["complete_call_seconds"]=time.perf_counter()-tick
            persist()
        row["status"]="ok" if row["input_status"]=="ok" and all(
            record["status"]=="ok" for record in row["methods"].values()) else "partial_failure"
        print(json.dumps(dict(name=row["name"],status=row["status"],methods={
            key:value["status"] for key,value in row["methods"].items()})),flush=True)
        persist()
    report.update(prediction_complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        successful_directions=sum(row["status"]=="ok" for row in rows),
        successful_raw_match_directions=sum(row["raw_matches"]["status"]=="ok" for row in rows),
        method_success_counts={method:sum(row["methods"][method]["status"]=="ok" for row in rows) for method in METHODS},
        safe_method_incomplete_budget_counts={method:sum(row["methods"][method].get("budget_complete") is False for row in rows)
            for method in ("analytic","f2")})
    persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("canvas","affines_from","output"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    report=run(parser.parse_args())
    print(json.dumps({key:report[key] for key in ("prediction_complete","cohort_size","successful_directions","method_success_counts","elapsed_seconds")}))


if __name__=="__main__":main()
