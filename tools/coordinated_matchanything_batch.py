"""Substitute ONE frozen MatchAnything point table in all25 shared-gray controls.

Load the model once, finish all extraction attempts, release it, then optimize.
No labels, SG fallback, new affine, or new image functional are introduced.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.coordinated_stain_proxy import source_cases, write_scoring_manifests


def configuration(source, matches, output):
    """Preserve the original shared-gray control; only point table/output change."""
    values = copy.deepcopy(source)
    required = dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,
        image_side=512,image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,
        cycles=1,strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.1,
        oob_weight=1.,minimum_jacobian=.001,precision="float64",image_precision="float32",
        interpolation="p1_ac",mind_frame="shared_affine")
    defaults = dict(preprocessing="raw_inverted",mind_order="transport",image_weight=1.,
        seed_initializer="identity",fixed_mask=None,image_objective="continuation",capture_prefix="none",
        control_hierarchy="fixed",coordinate_mode="alternating",proposal_filter_steps=0,
        fine_patch_cells=0,inner_steps_by_level=[30]*5,match_robust_scale=8.)
    if any(values.get(k)!=v for k,v in required.items()) or any(values.get(k,v)!=v for k,v in defaults.items()):
        raise ValueError("original shared-gray identity-start analytic300 with point weight.1/scale8 required")
    for key in ("fixed","moving","affine","matches"):
        if not isinstance(values.get(key),str) or not values[key]:
            raise ValueError("original fixed/moving/affine/matches paths required")
        values[key] = Path(values[key])
    values.update(matches=Path(matches),output=Path(output))
    return argparse.Namespace(**values)


def validate_table(cfg):
    """Enforce the actual production loader before calling any optimizer."""
    from tools.coordinated_real_case import load_image_matches
    with np.load(cfg.affine,allow_pickle=False) as saved:
        a=np.asarray(saved["post_affine_matrix"],dtype=np.float32)
        b=np.asarray(saved["post_affine_offset"],dtype=np.float32)
    _,metadata=load_image_matches(cfg.matches,a,b,fixed_path=cfg.fixed,moving_path=cfg.moving,
        image_side=cfg.image_side,device="cpu",dtype=torch.float64,robust_scale=8.)
    return metadata


def validate_export(cfg,result):
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    with np.load(cfg.affine,allow_pickle=False) as saved:
        a,b=saved["post_affine_matrix"],saved["post_affine_offset"]
    _check_safe_export(cfg.output,result,a,b,cfg.grid_side)
    ratio=_actual_ratio(cfg.output)
    if ratio<=cfg.minimum_jacobian or result["gradient_steps"]!=300 or result["failed_trials"]!=0:
        raise ValueError("strict exported floor and complete300 gradients required")
    return ratio


def run(miit_control,existing_control,output,source_root,checkpoint,*,model_factory=None,
        optimizer=None,export_validator=None):
    """Injected model/optimizer are for local order/failure tests, not fallback routes."""
    if model_factory is None:
        from tools.coordinated_matchanything import FrozenMatchAnything
        model_factory=FrozenMatchAnything
    if optimizer is None:
        from tools.coordinated_real_case import optimize
        optimizer=optimize
    if export_validator is None: export_validator=validate_export
    started=time.perf_counter()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    rows=source_cases(miit_control,existing_control)
    for row in rows:
        cfg=configuration(row["original_configuration"],output/(row["name"]+"_matchanything.json"),
            output/(row["name"]+"_analytic.npz"))
        row.update(status="pending",extraction_status="pending",optimization_status="pending",
            output=cfg.output.name,report=cfg.output.with_suffix(".json").name,
            match_table=cfg.matches.name,
            configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()})
    devices={r["configuration"]["device"] for r in rows}
    if len(devices)!=1:raise ValueError("one original device for this serial batch required")
    device=next(iter(devices))
    cuda=str(device).startswith("cuda")
    manifest=dict(protocol="ONE frozen MatchAnything ELoFTR point substitution in shared-gray analytic300",
        cohort_size=25,specimen_count=4,prediction_complete=False,extraction_complete=False,
        annotations_read=False,changed_variables=["matches","output"],rows=rows,
        matcher_preprocessing="ordinary PIL grayscale, once affine prewarped moving; original SG used inverted gray",
        scoring_policy="all25 extraction terminal before any optimization; all25 prediction terminal before scoring",
        memory_scope="one model resident during extraction; explicitly closed and released before optimizer calls",
        timing_scope="one cold model setup; per-pair extraction includes table serialization/validation; optimization includes export",
        objective_caution="new point observations change E; old/new total energies are not same-functional convergence")
    output.mkdir(parents=True)
    serialization_seconds=0.
    def persist():
        nonlocal serialization_seconds
        tick=time.perf_counter()
        manifest["elapsed_seconds"]=time.perf_counter()-started
        manifest["manifest_serialization_seconds"]=serialization_seconds
        (output/"predictions.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        serialization_seconds+=time.perf_counter()-tick
    def sync():
        if cuda:torch.cuda.synchronize(device)
    persist()
    model=None
    setup_tick=time.perf_counter()
    try:
        if cuda:torch.cuda.reset_peak_memory_stats(device)
        model=model_factory(Path(source_root),Path(checkpoint),device=device)
        sync()
        manifest["model_setup"]=copy.deepcopy(model.setup_report)
        manifest["model_setup_status"]="ok"
    except Exception as error:
        manifest.update(model_setup_status="failed",model_setup_error=f"{type(error).__name__}: {error}")
    manifest["model_setup_seconds"]=time.perf_counter()-setup_tick
    manifest["model_setup_peak_allocated_bytes"]=int(torch.cuda.max_memory_allocated(device)) if cuda else None
    persist()
    extraction_started=time.perf_counter()
    extraction_calls=0
    try:
        for row in rows:
            tick=time.perf_counter()
            try:
                if manifest["model_setup_status"]!="ok":raise RuntimeError(manifest["model_setup_error"])
                cfg=configuration(row["original_configuration"],output/row["match_table"],output/row["output"])
                if cuda:torch.cuda.reset_peak_memory_stats(device)
                extraction_calls+=1
                record=model.extract(cfg.fixed,cfg.moving,cfg.affine,output=cfg.matches)
                sync()
                # Point arrays are already saved in the ordinary table; avoid duplicating them25times.
                extraction={k:v for k,v in record.items() if k not in
                    ("source_points_unit","target_points_unit","confidence")}
                json.dumps(extraction,allow_nan=False)
                row["extraction"]=extraction
                row["point_loader_metadata"]=validate_table(cfg)
                if record.get("status")!="ok":raise ValueError("matcher did not declare eligible extraction")
                row["extraction_status"]="ok"
            except Exception as error:
                row.update(status="failed",extraction_status="failed",optimization_status="skipped",
                    error=f"{type(error).__name__}: {error}")
            row["extraction_complete_call_seconds"]=time.perf_counter()-tick
            row["extraction_peak_allocated_bytes"]=int(torch.cuda.max_memory_allocated(device)) if cuda else None
            persist()
            print(json.dumps(dict(name=row["name"],phase="extraction",status=row["extraction_status"])),flush=True)
    finally:
        release_tick=time.perf_counter()
        if model is not None:model.close()
        model=None
        gc.collect()
        if cuda:torch.cuda.empty_cache()
        sync()
        manifest["model_release_seconds"]=time.perf_counter()-release_tick
        manifest["model_released_before_optimization"]=True
    manifest["extraction_wall_seconds"]=time.perf_counter()-extraction_started
    manifest["extraction_complete"]=True
    persist()
    optimization_started=time.perf_counter()
    for row in rows:
        if row["extraction_status"]!="ok":continue
        tick=time.perf_counter()
        try:
            cfg=configuration(row["original_configuration"],output/row["match_table"],output/row["output"])
            result=optimizer(cfg)
            sync()
            ratio=export_validator(cfg,result)
            diagnostics={k:result.get(k) for k in ("gradient_steps","failed_trials","evaluations",
                "objective_evaluations","saved_binary_certificate","peak_allocated_bytes","initial","final",
                "loading_seconds","feature_seconds","serialization_seconds","certification_seconds",
                "optimize_seconds","end_to_end_seconds","mind_frame","image_preprocessing")}
            json.dumps(diagnostics,allow_nan=False)
            row.update(diagnostics,actual_minimum_corner_ratio=ratio,status="ok",optimization_status="ok",budget_complete=True)
        except Exception as error:
            row.update(status="failed",optimization_status="failed",error=f"{type(error).__name__}: {error}")
        row["optimization_complete_call_seconds"]=time.perf_counter()-tick
        row["complete_call_seconds"]=row["extraction_complete_call_seconds"]+row["optimization_complete_call_seconds"]
        persist()
        print(json.dumps(dict(name=row["name"],phase="optimization",status=row["status"])),flush=True)
    for row in rows:
        row.setdefault("complete_call_seconds",row["extraction_complete_call_seconds"])
    manifest["optimization_wall_seconds"]=time.perf_counter()-optimization_started
    manifest["prediction_complete"]=True
    manifest["cost_totals"]=dict(model_setup_count=1,
        extraction_attempts=sum(r["extraction_status"] in ("ok","failed") for r in rows),
        extraction_calls=extraction_calls,
        optimizer_calls=sum(r["optimization_status"] in ("ok","failed") for r in rows),
        extraction_complete_call_seconds=sum(r["extraction_complete_call_seconds"] for r in rows),
        optimization_complete_call_seconds=sum(r.get("optimization_complete_call_seconds",0.) for r in rows))
    persist()
    write_scoring_manifests(manifest,output)
    # The reused MIIT adapter retains source SG context; make its point provenance
    # explicit now that this experiment substitutes the table itself.
    miit_path=output/"miit_predictions.json"
    miit=json.loads(miit_path.read_text(encoding="utf-8"))
    for row in miit["rows"]:
        if "raw_matches" in row:row["archived_sg_raw_matches"]=row["raw_matches"]
        analytic=row["methods"]["analytic"]
        row["raw_matches"]=dict(path=analytic["match_table"],status=analytic["extraction_status"],
            matcher="MatchAnything ELoFTR",targets_manual_landmarks_or_dense_teacher_loaded=False)
    miit_path.write_text(json.dumps(miit,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("miit-control","existing-control","output","source-root","checkpoint"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    result=run(args.miit_control,args.existing_control,args.output,args.source_root,args.checkpoint)
    if any(r["status"]!="ok" for r in result["rows"]):raise SystemExit(1)


if __name__=="__main__":main()
