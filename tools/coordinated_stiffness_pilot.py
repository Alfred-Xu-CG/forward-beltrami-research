"""Corrected-MIND Adam900 or stiffness300 on the same three MIIT inputs.

Prediction is image-only. Archived F2/DHR remain context, never initialization.
The stiffness application is imported lazily so the Adam control is independent.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from tools.coordinated_miit_score import load_manifest
from tools.coordinated_miit_multiscale_pilot import configuration as original_configuration
from tools.coordinated_real_case import optimize
from tools.coordinated_lung_all20 import _check_safe_export
from tools.coordinated_lung_all20_score import _affine
from tools.coordinated_miit_transfer import _finite_json, _actual_ratio


def configuration(report, output, arm, *, device=None, threads=None, inputs=None):
    if arm not in ("adam900", "stiffness300"):
        raise ValueError("declare adam900 or stiffness300")
    cfg = original_configuration(report, output, "continuation", device=device,
        threads=threads, inputs=inputs, data_term="mind")
    if (cfg.method != "analytic" or cfg.cycles != 1 or cfg.grid_side != 257
            or cfg.image_side != 512 or cfg.levels != [17,33,65,129,257]
            or cfg.image_levels != [32,64,128,256,512]
            or cfg.strain_model != "p1_arap" or cfg.strain_weight != 3.
            or cfg.interpolation != "p1_ac" or cfg.precision != "float64"
            or cfg.image_precision != "float32" or cfg.shape_weight != 1e-4
            or cfg.match_weight != .1 or cfg.oob_weight != 1.
            or cfg.loss != "mind" or cfg.mind_order != "transport"
            or cfg.preprocessing != "raw_inverted" or cfg.minimum_jacobian != .001
            or cfg.control_hierarchy != "fixed" or cfg.coordinate_mode != "alternating"
            or getattr(cfg,"fixed_mask",None) is not None
            or getattr(cfg,"image_weight",1.) != 1.
            or getattr(cfg,"proposal_filter_steps",0) != 0
            or getattr(cfg,"fine_patch_cells",0) != 0
            or getattr(cfg,"capture_prefix","none") != "none"):
        raise ValueError("unchanged declared MIIT analytic recipe required")
    cfg.inner_steps = 90 if arm == "adam900" else 30
    cfg.inner_steps_by_level = [cfg.inner_steps]*5
    return cfg


def run(source, output, arm, *, device=None, threads=None, inputs=None):
    if arm not in ("adam900", "stiffness300"):
        raise ValueError("declare adam900 or stiffness300")
    source, output = Path(source), Path(output)
    manifest, directory = load_manifest(source)
    if output.exists(): raise FileExistsError(output)
    prefix = Path(os.path.relpath(directory.resolve(), output.resolve()))
    def relocate(value):
        path=Path(value)
        return str(path if path.is_absolute() else prefix/path)
    rows=copy.deepcopy(manifest["rows"])
    for row in rows:
        for key in ("fixed","moving","affine","layout"): row[key]=relocate(row[key])
        for record,key in ((row.get("raw_matches",{}),"path"),
                           (row.get("affine_estimation",{}),"report")):
            if key in record: record[key]=relocate(record[key])
        for method in ("f2","dhr"):
            record=row["methods"][method]
            for key in ("output","report","output_directory","field","configuration","postprocessing_params"):
                if isinstance(record.get(key),str): record[key]=relocate(record[key])
            record["pilot_reference"]="unchanged archived context; NOT rerun or used as initialization"
        old=row["methods"]["analytic"]
        row["archived_analytic300"]={key:relocate(old[key]) for key in ("output","report")}
        row["methods"]["analytic"]=dict(status="pending",output=row["name"]+"_analytic.npz",
            report=row["name"]+"_analytic.json")
        row["status"]="pending"
    report=dict(protocol="ONE same-functional ARAP-stiffness optimizer comparison",arm=arm,
        prediction_complete=False,annotations_read=False,label_informed_development=True,
        source_predictions=str(source),cohort_size=3,rows=rows,
        unchanged="same shared-affine MIND, raw support, original affine/matches, ARAP3, shape1e-4, matches.1, OOB, levels and best-full selection",
        changed_variable="Adam900 budget control" if arm=="adam900" else "physical-fiber stiffness-preconditioned descent, at most300 gradients",
        timing="complete calls and actual gradient/forward counts; budgets do NOT imply equal compute",
        global_operator="stiffness arm uses global separable spectral inverse; safe decoder itself is unchanged")
    output.mkdir(parents=True); started=time.perf_counter()
    def persist():
        report["elapsed_seconds"]=time.perf_counter()-started
        (output/"predictions.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for original,row in zip(manifest["rows"],rows,strict=True):
        record=row["methods"]["analytic"]; tick=time.perf_counter()
        try:
            old=original["methods"]["analytic"]
            if old["status"]!="ok": raise ValueError("successful original analytic reference required")
            path=Path(old["report"]); path=path if path.is_absolute() else directory/path
            cfg=configuration(json.loads(path.read_text(encoding="utf-8")),output/record["output"],arm,
                device=device,threads=threads,inputs=inputs)
            if arm=="adam900": result=optimize(cfg)
            else:
                from tools.coordinated_stiffness_application import optimize_stiffness
                result=optimize_stiffness(cfg)
            if cfg.device.startswith("cuda"): torch.cuda.synchronize()
            a,b=_affine(cfg.affine); _check_safe_export(cfg.output,result,a,b,cfg.grid_side)
            clean,bad=_finite_json({key:result.get(key) for key in (
                "initial","final","gradient_steps","evaluations","objective_evaluations","failed_trials",
                "saved_binary_certificate","peak_allocated_bytes","feature_seconds","loading_seconds",
                "serialization_seconds","certification_seconds","optimize_seconds","end_to_end_seconds",
                "mind_frame","accepted_steps","backtracks","stage_stop_reasons","selected_stage")})
            record.update(clean)
            if bad: raise ValueError("nonfinite optimizer diagnostics")
            ratio=_actual_ratio(cfg.output)
            record.update(actual_minimum_corner_ratio=ratio,actual_strict_floor_valid=ratio>cfg.minimum_jacobian)
            if not ratio>cfg.minimum_jacobian: raise ValueError("strict actual floor required")
            gradients=result["gradient_steps"]
            if arm=="adam900" and (gradients!=900 or result["failed_trials"]!=0):
                raise ValueError("Adam900 did not complete its declared budget")
            if arm=="stiffness300" and not 0<=gradients<=300:
                raise ValueError("stiffness actual gradient budget outside0..300")
            record.update(status="ok",budget_complete=gradients==(900 if arm=="adam900" else 300),
                expected_maximum_gradient_steps=900 if arm=="adam900" else 300,
                limited_or_early_stage_stops=arm=="stiffness300" and gradients<300)
        except Exception as error:
            record.update(status="failed",error=f"{type(error).__name__}: {error}")
        record["complete_call_seconds"]=time.perf_counter()-tick
        row["status"]="ok" if all(v["status"]=="ok" for v in row["methods"].values()) else "partial_failure"
        persist(); print(json.dumps(dict(name=row["name"],arm=arm,status=record["status"])),flush=True)
    report["prediction_complete"]=True; persist(); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--arm",choices=("adam900","stiffness300"),required=True)
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    report=run(args.predictions,args.output,args.arm,device=args.device,threads=args.threads,inputs=args.inputs)
    if any(row["status"]!="ok" for row in report["rows"]): raise SystemExit(1)


if __name__=="__main__": main()
