"""One paired three-case MS objective experiment; prediction never reads labels."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from tools.coordinated_miit_score import load_manifest
from tools.coordinated_lung_all20 import _check_safe_export
from tools.coordinated_lung_all20_score import _affine
from tools.coordinated_miit_transfer import _finite_json,_actual_ratio
from tools.coordinated_real_case import optimize


def configuration(report,output,objective,*,device=None,threads=None,inputs=None,fixed_mask=None,data_term="mind"):
    values=copy.deepcopy(report["configuration"])
    if values.get("method") not in ("analytic","f2") or values.get("output_selection")!="best_full":
        raise ValueError("original analytic/F2 best-full recipe required")
    for key in ("fixed","moving","affine","matches"):
        if values.get(key) is not None:
            value=Path(values[key])
            values[key]=Path(inputs)/value.name if inputs is not None else value
    values.update(output=Path(output),mind_frame="shared_affine",image_objective=objective)
    if data_term not in ("mind","ngf"):raise ValueError("mind or ngf data term required")
    if data_term=="ngf":
        if objective!="continuation" or fixed_mask is not None or values.get("fixed_mask") is not None:
            raise ValueError("NGF pilot requires original raw support and ordinary continuation")
        values.update(loss="ngf",mind_frame="original",mind_order="transport")
    if fixed_mask is not None:
        if objective!="continuation":raise ValueError("fixed-support pilot changes support only; continuation objective required")
        values["fixed_mask"]=Path(fixed_mask)
    if device is not None:values["device"]=device
    if threads is not None:values["threads"]=threads
    return argparse.Namespace(**values)


def run(source,output,objective,*,device=None,threads=None,inputs=None,fixed_supports=None,data_term="mind"):
    source=Path(source);output=Path(output)
    manifest,directory=load_manifest(source)
    if data_term not in ("mind","ngf") or (data_term=="ngf" and
            (objective!="continuation" or fixed_supports is not None)):
        raise ValueError("NGF requires ordinary continuation and original raw support")
    support_paths=None
    if fixed_supports is not None:
        if objective!="continuation":raise ValueError("fixed-support pilot changes support only; continuation objective required")
        from tools.coordinated_tissue_support import load_support_paths
        support_paths=load_support_paths(fixed_supports)
    if output.exists():raise FileExistsError(output)
    prefix=Path(os.path.relpath(directory.resolve(),output.resolve()))
    def relocate(value):
        p=Path(value);return str(p if p.is_absolute() else prefix/p)
    rows=copy.deepcopy(manifest["rows"])
    for row in rows:
        for key in ("fixed","moving","affine","layout"):row[key]=relocate(row[key])
        for key in ("path",):
            if key in row.get("raw_matches",{}):row["raw_matches"][key]=relocate(row["raw_matches"][key])
        for key in ("report",):
            if key in row.get("affine_estimation",{}):row["affine_estimation"][key]=relocate(row["affine_estimation"][key])
        native=row["methods"]["dhr"]
        for key in ("output_directory","report","field","configuration","postprocessing_params"):
            if isinstance(native.get(key),str):native[key]=relocate(native[key])
        native["pilot_reference"]="unchanged archived DHR; NOT rerun"
        for method in ("analytic","f2"):
            row["methods"][method]=dict(status="pending",output=row["name"]+"_"+method+".npz",
                report=row["name"]+"_"+method+".json")
        row["status"]="pending"
    report=dict(protocol="ONE equal-weight simultaneous multiscale objective pilot",cohort_size=3,
        prediction_complete=False,annotations_read=False,label_informed_development=True,
        source_predictions=str(source),rows=rows,image_objective=objective,
        shared_frame="shared_affine in BOTH paired arms",all_scales=[32,64,128,256,512],
        changed_variable="MS image sum at EVERY stage and final selection versus original raster continuation/full512 selector",
        unchanged="same affine, boundary, four-corner floor, ARAP3, shape1e-4, matches.1, rates and 300 gradients",
        objective_caution="different functionals/schedules; totals are not matched-energy or pure-optimizer evidence",
        timing="complete serial calls include setup/export; single calls, no warm ABBA speed claim")
    if support_paths is not None:
        report.update(protocol="ONE released fixed-tissue-support pilot with original raster continuation",
            supplied_tissue_annotation=True,evidence_scope="images plus released semi-manual fixed tissue masks; no manual correspondence landmarks",
            fixed_supports=str(fixed_supports),
            changed_variable="fixed support changes from original grayscale threshold to released semi-manual tissue; no moving overlap filter",
            objective_caution="ROI evidence ablation, not novel optimizer evidence; evaluation labels unchanged",
            annotation_flag_scope="annotations_read=False refers to evaluation correspondence coordinates; supplied tissue masks ARE annotations")
    if data_term=="ngf":
        report.update(protocol="ONE FAIR-style postwarp intensity NGF pilot",data_term="ngf",
            shared_frame="same original affine; NGF samples original intensity ONCE then takes fixed-frame gradients",
            changed_variable="shared-affine transported MIND replaced by squared-dot NGF of warped original intensity; same raster continuation",
            objective_caution="different data functional and sampling order; not pure geometry/optimizer evidence",
            ngf_edge_rule="frozen per-scale max(side/255, .1 fixed-mask mean gradient norm), fixed and initial-warped moving separately",
            supplied_tissue_annotation=False)
    output.mkdir(parents=True);started=time.perf_counter()
    def persist():
        report["elapsed_seconds"]=time.perf_counter()-started
        (output/"predictions.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for original,row in zip(manifest["rows"],rows,strict=True):
        for method in ("analytic","f2"):
            record=row["methods"][method];tick=time.perf_counter()
            try:
                old=original["methods"][method]
                if old["status"]!="ok":raise ValueError("successful source prediction required")
                path=Path(old["report"]);path=path if path.is_absolute() else directory/path
                cfg=configuration(json.loads(path.read_text(encoding="utf-8")),output/record["output"],objective,
                    device=device,threads=threads,inputs=inputs,
                    fixed_mask=support_paths[original["name"]] if support_paths is not None else None,data_term=data_term)
                if cfg.image_levels!=[32,64,128,256,512] or cfg.levels!=[17,33,65,129,257]:
                    raise ValueError("declared production raster/control levels required")
                result=optimize(cfg)
                if cfg.device.startswith("cuda"):torch.cuda.synchronize()
                a,b=_affine(cfg.affine);_check_safe_export(cfg.output,result,a,b,cfg.grid_side)
                clean,bad=_finite_json({key:result.get(key) for key in (
                    "initial","final","gradient_steps","evaluations","objective_evaluations","failed_trials",
                    "saved_binary_certificate","peak_allocated_bytes","feature_seconds","loading_seconds",
                    "serialization_seconds","certification_seconds","optimize_seconds","end_to_end_seconds",
                    "mind_frame","image_objective","image_objective_scales","image_objective_weights","ngf_by_resolution")})
                record.update(clean)
                if bad:raise ValueError("nonfinite optimizer diagnostics")
                ratio=_actual_ratio(cfg.output)
                record.update(actual_minimum_corner_ratio=ratio,actual_strict_floor_valid=ratio>cfg.minimum_jacobian)
                if not ratio>cfg.minimum_jacobian:raise ValueError("strict actual floor required")
                if result["gradient_steps"]!=300 or result["failed_trials"]!=0:
                    raise ValueError("declared 300-gradient budget did not complete")
                record.update(status="ok",budget_complete=True,expected_gradient_steps=300)
            except Exception as error:
                record.update(status="failed",error=f"{type(error).__name__}: {error}")
            record["complete_call_seconds"]=time.perf_counter()-tick
            persist();print(json.dumps(dict(name=row["name"],method=method,status=record["status"])),flush=True)
        row["status"]="ok" if all(r["status"]=="ok" for r in row["methods"].values()) else "partial_failure"
        persist()
    report["prediction_complete"]=True;persist();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--image-objective",choices=("continuation","simultaneous_multiscale"),required=True)
    parser.add_argument("--inputs",type=Path,help="optional same-input directory relocation, basenames preserved")
    parser.add_argument("--fixed-supports",type=Path,help="optional prepared support.json; changes only fixed support, requires continuation")
    parser.add_argument("--data-term",choices=("mind","ngf"),default="mind",
        help="one NGF substitution with original raw support/continuation; default keeps previous pilots")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    result=run(args.predictions,args.output,args.image_objective,device=args.device,threads=args.threads,inputs=args.inputs,fixed_supports=args.fixed_supports,data_term=args.data_term)
    if any(row["status"]!="ok" for row in result["rows"]):raise SystemExit(1)


if __name__=="__main__":main()
