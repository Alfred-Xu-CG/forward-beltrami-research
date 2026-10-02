"""One existing22 analytic control: ONLY original MIND frame -> shared_affine.

Reuses original image/affine/raw-match/configuration archives. No matching,
manual labels, competing field, coupled seed or preconditioner is introduced.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.coordinated_dhr_existing_inputs import existing_rows
from tools.coordinated_lung_all20 import _check_safe_export
from tools.coordinated_f2_floor_compare import _actual_ratio
from tools.coordinated_real_case import optimize


def configuration(source, output):
    values=copy.deepcopy(source)
    required=dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,image_side=512,
                  image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,cycles=1,
                  strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.1,
                  precision="float64",image_precision="float32",interpolation="p1_ac")
    defaults=dict(mind_frame="original",mind_order="transport",image_objective="continuation",capture_prefix="none",
                  control_hierarchy="fixed",coordinate_mode="alternating",proposal_filter_steps=0,fine_patch_cells=0)
    if any(values.get(k)!=v for k,v in required.items()) or any(values.get(k,v)!=v for k,v in defaults.items()):
        raise ValueError("original analytic300 ARAP3 transported-MIND continuation control required")
    for key in ("fixed","moving","affine","matches"):
        if not isinstance(values.get(key),str) or not values[key]:
            raise ValueError("existing original images/affine/raw-match paths required")
        values[key]=Path(values[key])
    values.update(output=Path(output),mind_frame="shared_affine")
    return argparse.Namespace(**values)


def source_cases(lung_predictions, development):
    source=Path(lung_predictions).resolve()
    if source.is_dir():source=source/"predictions.json"
    value=json.loads(source.read_text(encoding="utf-8"))
    expected=[r["name"] for r in existing_rows(Path("unused"))]
    if (value.get("prediction_complete") is not True or value.get("annotations_read") is not False
            or value.get("cohort_size")!=20 or [r.get("name") for r in value.get("rows",[])]!=expected[:20]):
        raise ValueError("original image-only complete ordered lung20 prediction source required")
    result=[]
    for row in value["rows"]:
        old=row["methods"]["analytic"]
        if old.get("status")!="ok":raise ValueError("successful archived analytic source required")
        path=Path(old["report"])
        path=(path if path.is_absolute() else source.parent/path).resolve()
        result.append(dict(name=row["name"],source_report=str(path)))
    for name in expected[20:]:
        result.append(dict(name=name,source_report=str((Path(development)/(name+"_analytic_mind_edge257.json")).resolve())))
    for row in result:
        report=json.loads(Path(row["source_report"]).read_text(encoding="utf-8"))
        if report.get("gradient_steps")!=300 or report.get("failed_trials")!=0:
            raise ValueError("source analytic300 budget must already be complete")
        row["original_configuration"]=report["configuration"]
        configuration(row["original_configuration"],Path("unused.npz"))
    return result


def run(lung_predictions,development,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cases=source_cases(lung_predictions,development)
    rows=[]
    for case in cases:
        cfg=configuration(case["original_configuration"],output/(case["name"]+"_analytic.npz"))
        rows.append(dict(**case,status="pending",output=cfg.output.name,report=cfg.output.with_suffix(".json").name,
                         configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
    manifest=dict(protocol="existing22 corrected shared-affine analytic300 control; no coupled seed or preconditioning",
                  cohort_size=22,prediction_complete=False,annotations_read=False,rows=rows,
                  changed_variables=["mind_frame","output"],
                  source_lung_predictions=str(Path(lung_predictions).resolve()),source_development=str(Path(development).resolve()),
                  unchanged="same original images, affine, frozen raw matches, P1ac257, ARAP3/shape1e-4/points.1,300gradients,identity residual start")
    output.mkdir(parents=True);started=time.perf_counter()
    def persist():
        manifest["elapsed_seconds"]=time.perf_counter()-started
        (output/"predictions.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for row in rows:
        tick=time.perf_counter()
        try:
            cfg=configuration(row["original_configuration"],output/row["output"])
            for key in ("fixed","moving","affine","matches"):
                if not getattr(cfg,key).is_file():raise FileNotFoundError(getattr(cfg,key))
            result=optimize(cfg)
            if str(cfg.device).startswith("cuda"):torch.cuda.synchronize()
            with np.load(cfg.affine,allow_pickle=False) as saved:
                a,b=saved["post_affine_matrix"],saved["post_affine_offset"]
            _check_safe_export(cfg.output,result,a,b,cfg.grid_side)
            ratio=_actual_ratio(cfg.output)
            diagnostics={key:result.get(key) for key in ("gradient_steps","failed_trials","evaluations","objective_evaluations","saved_binary_certificate",
                        "peak_allocated_bytes","initial","final","optimize_seconds","end_to_end_seconds","mind_frame")}
            json.dumps(diagnostics,allow_nan=False)
            row.update(diagnostics)
            row["actual_minimum_corner_ratio"]=ratio
            if ratio<=cfg.minimum_jacobian or result["gradient_steps"]!=300 or result["failed_trials"]!=0:
                raise ValueError("actual strict floor and complete300-gradient budget required")
            row.update(status="ok",budget_complete=True)
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
        row["complete_call_seconds"]=time.perf_counter()-tick
        persist();print(json.dumps(dict(name=row["name"],status=row["status"])),flush=True)
    manifest["prediction_complete"]=True;persist();return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("lung-predictions","development","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    result=run(args.lung_predictions,args.development,args.output)
    if any(row["status"]!="ok" for row in result["rows"]):raise SystemExit(1)


if __name__=="__main__":main()
