"""Frozen SG+MA complement versus matched-strength SG; no rematching or labels."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.coordinated_stain_proxy import source_cases, write_scoring_manifests
from tools.coordinated_matchanything_batch import configuration as original_configuration
from tools.coordinated_matchanything_batch import validate_table, validate_export


CONVENTION="fixed unit q -> affine-aligned moving unit p; full moving point is A p+b"


def fuse_tables(sg,ma):
    """Preserve all rows; each original eligible confidence mass becomes exactly1."""
    basename=lambda p:Path(str(p).replace("\\","/")).name
    for key in ("fixed","moving"):
        if basename(sg[key])!=basename(ma[key]):raise ValueError("same image pair required")
    for record in (sg,ma):
        if (record.get("image_side")!=512 or record.get("coordinate_convention")!=CONVENTION
                or record.get("targets_manual_landmarks_or_dense_teacher_loaded") is not False):
            raise ValueError("same512 affine-aligned pixel-center convention and image-only provenance required")
    for key in ("post_affine_matrix","post_affine_offset"):
        if not np.array_equal(np.asarray(sg[key]),np.asarray(ma[key])):raise ValueError("same frozen affine required")
    a=np.asarray(sg["post_affine_matrix"],dtype=np.float32)
    b=np.asarray(sg["post_affine_offset"],dtype=np.float32)
    if a.shape!=(2,2) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a.astype(float))<=0:
        raise ValueError("finite positive frozen affine required")
    sources=[];targets=[];weights=[];diagnostics=[]
    for name,record in (("sg",sg),("ma",ma)):
        q=np.asarray(record["source_points_unit"],dtype=np.float64)
        p=np.asarray(record["target_points_unit"],dtype=np.float64)
        c=np.asarray(record["confidence"],dtype=np.float64)
        if q.ndim!=2 or q.shape[-1]!=2 or p.shape!=q.shape or c.shape!=(len(q),):
            raise ValueError("Nx2 point rows and N confidences required")
        if any(not np.isfinite(v).all() or (v<0).any() or (v>1).any() for v in (q,p,c)):
            raise ValueError("finite unit coordinates/confidences required; no clipping")
        world=p@a.T+b
        eligible=((world>=0)&(world<=1)).all(-1)
        mass=float((c*eligible).sum())
        if not np.isfinite(mass) or mass<=0 or int(((c>0)&eligible).sum())<8:
            raise ValueError("each matcher requires positive mass and at least8 positive eligible rows")
        normalized=c/mass
        if not np.isfinite(normalized).all() or (normalized>1).any():
            raise ValueError("normalized original confidence exceeds1; incompatible table, no clipping")
        sources.append(q);targets.append(p);weights.append(normalized)
        diagnostics.append(dict(matcher=name,original_rows=len(q),original_eligible_confidence_mass=mass,
            original_positive_eligible_rows=int(((c>0)&eligible).sum()),
            static_ineligible_rows=int((~eligible).sum()),normalized_eligible_mass=float((normalized*eligible).sum())))
    result={key:copy.deepcopy(sg[key]) for key in ("fixed","moving","post_affine_matrix","post_affine_offset","image_side")}
    result.update(status="ok",source_points_unit=np.concatenate(sources).tolist(),
        target_points_unit=np.concatenate(targets).tolist(),confidence=np.concatenate(weights).tolist(),
        coordinate_convention=CONVENTION,targets_manual_landmarks_or_dense_teacher_loaded=False,
        method="all original SG rows then all original MA rows; independently normalized eligible masses",
        composition=diagnostics,match_weight=.2,exact_functional=".1 P_SG + .1 P_MA",
        rows_removed=0,rows_clipped=0)
    return result


def configuration(source,arm,output,fused_matches=None):
    if arm not in ("fusion","sg2"):raise ValueError("fusion or sg2 required")
    if arm=="fusion" and fused_matches is None:raise ValueError("fused table required")
    matches=fused_matches if arm=="fusion" else source["matches"]
    cfg=original_configuration(source,matches,output)
    cfg.match_weight=.2
    return cfg


def run(miit_control,existing_control,ma_predictions,output,*,optimizer=None,export_validator=None):
    if optimizer is None:
        from tools.coordinated_real_case import optimize
        optimizer=optimize
    if export_validator is None:export_validator=validate_export
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cases=source_cases(miit_control,existing_control)
    ma_path=Path(ma_predictions).resolve()
    if ma_path.is_dir():ma_path=ma_path/"predictions.json"
    ma=json.loads(ma_path.read_text(encoding="utf-8"))
    if (ma.get("prediction_complete") is not True or ma.get("annotations_read") is not False
            or [r["name"] for r in ma.get("rows",[])]!=[r["name"] for r in cases]
            or any(r["status"] not in ("ok","failed") for r in ma["rows"])):
        raise ValueError("same25 terminal frozen MA extraction/prediction cohort required")
    output.mkdir(parents=True)
    (output/"fused_tables").mkdir()
    arms={}
    for arm in ("fusion","sg2"):
        (output/arm).mkdir()
        rows=[]
        for case in cases:
            cfg=configuration(case["original_configuration"],arm,output/arm/(case["name"]+"_analytic.npz"),
                output/"fused_tables"/(case["name"]+".json"))
            rows.append(dict(**copy.deepcopy(case),status="pending",output=cfg.output.name,
                report=cfg.output.with_suffix(".json").name,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
        arms[arm]=dict(protocol="frozen SG-preserving complement and matched-strength control",arm=arm,
            cohort_size=25,specimen_count=4,prediction_complete=False,annotations_read=False,rows=rows,
            changed_variables=["matches","match_weight","output"] if arm=="fusion" else ["match_weight","output"])
    outer=dict(protocol="two predeclared25-case arms; all50 terminal before any scoring",attempt_denominator=50,
        prediction_complete=False,composition_complete=False,annotations_read=False,composition_rows=[],
        frozen_ma_predictions=str(ma_path),frozen_ma_historical_setup_seconds=ma.get("model_setup_seconds"),
        frozen_ma_historical_cost_totals=ma.get("cost_totals"),
        cost_scope="current table composition plus optimization; no model loaded/rematching. Historical MA setup/extraction must be retained separately for end-to-end comparison.")
    started=time.perf_counter()
    def persist():
        outer["elapsed_seconds"]=time.perf_counter()-started
        (output/"predictions.json").write_text(json.dumps(outer,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        for arm,manifest in arms.items():
            (output/arm/"predictions.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for index,(case,ma_row) in enumerate(zip(cases,ma["rows"],strict=True)):
        tick=time.perf_counter();composition=dict(name=case["name"],status="pending")
        sg_cfg=configuration(case["original_configuration"],"sg2",output/"sg2"/arms["sg2"]["rows"][index]["output"])
        try:
            validate_table(sg_cfg)
        except Exception as error:
            for arm in arms:arms[arm]["rows"][index].update(status="failed",error=f"SG table invalid: {type(error).__name__}: {error}")
        try:
            if arms["sg2"]["rows"][index]["status"]=="failed":raise ValueError("original SG table invalid")
            if ma_row.get("extraction_status")!="ok":raise ValueError("frozen MA extraction failed; no SG fallback")
            table_path=Path(ma_row["match_table"])
            if not table_path.is_absolute():table_path=ma_path.parent/table_path
            fused=fuse_tables(json.loads(sg_cfg.matches.read_text(encoding="utf-8")),
                json.loads(table_path.read_text(encoding="utf-8")))
            fusion_cfg=configuration(case["original_configuration"],"fusion",output/"fusion"/arms["fusion"]["rows"][index]["output"],
                output/"fused_tables"/(case["name"]+".json"))
            fused.update(original_sg_table=str(sg_cfg.matches),original_ma_table=str(table_path))
            fusion_cfg.matches.write_text(json.dumps(fused,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            metadata=validate_table(fusion_cfg)
            composition.update(status="ok",sources=fused["composition"],combined_point_loader=metadata,
                output=str(fusion_cfg.matches))
        except Exception as error:
            composition.update(status="failed",error=f"{type(error).__name__}: {error}")
            arms["fusion"]["rows"][index].update(status="failed",error=composition["error"])
        composition["complete_call_seconds"]=time.perf_counter()-tick
        outer["composition_rows"].append(composition)
        persist()
    outer["composition_complete"]=True
    outer["composition_seconds"]=sum(r["complete_call_seconds"] for r in outer["composition_rows"])
    persist()
    for arm,manifest in arms.items():
        arm_tick=time.perf_counter()
        for row in manifest["rows"]:
            if row["status"]=="failed":continue
            tick=time.perf_counter()
            try:
                cfg=configuration(row["original_configuration"],arm,output/arm/row["output"],
                    output/"fused_tables"/(row["name"]+".json"))
                result=optimizer(cfg)
                if str(cfg.device).startswith("cuda"):torch.cuda.synchronize()
                ratio=export_validator(cfg,result)
                diagnostics={k:result.get(k) for k in ("gradient_steps","failed_trials","evaluations","objective_evaluations",
                    "saved_binary_certificate","peak_allocated_bytes","initial","final","loading_seconds","feature_seconds",
                    "serialization_seconds","certification_seconds","optimize_seconds","end_to_end_seconds","mind_frame")}
                json.dumps(diagnostics,allow_nan=False)
                row.update(diagnostics,status="ok",budget_complete=True,actual_minimum_corner_ratio=ratio)
            except Exception as error:
                row.update(status="failed",error=f"{type(error).__name__}: {error}")
            row["complete_call_seconds"]=time.perf_counter()-tick
            persist()
            print(json.dumps(dict(arm=arm,name=row["name"],status=row["status"])),flush=True)
        manifest["elapsed_seconds"]=time.perf_counter()-arm_tick
    outer["prediction_complete"]=True
    outer["arms"]={arm:dict(manifest=str(output/arm/"predictions.json"),
        successful=sum(r["status"]=="ok" for r in manifest["rows"]),failed=sum(r["status"]=="failed" for r in manifest["rows"]))
        for arm,manifest in arms.items()}
    for manifest in arms.values():manifest.update(prediction_complete=True,all50_terminal=True)
    persist()
    for arm,manifest in arms.items():
        write_scoring_manifests(manifest,output/arm)
        # Keep MIIT point-table provenance aligned with this arm.
        path=output/arm/"miit_predictions.json"
        value=json.loads(path.read_text(encoding="utf-8"))
        for row in value["rows"]:
            if "raw_matches" in row:row["archived_sg_raw_matches"]=row["raw_matches"]
            row["raw_matches"]=dict(path=row["methods"]["analytic"]["configuration"]["matches"],arm=arm)
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return outer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("miit-control","existing-control","ma-predictions","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    result=run(args.miit_control,args.existing_control,args.ma_predictions,args.output)
    if any(arm["failed"] for arm in result["arms"].values()):raise SystemExit(1)


if __name__=="__main__":main()
