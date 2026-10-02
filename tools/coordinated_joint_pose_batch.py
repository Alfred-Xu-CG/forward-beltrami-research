"""Fresh paired frozen250/joint300 comparison on the archived25 fusion inputs.

No matching, model loading, label access or per-case recipe selection. Both arms
finish all attempts before scorer-compatible manifests become available.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.coordinated_matchanything_batch import configuration as validate_original
from tools.coordinated_stain_proxy import write_scoring_manifests


ARMS={"frozen250":250,"joint300":300}


def configuration(source,arm,output):
    if arm not in ARMS:raise ValueError("frozen250 or joint300 arm required")
    if source.get("match_weight")!=.2:raise ValueError("archived independently normalized SG+MA fusion required")
    original=copy.deepcopy(source);original["match_weight"]=.1
    cfg=validate_original(original,source["matches"],output)
    cfg.match_weight=.2
    cfg.inner_steps=25
    cfg.inner_steps_by_level=[25]*5
    cfg.pose_mode="joint_positive_affine" if arm=="joint300" else "frozen_affine"
    if arm=="joint300":cfg.pose_steps_per_level=10
    return cfg


def source_cases(fusion_predictions):
    path=Path(fusion_predictions).resolve()
    if path.is_dir():path=path/"predictions.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    from tools.coordinated_dhr_existing_inputs import existing_rows
    expected=["miit_2_to_3","miit_7_to_8","miit_10_to_11"]+[r["name"] for r in existing_rows(Path("unused"))]
    rows=manifest.get("rows",[])
    if (manifest.get("prediction_complete") is not True or manifest.get("all50_terminal") is not True
            or manifest.get("annotations_read") is not False or manifest.get("arm")!="fusion"
            or [r["name"] for r in rows]!=expected):
        raise ValueError("complete ordered archived fusion25 cohort required")
    result=[]
    for row in rows:
        if row.get("status")!="ok" or row.get("gradient_steps")!=300 or row.get("failed_trials")!=0:
            raise ValueError("successful archived fusion300 control required")
        values=row["configuration"]
        configuration(values,"frozen250","unused.npz")
        report=Path(row["report"])
        if not report.is_absolute():report=path.parent/report
        result.append(dict(name=row["name"],cohort=row["cohort"],source_manifest=row["source_manifest"],
            source_report=str(report),original_configuration=copy.deepcopy(values)))
    return result


def validate_frozen_export(cfg,result):
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    with np.load(cfg.affine,allow_pickle=False) as saved:
        a,b=saved["post_affine_matrix"],saved["post_affine_offset"]
    _check_safe_export(cfg.output,result,a,b,cfg.grid_side)
    ratio=_actual_ratio(cfg.output)
    if ratio<=cfg.minimum_jacobian:raise ValueError("actual strict frozen-affine floor required")
    return dict(valid=True,original_affine_normalized_minimum_corner_ratio=ratio,
        residual_minimum_corner_ratio=ratio,pose_mode="frozen_affine")


def run(fusion_predictions,output,*,frozen_optimizer=None,joint_optimizer=None,
        frozen_validator=None,joint_validator=None):
    if frozen_optimizer is None:
        from tools.coordinated_real_case import optimize
        frozen_optimizer=optimize
    if joint_optimizer is None:
        from tools.coordinated_joint_pose import optimize_joint_pose
        joint_optimizer=optimize_joint_pose
    if frozen_validator is None:frozen_validator=validate_frozen_export
    if joint_validator is None:
        from tools.coordinated_joint_pose import validate_joint_export
        joint_validator=validate_joint_export
    cases=source_cases(fusion_predictions)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    manifests={}
    for arm,budget in ARMS.items():
        (output/arm).mkdir()
        rows=[]
        for case in cases:
            cfg=configuration(case["original_configuration"],arm,output/arm/(case["name"]+"_analytic.npz"))
            rows.append(dict(**copy.deepcopy(case),status="pending",output=cfg.output.name,
                report=cfg.output.with_suffix(".json").name,expected_gradient_steps=budget,pose_mode=cfg.pose_mode,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
        manifests[arm]=dict(protocol="paired joint positive global pose plus hard P1 residual",arm=arm,
            cohort_size=25,specimen_count=4,prediction_complete=False,annotations_read=False,rows=rows,
            expected_gradient_steps=budget,expected_residual_gradient_steps=250,
            expected_pose_gradient_steps=50 if arm=="joint300" else 0,
            changed_variables=["inner_steps","inner_steps_by_level","output","pose_mode"]+
                (["pose_steps_per_level"] if arm=="joint300" else []),
            evidence="unchanged frozen SG+MA table .2; shared-gray MIND; original images/affine/eligibility/support")
    outer=dict(protocol="all50 paired frozen250/joint300 attempts before scoring",attempt_denominator=50,
        prediction_complete=False,annotations_read=False,source_fusion_predictions=str(Path(fusion_predictions).resolve()),
        budget_scope="frozen250=250 residual; joint300=250 residual+50 pose; not equal compute",
        cost_scope="fresh optimizer calls only; frozen affine/matcher and archived fusion-table costs excluded, not zero end-to-end")
    started=time.perf_counter()
    def persist():
        outer["elapsed_seconds"]=time.perf_counter()-started
        (output/"predictions.json").write_text(json.dumps(outer,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        for arm,manifest in manifests.items():
            (output/arm/"predictions.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for arm,manifest in manifests.items():
        tick_arm=time.perf_counter()
        for row in manifest["rows"]:
            tick=time.perf_counter()
            try:
                cfg=configuration(row["original_configuration"],arm,output/arm/row["output"])
                for key in ("fixed","moving","affine","matches"):
                    if not getattr(cfg,key).is_file():raise FileNotFoundError(getattr(cfg,key))
                result=(joint_optimizer if arm=="joint300" else frozen_optimizer)(cfg)
                if str(cfg.device).startswith("cuda"):torch.cuda.synchronize()
                if result.get("gradient_steps")!=ARMS[arm] or result.get("failed_trials")!=0:
                    raise ValueError("declared full gradient budget without failed trials required")
                if arm=="joint300":
                    if result.get("residual_gradient_steps")!=250 or result.get("pose_gradient_steps")!=50:
                        raise ValueError("joint budget requires250 residual plus50 pose gradients")
                    validation=joint_validator(cfg.output,minimum_jacobian=cfg.minimum_jacobian)
                else:validation=frozen_validator(cfg,result)
                if validation.get("valid") is not True:raise ValueError("actual saved-map validation failed")
                diagnostics={key:result.get(key) for key in ("gradient_steps","failed_trials","evaluations",
                    "objective_evaluations","saved_binary_certificate","peak_allocated_bytes","initial","final",
                    "loading_seconds","feature_seconds","serialization_seconds","certification_seconds",
                    "optimize_seconds","end_to_end_seconds","mind_frame","pose_parameters","pose_history")}
                diagnostics.update(export_validation=validation,residual_gradient_steps=250,
                    pose_gradient_steps=50 if arm=="joint300" else 0,
                    actual_minimum_corner_ratio=validation.get("original_affine_normalized_minimum_corner_ratio"))
                json.dumps(diagnostics,allow_nan=False)
                row.update(diagnostics,status="ok",budget_complete=True)
            except Exception as error:
                row.update(status="failed",error=f"{type(error).__name__}: {error}")
            row["complete_call_seconds"]=time.perf_counter()-tick
            persist()
            print(json.dumps(dict(arm=arm,name=row["name"],status=row["status"])),flush=True)
        manifest["elapsed_seconds"]=time.perf_counter()-tick_arm
    outer["prediction_complete"]=True
    outer["arms"]={arm:dict(manifest=str(output/arm/"predictions.json"),expected_gradient_steps=ARMS[arm],
        successful=sum(r["status"]=="ok" for r in manifest["rows"]),failed=sum(r["status"]=="failed" for r in manifest["rows"]))
        for arm,manifest in manifests.items()}
    for manifest in manifests.values():manifest.update(prediction_complete=True,all50_terminal=True)
    persist()
    for arm,manifest in manifests.items():
        write_scoring_manifests(manifest,output/arm)
        for subgroup in ("miit","existing"):
            path=output/arm/(subgroup+"_predictions.json")
            value=json.loads(path.read_text(encoding="utf-8"))
            value.update(all50_terminal=True,expected_gradient_steps=ARMS[arm],
                pose_mode="joint_positive_affine" if arm=="joint300" else "frozen_affine")
            if subgroup=="miit":
                for row in value["rows"]:
                    if "raw_matches" in row:row["archived_sg_raw_matches"]=row["raw_matches"]
                    row["raw_matches"]=dict(path=row["methods"]["analytic"]["configuration"]["matches"],
                        source="unchanged archived independently normalized SG+MA table")
            path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return outer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-predictions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.fusion_predictions,args.output)
    if any(v["failed"] for v in result["arms"].values()):raise SystemExit(1)


if __name__=="__main__":main()
