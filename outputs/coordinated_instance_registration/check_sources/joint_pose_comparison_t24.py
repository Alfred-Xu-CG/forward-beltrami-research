"""Compare completed frozen250/joint300 with archived frozen fusion300."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("fusion_comparison",Path(__file__).with_name("fusion_comparison_t23.py"))
HELPER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(HELPER)
read,flatten,aggregate,summary=HELPER.read,HELPER.flatten,HELPER.aggregate,HELPER.summary
METHODS=("frozen300","frozen250","joint300")
CONTRASTS=(("frozen250_minus_frozen300","frozen250","frozen300"),
    ("joint300_minus_frozen300","joint300","frozen300"),("joint300_minus_frozen250","joint300","frozen250"))


def require_complete(outer,manifests):
    if (outer.get("prediction_complete") is not True or outer.get("attempt_denominator")!=50
            or outer.get("annotations_read") is not False):raise ValueError("all50 fresh attempts must terminate before scores")
    for method,manifest in manifests.items():
        if (manifest.get("prediction_complete") is not True or manifest.get("all50_terminal") is not True
                or manifest.get("annotations_read") is not False or len(manifest.get("rows",[]))!=25
                or any(r.get("status") not in ("ok","failed") for r in manifest["rows"])):
            raise ValueError("complete25 direction manifests required, including failures")
    if any([r["name"] for r in m["rows"]]!=[r["name"] for r in manifests["frozen300"]["rows"]] for m in manifests.values()):
        raise ValueError("same ordered25 directions required")


def interpretation(record,report,method):
    export=report.get("joint_export_validation",record.get("export_validation",{}))
    support=report.get("final_nominal_descriptor_support")
    if support is None:support=(report.get("mind_frame_by_resolution") or {}).get("512")
    return dict(pose_mode=record.get("pose_mode",report.get("pose_mode","frozen_affine")),
        selected_stage=report.get("selected_stage"),selected_stage_scope=report.get("selected_stage_scope"),
        pose_parameters=report.get("pose_parameters"),pose_parameter_order=report.get("pose_parameter_order"),
        pose_matrix=report.get("pose_matrix"),pose_determinant=report.get("pose_determinant"),
        minimum_original_affine_normalized_corner_ratio=export.get("original_affine_normalized_minimum_corner_ratio",
            record.get("actual_minimum_corner_ratio")),
        minimum_residual_corner_ratio=export.get("residual_minimum_corner_ratio"),
        initial_outside_fraction=(report.get("initial") or {}).get("outside_fraction"),
        final_outside_fraction=(report.get("final") or {}).get("outside_fraction"),
        nominal_descriptor_support=support,final_texture=report.get("final_texture"),
        feature_rebuilds_by_resolution=report.get("feature_rebuilds_by_resolution"),
        gradient_steps=report.get("gradient_steps",record.get("gradient_steps")),
        residual_gradient_steps=report.get("residual_gradient_steps",report.get("gradient_steps") if method!="joint300" else None),
        pose_gradient_steps=report.get("pose_gradient_steps",0 if method!="joint300" else None))


def build(outer,manifests,scores,reports):
    require_complete(outer,manifests)
    ordered={method:flatten(*scores[method]) for method in METHODS}
    identities=[(name,cohort) for name,cohort,_ in ordered["frozen300"]]
    expected={"miit":3,"lung_all20":20,"histo":1,"rat_kidney":1}
    if len(identities)!=25 or len(set(identities))!=25 or {c:sum(cohort==c for _,cohort in identities) for c in expected}!=expected:
        raise ValueError("fixed four-specimen25-direction score cohort required")
    if any([(n,c) for n,c,_ in ordered[m]]!=identities for m in METHODS):raise ValueError("identical scored denominators required")
    if [n for n,_ in identities]!=[r["name"] for r in manifests["frozen300"]["rows"]]:raise ValueError("prediction/score identities differ")
    cohorts={}
    for cohort in expected:
        groups={m:aggregate([r for _,c,r in ordered[m] if c==cohort]) for m in METHODS}
        deltas={}
        for label,left,right in CONTRASTS:
            a,b=groups[left]["all_directions_canvas_pixels"],groups[right]["all_directions_canvas_pixels"]
            deltas[label]=None if a is None or b is None else {k:a[k]-b[k] for k in a}
        cohorts[cohort]=dict(methods=groups,deltas=deltas)
    rows=[]
    for i,(name,cohort) in enumerate(identities):
        entries={m:ordered[m][i][2] for m in METHODS}
        details={m:interpretation(manifests[m]["rows"][i],reports[m].get(name,{}),m) for m in METHODS}
        for m in METHODS:
            if manifests[m]["rows"][i]["status"]!="ok" and entries[m]["status"]=="ok":raise ValueError("failed prediction cannot score successfully")
        metrics={}
        for metric in ("mean","p90","maximum"):
            values={m:e["metrics"]["canvas_pixels"][metric] if e["status"]=="ok" else None for m,e in entries.items()}
            metrics[metric]=dict(values=values,deltas={label:None if values[a] is None or values[b] is None else values[a]-values[b]
                for label,a,b in CONTRASTS})
        support_delta={}
        for label,a,b in CONTRASTS:
            sa=(details[a]["nominal_descriptor_support"] or {}).get("shared_affine_descriptor_support",{})
            sb=(details[b]["nominal_descriptor_support"] or {}).get("shared_affine_descriptor_support",{})
            support_delta[label]={k:sa[k]-sb[k] if k in sa and k in sb else None
                for k in ("full_domain_fraction","fixed_mask_weighted_fraction")}
        rows.append(dict(name=name,cohort=cohort,status={m:e["status"] for m,e in entries.items()},metrics=metrics,
            diagnostics=details,nominal_support_deltas=support_delta,
            failures={m:e.get("error",manifests[m]["rows"][i].get("error")) for m,e in entries.items() if e["status"]!="ok"}))
    costs={}
    for method in METHODS:
        records=manifests[method]["rows"]
        per_pair=[]
        for record in records:
            report=reports[method].get(record["name"],{})
            per_pair.append(dict(name=record["name"],status=record["status"],complete_call_seconds=record.get("complete_call_seconds"),
                **{k:report.get(k,record.get(k)) for k in ("optimize_seconds","peak_allocated_bytes","serialization_seconds",
                    "certification_seconds","feature_seconds","final_diagnostics_seconds","gradient_steps",
                    "pose_gradient_steps","residual_gradient_steps")}))
            if method!="joint300":
                per_pair[-1].update(pose_gradient_steps=0,residual_gradient_steps=per_pair[-1]["gradient_steps"])
        measures={k:summary([p[k] for p in per_pair],25) for k in ("complete_call_seconds","optimize_seconds","peak_allocated_bytes")}
        measures["peak_allocated_bytes"]["total"]=None
        costs[method]=dict(per_pair=per_pair,measures=measures,arm_wall_seconds=manifests[method].get("elapsed_seconds"),
            planned_gradients=dict(total=250 if method=="frozen250" else 300,
                residual=300 if method=="frozen300" else 250,pose=50 if method=="joint300" else 0))
    equal={label:None if any(c["deltas"][label] is None for c in cohorts.values()) else {
        k:statistics.mean(c["deltas"][label][k] for c in cohorts.values()) for k in ("mean_pair_mean","mean_pair_p90")}
        for label,_,_ in CONTRASTS}
    return dict(scope="four previously viewed specimens;25 correlated directions;no per-case best-method selection",
        units="512 moving-canvas pixels",cohorts=cohorts,rows=rows,equal_specimen_deltas=equal,costs=costs,
        fresh_batch_wall_seconds=outer.get("elapsed_seconds"),
        support_scope="Nominal static descriptor footprint at identity residual for original/selected pose; not actual deformed-query support or anatomical correctness. Diagnostics do not select maps or filter labels.",
        cost_scope="Fresh optimizer timing; frozen initialization and SG/MA extraction remain excluded earlier costs. Pose and residual gradients differ in cost;300 is not equal compute. Peaks are nonadditive; missing values remain null.",
        inference_scope="Learned global pose changes output polygon and prior. Failure leaves full-cohort and affected equal-specimen aggregates undefined. No patient-level generalization or same-functional convergence claim.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    locations=dict(frozen300=ROOT/"match_fusion_all50_t23"/"fusion",frozen250=directory/"frozen250",joint300=directory/"joint300")
    outer=read(directory/"predictions.json")
    manifests={m:read(path/"predictions.json") for m,path in locations.items()}
    require_complete(outer,manifests)  # No score/report reads before the full batch terminates.
    scores={m:(read(path/"miit_scores.json"),read(path/"existing_scores.json")) for m,path in locations.items()}
    reports={}
    for method,path in locations.items():
        reports[method]={}
        for row in manifests[method]["rows"]:
            report_path=Path(row["report"])
            if not report_path.is_absolute():report_path=path/report_path
            if report_path.exists():reports[method][row["name"]]=read(report_path)
    result=build(outer,manifests,scores,reports)
    result["prediction_manifest"]=str(directory/"predictions.json")
    destination=args.output or directory/"comparison.json"
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(cohorts=result["cohorts"],costs={m:v["measures"] for m,v in result["costs"].items()}),indent=2))


if __name__=="__main__":main()
