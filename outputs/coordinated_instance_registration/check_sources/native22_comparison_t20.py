"""One read-only comparison of already scored existing22 outputs; no raw labels."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"outputs/coordinated_instance_registration"
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def metric(value):
    if "mean_canvas_px" in value:
        return dict(mean=value["mean_canvas_px"],p90=value["p90_canvas_px"],maximum=value["max_canvas_px"],
                    ids=sorted(value["per_landmark_canvas_px"]))
    return dict(mean=value["mean"],p90=value["p90"],maximum=value.get("maximum",value.get("max")),
                ids=sorted(value.get("per_label",value.get("per_landmark"))))
def item(value,source,configuration_source=None,configuration=None):
    return dict(**metric(value),source_score=relative(source),configuration_source=configuration_source,configuration=configuration)

lung_score_path=BASE/"lung_all20_f2reserve_t134/all_id_scores.json"
lung_prediction_path=lung_score_path.parent/"predictions.json"
lung=read(lung_score_path); lung_pred=read(lung_prediction_path)
old_lung=read(BASE/"lung_all20_fixedrecipe_t123/all_id_scores.json")
old_lung_pred=read(BASE/"lung_all20_fixedrecipe_t123/predictions.json")
shared_path=BASE/"existing22_shared_affine_a300_t20/landmark_scores.json"
shared=read(shared_path); shared_pred=read(shared_path.parent/"predictions.json")
native_path=BASE/"native22_dhr_standard_t20/landmark_scores.json"
native=read(native_path); native_pred=read(native_path.parent/"predictions.json")
initial_path=native_path.parent/"initial_only_scores.json"
initial=read(initial_path)
assert all(p["prediction_complete"] for p in (lung_pred,shared_pred,native_pred))
assert all(p["scored_pairs"]==22 for p in (shared,native,initial))
rows=[]
for index,(a,d,i) in enumerate(zip(shared["rows"],native["rows"],initial["rows"],strict=True)):
    assert a["name"]==d["name"]==i["name"]
    name=a["name"]
    methods={}
    if index<20:
        old=lung["rows"][index]; pred=lung_pred["rows"][index]
        assert old["name"]==pred["name"]==name
        for key,method in (("original_A","analytic"),("F2_declared","f2"),("reduced_DHR","dhr"),("common_initial","common_affine")):
            if method in ("analytic","f2"):
                source=(lung_prediction_path.parent/pred["methods"][method]["report"]).resolve()
                cfg_source=relative(source);cfg=read(source)["configuration"]
            elif method=="dhr":
                source=(lung_prediction_path.parent/pred["methods"][method]["configuration"]).resolve()
                cfg_source=relative(source);cfg=read(source)
            else:cfg_source=str(pred["affine"]);cfg=None
            methods[key]=item(old["methods"][method]["metrics"]["canvas_pixels"],lung_score_path,cfg_source,cfg)
    else:
        score_path=BASE/f"development_p1arap3_stage30_t88/{name}_independent_score.json"
        old=read(score_path)
        for key,method in (("original_A","analytic"),("F2_declared","f2"),("common_initial","common_affine")):
            if method!="common_affine":
                source=score_path.parent/f"{name}_{method}_mind_edge257.json"
                cfg_source=relative(source);cfg=read(source)["configuration"]
            else:cfg_source=old["affine"];cfg=None
            methods[key]=item(old["results"][method],score_path,cfg_source,cfg)
        reduced_path=BASE/f"{name}_dhr_common_score.json"
        source=BASE/f"dhr_{name}_common/config.json"
        methods["reduced_DHR"]=item(read(reduced_path)["results"]["DHR_common"],reduced_path,relative(source),read(source))
    methods["shared_frame_A"]=item(a["metrics"]["canvas_pixels"],shared_path,
        relative(shared_path.parent/shared_pred["rows"][index]["report"]),shared_pred["rows"][index]["configuration"])
    config_path=(native_path.parent/native_pred["rows"][index]["configuration"]).resolve()
    methods["native_STANDARD"]=item(d["metrics"]["canvas_pixels"],native_path,relative(config_path),read(config_path))
    methods["native_initial"]=item(i["metrics"]["canvas_pixels"],initial_path,
        relative(native_path.parent/"predictions.json"),
        dict(theta=i["initial_transform"],frame=i["initial_transform_frame"],preprocessed_shape=i["preprocessed_shape"]))
    assert all(m["ids"]==methods["shared_frame_A"]["ids"] for m in methods.values())
    rows.append(dict(name=name,cohort="lung20" if index<20 else name,landmark_count=len(methods["shared_frame_A"]["ids"]),methods=methods,
        shared_frame_minus_original_A_mean=methods["shared_frame_A"]["mean"]-methods["original_A"]["mean"],
        native_nonrigid_mean_change=methods["native_STANDARD"]["mean"]-methods["native_initial"]["mean"],
        native_nonrigid_p90_change=methods["native_STANDARD"]["p90"]-methods["native_initial"]["p90"],
        common_initial_to_shared_A_mean_change=methods["shared_frame_A"]["mean"]-methods["common_initial"]["mean"],
        native_local_topology=d["topology"],shared_A_certificate=a["map_certificate"]))
cohorts={}
for cohort in ("lung20","histo","rat_kidney"):
    selected=[r for r in rows if r["cohort"]==cohort]
    cohorts[cohort]=dict(directions=len(selected),methods={method:dict(
        mean_pair_mean=float(np.mean([r["methods"][method]["mean"] for r in selected])),
        mean_pair_p90=float(np.mean([r["methods"][method]["p90"] for r in selected]))) for method in rows[0]["methods"]},
        shared_A_better_than_original_A_mean=sum(r["shared_frame_minus_original_A_mean"]<0 for r in selected),
        native_nonrigid_worse_than_native_initial_mean=sum(r["native_nonrigid_mean_change"]>0 for r in selected),
        native_nonrigid_worse_than_native_initial_p90=sum(r["native_nonrigid_p90_change"]>0 for r in selected),
        shared_A_better_than_native_final_mean=sum(r["methods"]["shared_frame_A"]["mean"]<r["methods"]["native_STANDARD"]["mean"] for r in selected))
equal={method:{metric:float(np.mean([c["methods"][method][metric] for c in cohorts.values()]))
               for metric in ("mean_pair_mean","mean_pair_p90")} for method in rows[0]["methods"]}
report=dict(scope="same existing20lung directions plus Histo77/kidney69; three previously viewed specimens, not22independentpatients",
    units="same512 moving-canvas pixels; p90 column is mean of within-direction p90s, not pooled p90",
    protocols="originalA/F2 retainoriginaldescriptorframe; sharedA changesONLYmind_frame/output; reducedDHR uses512/commonaffine/reducedschedule; nativeSTANDARD usesnativeimages/owninitializer/fullpreset",
    timing_caution="single calls, noABBA; native includesitsinitializer whileA reusesfrozenimageaffine/matches, not matched-objective kernel speed",
    lung_F2_version=dict(primary_source=relative(lung_score_path),floor_safety_fraction=.95,accepted_gain=1.,
        original_incomplete_source="outputs/coordinated_instance_registration/lung_all20_fixedrecipe_t123/all_id_scores.json",
        original_incomplete_mean=old_lung["aggregates"]["f2"]["all20_equal_direction_metrics"]["canvas_pixels"]["mean_of_direction_means"],
        original_incomplete_cases=[dict(name=r["name"],gradient_steps=r["methods"]["f2"]["gradient_steps"],failed_trials=r["methods"]["f2"]["failed_trials"])
                                  for r in old_lung_pred["rows"] if not r["methods"]["f2"]["budget_complete"]]),
    initialization_caution="observed sequential decomposition; not a rerun from exchanged initializers or isolation of one nonrigid factor",
    cohorts=cohorts,equal_specimen_canvas=equal,rows=rows)
output=BASE/"native22_comparison_t20.json"
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
print(json.dumps(dict(cohorts=cohorts,equal_specimen_canvas=equal),indent=2))
