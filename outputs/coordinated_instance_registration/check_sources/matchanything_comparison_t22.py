"""Read terminal MatchAnything predictions and saved scores; never run a model."""
import argparse
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_terminal(prediction):
    rows=prediction.get("rows",[])
    if (prediction.get("prediction_complete") is not True or prediction.get("annotations_read") is not False
            or len(rows)!=25 or len({r.get("name") for r in rows})!=25
            or any(r.get("status") not in ("ok","failed") for r in rows)):
        raise ValueError("all25 image-only predictions must be terminal before reading scores")


def summary(values,denominator):
    known=[float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in known):raise ValueError("finite cost diagnostics required")
    return dict(recorded_count=len(known),denominator=denominator,missing_count=denominator-len(known),
        minimum=min(known) if known else None,maximum=max(known) if known else None,
        mean=statistics.mean(known) if known else None,total=sum(known) if known else None)


def aggregate(records):
    ok=[r for r in records if r.get("status")=="ok"]
    values=None
    if len(ok)==len(records):
        metrics=[r["metrics"]["canvas_pixels"] for r in ok]
        values=dict(mean_pair_mean=statistics.mean(r["mean"] for r in metrics),
            mean_pair_p90=statistics.mean(r["p90"] for r in metrics),
            worst_pair_maximum=max(r["maximum"] for r in metrics))
    return dict(pair_denominator=len(records),scored_pairs=len(ok),failure_count=len(records)-len(ok),
        all_directions_canvas_pixels=values)


def build(prediction,miit,existing,gray_miit,gray_existing,gray_miit_prediction,gray_existing_prediction):
    require_terminal(prediction)
    predictions={r["name"]:r for r in prediction["rows"]}
    rows=[];cohorts={}
    old_cost_rows={r["name"]:r["methods"]["analytic"] for r in gray_miit_prediction["rows"]}
    old_cost_rows.update({r["name"]:r for r in gray_existing_prediction["rows"]})
    groups=(("miit",gray_miit["rows"],miit["rows"],True),
        ("lung_all20",gray_existing["rows"][:20],existing["rows"][:20],False),
        ("histo",gray_existing["rows"][20:21],existing["rows"][20:21],False),
        ("rat_kidney",gray_existing["rows"][21:],existing["rows"][21:],False))
    expected_sizes={"miit":3,"lung_all20":20,"histo":1,"rat_kidney":1}
    for cohort,left_rows,right_rows,nested in groups:
        if (len(left_rows)!=expected_sizes[cohort] or [r["name"] for r in left_rows]!=[r["name"] for r in right_rows]):
            raise ValueError("same ordered complete cohort scores required; failures cannot disappear")
        left=[r["methods"]["analytic"] if nested else r for r in left_rows]
        right=[r["methods"]["analytic"] if nested else r for r in right_rows]
        a,b=aggregate(left),aggregate(right)
        av,bv=a["all_directions_canvas_pixels"],b["all_directions_canvas_pixels"]
        cohorts[cohort]=dict(shared_gray_sg=a,matchanything=b,
            delta=None if av is None or bv is None else {k:bv[k]-av[k] for k in av})
        for old,new,l,r in zip(left_rows,right_rows,left,right):
            p=predictions[new["name"]]
            if p["status"]!="ok" and r["status"]=="ok":raise ValueError("failed prediction cannot have successful score")
            item=dict(name=new["name"],cohort=cohort,prediction_status=p["status"],
                shared_gray_sg_status=l["status"],matchanything_status=r["status"],metrics=None,
                failure=r.get("error",p.get("error")),
                costs=dict(extraction_complete_call_seconds=p.get("extraction_complete_call_seconds"),
                    extraction_peak_allocated_bytes=p.get("extraction_peak_allocated_bytes"),
                    optimizer_complete_call_seconds=p.get("optimization_complete_call_seconds"),
                    optimizer_peak_allocated_bytes=p.get("peak_allocated_bytes"),
                    combined_pair_call_seconds=p.get("complete_call_seconds"),
                    optimizer_serialization_seconds=p.get("serialization_seconds"),
                    optimizer_certification_seconds=p.get("certification_seconds"),
                    shared_gray_sg_optimizer_complete_call_seconds=old_cost_rows[new["name"]].get("complete_call_seconds"),
                    shared_gray_sg_optimizer_peak_allocated_bytes=old_cost_rows[new["name"]].get("peak_allocated_bytes")))
            if l["status"]==r["status"]=="ok":
                item["metrics"]={k:dict(shared_gray_sg=l["metrics"]["canvas_pixels"][k],
                    matchanything=r["metrics"]["canvas_pixels"][k],
                    delta=r["metrics"]["canvas_pixels"][k]-l["metrics"]["canvas_pixels"][k])
                    for k in ("mean","p90","maximum")}
            rows.append(item)
    if {r["name"] for r in rows}!=set(predictions):raise ValueError("same25 prediction/score identities required")
    costs={key:summary([r["costs"][key] for r in rows],25) for key in rows[0]["costs"]}
    for key,value in costs.items():
        if "peak_allocated_bytes" in key:value["total"]=None  # Phase peaks are not additive.
    all_aggregates=[v["delta"] for v in cohorts.values()]
    equal_specimen_delta=None if any(v is None for v in all_aggregates) else {
        k:statistics.mean(v[k] for v in all_aggregates) for k in ("mean_pair_mean","mean_pair_p90")}
    return dict(scope="four previously viewed specimens;25 correlated directions,not25patients; development evidence",
        units="512 moving-canvas pixels; no pooled native-pixel or patient-level inference",pair_denominator=25,
        prediction_failures=sum(r["status"]!="ok" for r in prediction["rows"]),
        scoring_failures=sum(r["matchanything_status"]!="ok" for r in rows),
        cohorts=cohorts,rows=rows,equal_specimen_delta=equal_specimen_delta,costs=costs,
        one_time_cold_setup=dict(count=prediction.get("cost_totals",{}).get("model_setup_count"),
            seconds=prediction.get("model_setup_seconds"),status=prediction.get("model_setup_status"),
            peak_allocated_bytes=prediction.get("model_setup_peak_allocated_bytes"),
            release_seconds=prediction.get("model_release_seconds")),
        batch=dict(wall_seconds=prediction.get("elapsed_seconds"),
            extraction_wall_seconds=prediction.get("extraction_wall_seconds"),
            optimization_wall_seconds=prediction.get("optimization_wall_seconds"),
            extraction_calls=prediction.get("cost_totals",{}).get("extraction_calls"),
            optimizer_calls=prediction.get("cost_totals",{}).get("optimizer_calls"),
            model_released_before_optimization=prediction.get("model_released_before_optimization"),
            gray_control_separate_batch_wall_seconds=[gray_miit_prediction.get("elapsed_seconds"),gray_existing_prediction.get("elapsed_seconds")]),
        timing_scope="One-time cold model setup is separate from per-pair extraction+optimizer calls. Old frozen affine initialization and SG extraction are excluded from BOTH reused control and new registration timing; new MatchAnything extraction is included. Old controls were separate batches, not warmed timing repeats.",
        memory_scope="Extraction peaks include the warm resident model. Model is freed before optimizer calls. GPU allocated-memory peaks are phase maxima, never additive; unknown measurements stay null.",
        warning="Point evidence changes E; total energies are not comparable as same-functional convergence. Failure leaves cohort/equal-specimen aggregate undefined; failed-attempt costs remain included.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    directory=args.directory.resolve()
    prediction=read(directory/"predictions.json")
    require_terminal(prediction)  # Must precede opening ANY score file.
    miit=read(directory/"miit_scores.json");existing=read(directory/"existing_scores.json")
    gm=ROOT/"miit_multiscale_control_t19";ge=ROOT/"existing22_shared_affine_a300_t20"
    result=build(prediction,miit,existing,read(gm/"landmark_scores.json"),read(ge/"landmark_scores.json"),
        read(gm/"predictions.json"),read(ge/"predictions.json"))
    result["prediction_manifest"]=str(directory/"predictions.json")
    destination=args.output or directory/"comparison.json"
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(cohorts=result["cohorts"],cold_setup=result["one_time_cold_setup"],batch=result["batch"]),indent=2))


if __name__=="__main__":main()
