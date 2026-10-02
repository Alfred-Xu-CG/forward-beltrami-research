"""Compare completed frozen-evidence arms; no optimization or label loading."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("match_comparison",Path(__file__).with_name("matchanything_comparison_t22.py"))
HELPER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
read=HELPER.read
aggregate=HELPER.aggregate
summary=HELPER.summary


def require_complete(outer,arms):
    if (outer.get("prediction_complete") is not True or outer.get("composition_complete") is not True
            or outer.get("annotations_read") is not False or outer.get("attempt_denominator")!=50
            or len(outer.get("composition_rows",[]))!=25
            or any(r.get("status") not in ("ok","failed") for r in outer["composition_rows"])):
        raise ValueError("all50 attempts and25 compositions must be terminal before scores")
    for arm in ("sg2","fusion"):
        HELPER.require_terminal(arms[arm])
        if arms[arm].get("all50_terminal") is not True:raise ValueError("outer completion required")
    if [r["name"] for r in arms["sg2"]["rows"]]!=[r["name"] for r in arms["fusion"]["rows"]]:
        raise ValueError("same ordered25 directions required")


def flatten(miit,existing):
    return [(r["name"],"miit",r["methods"]["analytic"]) for r in miit["rows"]]+[
        (r["name"],"lung_all20" if i<20 else "histo" if i==20 else "rat_kidney",r)
        for i,r in enumerate(existing["rows"])]


def build(outer,arms,scores,baseline_scores,baseline_predictions):
    require_complete(outer,arms)
    ordered={"sg1":flatten(*baseline_scores),**{a:flatten(*scores[a]) for a in ("sg2","fusion")}}
    identities=[(n,c) for n,c,_ in ordered["sg1"]]
    if len(identities)!=25 or len(set(identities))!=25:raise ValueError("25 unique baseline directions required")
    if any([(n,c) for n,c,_ in ordered[a]]!=identities for a in ("sg2","fusion")):
        raise ValueError("exact same ordered score denominators required")
    expected_counts={"miit":3,"lung_all20":20,"histo":1,"rat_kidney":1}
    if {c:sum(cohort==c for _,cohort in identities) for c in expected_counts}!=expected_counts:
        raise ValueError("four fixed specimen cohorts required")
    for arm in arms:
        if [r["name"] for r in arms[arm]["rows"]]!=[n for n,_ in identities]:raise ValueError("score/prediction identity mismatch")
        for prediction,(_,_,score) in zip(arms[arm]["rows"],ordered[arm]):
            if prediction["status"]!="ok" and score["status"]=="ok":raise ValueError("failed prediction cannot score successfully")
    cohorts={}
    contrasts=(("fusion_minus_sg1","fusion","sg1"),("sg2_minus_sg1","sg2","sg1"),("fusion_minus_sg2","fusion","sg2"))
    for cohort in expected_counts:
        groups={a:aggregate([r for _,c,r in triples if c==cohort]) for a,triples in ordered.items()}
        deltas={}
        for label,a,b in contrasts:
            left,right=groups[a]["all_directions_canvas_pixels"],groups[b]["all_directions_canvas_pixels"]
            deltas[label]=None if left is None or right is None else {k:left[k]-right[k] for k in left}
        cohorts[cohort]=dict(methods=groups,deltas=deltas)
    rows=[]
    for i,(name,cohort) in enumerate(identities):
        methods={a:triples[i][2] for a,triples in ordered.items()}
        metrics={}
        for metric in ("mean","p90","maximum"):
            values={a:r["metrics"]["canvas_pixels"][metric] if r["status"]=="ok" else None for a,r in methods.items()}
            metrics[metric]=dict(values=values,deltas={label:None if values[a] is None or values[b] is None else values[a]-values[b]
                for label,a,b in contrasts})
        rows.append(dict(name=name,cohort=cohort,status={a:r["status"] for a,r in methods.items()},metrics=metrics,
            failures={a:r.get("error") for a,r in methods.items() if r["status"]!="ok"}))
    control_costs=[r["methods"]["analytic"] for r in baseline_predictions[0]["rows"]]+baseline_predictions[1]["rows"]
    costs={}
    for arm,records in (("sg1",control_costs),("sg2",arms["sg2"]["rows"]),("fusion",arms["fusion"]["rows"])):
        calls=summary([r.get("complete_call_seconds") for r in records],25)
        memory=summary([r.get("peak_allocated_bytes") for r in records],25);memory["total"]=None
        costs[arm]=dict(optimizer_complete_calls=calls,optimizer_peak_allocated_bytes=memory,
            per_pair=[dict(name=identities[i][0],status=r["status"],complete_call_seconds=r.get("complete_call_seconds"),
                peak_allocated_bytes=r.get("peak_allocated_bytes"),serialization_seconds=r.get("serialization_seconds"),
                certification_seconds=r.get("certification_seconds")) for i,r in enumerate(records)],
            arm_wall_seconds=arms[arm].get("elapsed_seconds") if arm!="sg1" else None)
    equal={label:None if any(v["deltas"][label] is None for v in cohorts.values()) else {
        k:statistics.mean(v["deltas"][label][k] for v in cohorts.values()) for k in ("mean_pair_mean","mean_pair_p90")}
        for label,_,_ in contrasts}
    return dict(scope="four previously viewed specimens;25 correlated directions per arm, not25patients",
        units="512 moving-canvas pixels",methods=dict(sg1="archived .1 P_SG",sg2=".2 P_SG",fusion=".1 P_SG + .1 P_MA"),
        cohorts=cohorts,rows=rows,equal_specimen_deltas=equal,costs=costs,
        current_batch_wall_seconds=outer.get("elapsed_seconds"),composition_seconds=outer.get("composition_seconds"),
        composition_failures=sum(r["status"]!="ok" for r in outer["composition_rows"]),
        historical_ma_costs=dict(predictions=outer.get("frozen_ma_predictions"),
            one_time_setup_seconds=outer.get("frozen_ma_historical_setup_seconds"),
            one_time_setup_count=(outer.get("frozen_ma_historical_cost_totals") or {}).get("model_setup_count"),
            extraction_calls=(outer.get("frozen_ma_historical_cost_totals") or {}).get("extraction_calls"),
            extraction_complete_call_seconds=(outer.get("frozen_ma_historical_cost_totals") or {}).get("extraction_complete_call_seconds")),
        cost_scope="Current calls exclude earlier frozen affine/SG extraction. Fusion uses cached MA tables: historical MA model setup and extraction remain separate required end-to-end costs, not zero. Historical MA optimizer time is not an extraction charge. GPU allocated peaks are not additive; null costs remain unmeasured.",
        interpretation="Fusion versus SG2 controls total point coefficient; this does not establish anatomical correctness or independent-patient generalization. Failed directions retain denominators and make corresponding full-cohort/equal-specimen aggregates undefined. Different point functionals prohibit same-E convergence claims.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    outer=read(directory/"predictions.json")
    arms={a:read(directory/a/"predictions.json") for a in ("sg2","fusion")}
    require_complete(outer,arms)  # Before opening any new scores.
    scores={a:(read(directory/a/"miit_scores.json"),read(directory/a/"existing_scores.json")) for a in arms}
    gm=ROOT/"miit_multiscale_control_t19";ge=ROOT/"existing22_shared_affine_a300_t20"
    result=build(outer,arms,scores,(read(gm/"landmark_scores.json"),read(ge/"landmark_scores.json")),
        (read(gm/"predictions.json"),read(ge/"predictions.json")))
    result["prediction_manifest"]=str(directory/"predictions.json")
    destination=args.output or directory/"comparison.json"
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(cohorts=result["cohorts"],composition_seconds=result["composition_seconds"]),indent=2))


if __name__=="__main__":main()
