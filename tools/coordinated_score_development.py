"""Read-only batch scoring of the THREE declared, reused development specimens.

No optimizer or machine matcher is invoked. This is not an official leaderboard
or independent-patient test. All shared manual IDs are retained by score_case.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tools.coordinated_score_case import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--maps-dir",type=Path,required=True)
    p.add_argument("--methods",nargs="+",default=["radial","analytic","f1","f2"])
    p.add_argument("--loss",default="mind")
    p.add_argument("--lr-calibration",choices=("edge","physical"),default="edge")
    args=p.parse_args();base=args.data_root;canvas=base/"birl_anhir_dev/canvas"
    labels={
        "histo":(base/"HistoReg_CD68_CD4/Landmarks_CD4.csv",base/"HistoReg_CD68_CD4/Landmarks_CD68.csv"),
        "lesions":(base/"birl_anhir_dev/labels_eval_only/lesions_/scale-5pc/Izd2-29-041-w35_HE.csv",
                   base/"birl_anhir_dev/labels_eval_only/lesions_/scale-5pc/Izd2-29-041-w35_proSPC.csv"),
        "rat_kidney":(base/"birl_anhir_dev/labels_eval_only/rat-kidney_/scale-5pc/Rat-Kidney_HE.csv",
                      base/"birl_anhir_dev/labels_eval_only/rat-kidney_/scale-5pc/Rat-Kidney_PanCytokeratin.csv")}
    for case,(fixed,moving) in labels.items():
        maps={m:args.maps_dir/(case+"_"+m+"_"+args.loss+"_"+args.lr_calibration+"257.npz") for m in args.methods}
        report=score(canvas/(case+"_layout.json"),fixed,moving,maps,canvas/(case+"_initial_affine.npz"),
                     args.maps_dir/(case+"_independent_score.json"))
        rows=[]
        for method,values in report["results"].items():
            row=dict(method=method,mean_canvas_px=values["mean_canvas_px"],p90_canvas_px=values["p90_canvas_px"])
            if method in maps:
                run=json.loads(maps[method].with_suffix(".json").read_text(encoding="utf-8"))
                row.update(gradient_steps=run["gradient_steps"],failed_trials=run["failed_trials"],
                    seconds=run["optimize_seconds"],peak_bytes=run["peak_allocated_bytes"],
                    selected_stage=run.get("selected_stage"),final_total=run["final"]["total"])
            rows.append(row)
        print(json.dumps(dict(case=case,required_shared_landmarks=report["landmark_count"],results=rows)))


if __name__=="__main__":
    main()
