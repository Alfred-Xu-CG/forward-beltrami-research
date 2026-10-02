"""Post-all22 scoring of corrected-frame A, reusing existing exact dataset readers.

Uses saved certified P1 maps with independent NumPy barycentric evaluation;
native DHR fields are not accepted here. No optimizer/model selection is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.coordinated_dhr_existing_inputs import existing_rows
from tools.coordinated_dhr_existing_score import _case_data, _aggregate
from tools.coordinated_dhr_released_score import summary
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit, canvas_unit_to_original_pixel
from tools.digital_q1_real_eval import load_effective_vertices


def score(predictions,data_root,expected_gradient_steps=300,*,budget_mode='fixed_gradients'):
    if isinstance(expected_gradient_steps,bool) or not isinstance(expected_gradient_steps,int) or expected_gradient_steps<=0:
        raise ValueError("positive integer expected_gradient_steps required")
    path=Path(predictions).resolve()
    if path.is_dir():path=path/"predictions.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    if budget_mode not in ('fixed_gradients','timed_final','distortion_cap'):
        raise ValueError('declare fixed_gradients, timed_final or distortion_cap budget mode')
    if budget_mode=='timed_final' and (manifest.get('budget_mode')!='timed_final'
            or manifest.get('all50_terminal') is not True or manifest.get('seconds_per_axis')!=2.):
        raise ValueError('explicit completed paired timed-final protocol required')
    if budget_mode=='distortion_cap' and (manifest.get('budget_mode')!='distortion_cap'
            or manifest.get('all25_terminal') is not True or manifest.get('maximum_gradient_steps')!=300):
        raise ValueError('explicit completed all25 capped-distortion schedule required')
    native=existing_rows(Path(data_root).resolve())
    records=manifest.get("rows",[])
    if (manifest.get("prediction_complete") is not True or manifest.get("annotations_read") is not False
            or manifest.get("cohort_size")!=22 or [r.get("name") for r in records]!=[r["name"] for r in native]
            or any(r.get("status") not in ("ok","failed") for r in records)):
        raise ValueError("all22 ordered image-only predictions must be terminal before labels")
    rows=[]
    for record,case in zip(records,native,strict=True):
        row={key:record.get(key) for key in ("name","status","complete_call_seconds","peak_allocated_bytes",
                    "source_report","actual_minimum_corner_ratio","gradient_steps","failed_trials")}
        rows.append(row)
        if record["status"]=="failed":
            row["error"]=record.get("error");continue
        try:
            if record["configuration"].get("mind_frame")!="shared_affine":
                raise ValueError('declared shared-affine evidence required')
            if budget_mode=='timed_final':
                from tools.coordinated_data_metric_batch import validate_timed_budget
                report=json.loads((path.parent/record['report']).read_text(encoding='utf-8'))
                validate_timed_budget(report,arm=manifest.get('timed_arm'))
                if any(record.get(k)!=report.get(k) for k in ('gradient_steps','prefix_gradient_steps',
                        'suffix_gradient_steps','terminal_budget','failed_trials')):
                    raise ValueError('timed prediction/report accounting mismatch')
            elif budget_mode=='distortion_cap':
                from tools.coordinated_distortion_budget_batch import validate_budget
                report=json.loads((path.parent/record['report']).read_text(encoding='utf-8'))
                validate_budget(report)
                if any(record.get(k)!=report.get(k) for k in ('gradient_steps','counts','distortion_budget',
                        'maximum_gradient_steps','numerical_failure','schedule_complete','initial_map')):
                    raise ValueError('capped-distortion prediction/report accounting mismatch')
            elif record.get("gradient_steps")!=expected_gradient_steps or record.get("failed_trials")!=0:
                raise ValueError(f"declared corrected-frame analytic{expected_gradient_steps} output required")
            archive=(path.parent/record["output"]).resolve()
            vertices,certificate=load_effective_vertices(archive)
            with np.load(archive,allow_pickle=False) as saved:
                if str(saved["interpolation"])!="p1_ac":raise ValueError("declared P1ac output required")
            if not certificate["composite_representation_valid"]:raise ValueError("invalid saved P1/affine map")
            native_record=dict(name=record["name"])
            for role in ("fixed","moving"):
                with Image.open(case[role]) as image:native_record[role+"_original_wh"]=list(image.size)
            wh,layouts,points,ids,fixed_only,moving_only=_case_data(native_record,data_root)
            query=original_pixel_to_canvas_unit(points["fixed"],layouts["fixed"],512)
            if not np.isfinite(query).all() or np.any(query<0) or np.any(query>1):
                raise ValueError("all fixed label queries must be in the declared source canvas")
            predicted=p1_at_queries_numpy(vertices,query,"ac")
            expected=original_pixel_to_canvas_unit(points["moving"],layouts["moving"],512)
            native_predicted=canvas_unit_to_original_pixel(predicted,layouts["moving"],512)
            row.update(status="ok",scored_landmarks=len(ids),available_pair_labels=ids,
                       fixed_only_ids=fixed_only,moving_only_ids=moving_only,map_certificate=certificate,
                       metrics=dict(canvas_pixels=summary(512*np.linalg.norm(predicted-expected,axis=1),ids),
                                    native_moving_pixels=summary(np.linalg.norm(native_predicted-points["moving"],axis=1),ids)))
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
    lung,histo,kidney=_aggregate(rows[:20]),_aggregate(rows[20:21]),_aggregate(rows[21:])
    equal=None
    if all(group["all_directions"] is not None for group in (lung,histo,kidney)):
        equal={key:float(np.mean([group["all_directions"]["canvas_pixels"][key] for group in (lung,histo,kidney)]))
               for key in ("mean_pair_mean","mean_pair_p90")}
    return dict(prediction_manifest=str(path),scope="same22 directions from three previously viewed specimens; corrected-frame A control",
                pair_denominator=22,scored_pairs=sum(r["status"]=="ok" for r in rows),rows=rows,
                expected_gradient_steps=expected_gradient_steps if budget_mode=='fixed_gradients' else None,
                budget_mode=budget_mode,
                lung_all20=lung,histo=histo,rat_kidney=kidney,equal_specimen_canvas=equal,
                units="original512 moving-canvas pixels; native pixels separate, no cross-specimen native averaging",
                label_policy="same80/77/69 readers, kidneyfixedonly70/71; no prediction-dependent label filtering",
                evaluation="independent NumPy P1ac barycentric evaluation of stored affine-composed map; after all22 terminal")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("predictions","data-root","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--expected-gradient-steps",type=int,default=300)
    parser.add_argument('--budget-mode',choices=('fixed_gradients','timed_final','distortion_cap'),default='fixed_gradients')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=score(args.predictions,args.data_root,args.expected_gradient_steps,budget_mode=args.budget_mode)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(scored_pairs=result["scored_pairs"],equal_specimen_canvas=result["equal_specimen_canvas"])))


if __name__=="__main__":main()
