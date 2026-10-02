"""Score all reverse MIIT outputs only after all nine attempts are terminal."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from tools.coordinated_miit_reverse import REVERSE, METHODS, read, save
from tools.coordinated_miit_score import read_points, validate_layout, metrics
from tools.coordinated_lung_all20_score import _affine, _safe_map
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.digital_compare_appearance import dhr_map_at_unit_queries
from tools.coordinated_dhr_released_score import field_corner_diagnostics

ALL_METHODS=("common_affine",)+METHODS


def load_completed(path):
    path=Path(path)
    if path.is_dir():path/= "predictions.json"
    value=read(path)
    if (value.get("prediction_complete") is not True or value.get("all9_terminal") is not True
            or value.get("annotations_read") is not False or value.get("attempt_denominator")!=9
            or value.get("cohort_size")!=3 or len(value.get("rows",[]))!=3):
        raise ValueError("all nine image-only attempts required BEFORE labels")
    for row,(moving,fixed) in zip(value["rows"],REVERSE,strict=True):
        if (row.get("name"),row.get("moving_section"),row.get("fixed_section"))!=(f"miit_{moving}_to_{fixed}",moving,fixed):
            raise ValueError("all three reverse directions in declared order required")
        if (set(row.get("methods",{}))!=set(METHODS)
                or any(row["methods"][m].get("status") not in ("ok","failed") for m in METHODS)
                or row.get("input_status") not in ("ok","failed")):
            raise ValueError("every method and input must be terminal BEFORE labels")
    return value,path.parent


def score(predictions,source_data):
    manifest,directory=load_completed(predictions)  # Must precede any CSV access.
    import SimpleITK as sitk
    rows=[];reference_ids=None
    for source in manifest["rows"]:
        row=dict(name=source["name"],moving_section=source["moving_section"],fixed_section=source["fixed_section"],methods={})
        rows.append(row)
        try:
            if source["input_status"]!="ok":raise ValueError(source.get("input_error","input preparation failed"))
            layout=validate_layout(read(directory/source["layout"]))
            a,b=_affine(directory/source["affine"])
            if (not np.array_equal(a,np.asarray(source["shared_affine_matrix"]))
                    or not np.array_equal(b,np.asarray(source["shared_affine_offset"]))):
                raise ValueError("manifest differs from actual shared reverse affine")
            points={};wh={}
            for role in ("fixed","moving"):
                section=source[role+"_section"]
                with Image.open(Path(source_data)/str(section)/"images/image.tif") as image:
                    wh[role]=np.asarray(image.size)
                    if image.getexif().get(274,1)!=1 or image.size!=tuple(layout[role]["original_wh"]):
                        raise ValueError("new moving/fixed native frames disagree with reversed layout")
                points[role]=read_points(Path(source_data)/str(section)/"landmarks"/f"{section:02d}.csv",
                                         wh[role],allow_missing_inf=True)
            nominal=sorted(points["fixed"])
            if set(points["moving"])!=set(nominal) or (reference_ids is not None and nominal!=reference_ids):
                raise ValueError("same124 nominal publisher IDs required")
            reference_ids=nominal
            absent={role:[label for label in nominal if np.isposinf(points[role][label]).all()] for role in points}
            missing=set(absent["fixed"])|set(absent["moving"])
            ids=[label for label in nominal if label not in missing]
            if not ids:raise ValueError("no available annotation pairs")
            fixed=np.stack([points["fixed"][label] for label in ids]);target=np.stack([points["moving"][label] for label in ids])
            q=original_pixel_to_canvas_unit(fixed,layout["fixed"],512)
            row.update(nominal_landmarks=124,scored_landmarks=len(ids),available_pair_labels=ids,
                unavailable_pair_labels=sorted(missing),unavailable_fixed_labels=absent["fixed"],unavailable_moving_labels=absent["moving"])
            row["methods"]["common_affine"]=dict(status="ok",metrics=metrics(q@a.astype(float).T+b,target,layout["moving"],ids))
            for method in METHODS:
                record=source["methods"][method]
                if record["status"]!="ok":
                    row["methods"][method]=copy.deepcopy(record);continue
                try:
                    if method!="native_shared":
                        vertices,metadata=_safe_map(record,directory,a,b)
                        if metadata["gradient_steps"]!=300 or metadata["failed_trials"]!=0:
                            raise ValueError("same completed300-gradient safe recipe required")
                        mapped=p1_at_queries_numpy(vertices,q,"ac")
                    else:
                        if (not np.array_equal(a,np.asarray(record["shared_affine_matrix"]))
                                or not np.array_equal(b,np.asarray(record["shared_affine_offset"]))):
                            raise ValueError("native shared inverse affine differs")
                        field=sitk.GetArrayFromImage(sitk.ReadImage(str(directory/record["field"])))
                        params=read(directory/record["postprocessing_params"])
                        query=torch.from_numpy(((fixed+.5)/wh["fixed"]).astype(np.float32)).reshape(1,1,-1,2)
                        native=dhr_map_at_unit_queries(field,params,fixed_size=tuple(wh["fixed"]),
                            moving_size=tuple(wh["moving"]),query=query)[0,0].double().numpy()*wh["moving"]-.5
                        mapped=original_pixel_to_canvas_unit(native,layout["moving"],512)
                        metadata=dict(topology=field_corner_diagnostics(field),initial_field_audit=record["initial_field_audit"])
                    row["methods"][method]=dict(status="ok",metrics=metrics(mapped,target,layout["moving"],ids),metadata=metadata)
                except Exception as error:row["methods"][method]=dict(status="failed",error=f"{type(error).__name__}: {error}")
        except Exception as error:
            for method in ALL_METHODS:row["methods"][method]=dict(status="failed",error=f"{type(error).__name__}: {error}")
    aggregate={}
    for method in ALL_METHODS:
        ok=[r for r in rows if r["methods"][method]["status"]=="ok"]
        value={unit:dict(mean_pair_mean=float(np.mean([r["methods"][method]["metrics"][unit]["mean"] for r in ok])),
                         mean_pair_p90=float(np.mean([r["methods"][method]["metrics"][unit]["p90"] for r in ok])),
                         maximum=float(max(r["methods"][method]["metrics"][unit]["maximum"] for r in ok)))
               for unit in ("canvas_pixels","native_moving_pixels")} if ok else None
        aggregate[method]=dict(pair_denominator=3,scored_pairs=len(ok),failed_pairs=3-len(ok),
                               all_three=value if len(ok)==3 else None,successful_only=value)
    return dict(scope=manifest["scope"],prediction_manifest=str(predictions),attempt_denominator=9,
        annotation_policy="same124 nominal IDs; only publisher (+inf,+inf) absence; all available IDs for every method; no prediction-based dropping",
        coordinate_convention="CSV native x,y treated as zero-based pixel centers; unverified publisher exact origin; new MOVING native pixels and512canvas units",
        rows=rows,aggregate=aggregate)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("predictions","source-data","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=score(args.predictions,args.source_data);save(args.output,result)
    print(json.dumps(result["aggregate"],allow_nan=False))


if __name__=="__main__":main()
