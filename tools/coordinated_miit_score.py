"""Post-prediction MIIT development scoring, never registration or selection.

Three section pairs come from one previously used prostate specimen. Native
CSV locations are interpreted as zero-based pixel centers, an explicit
resampling convention, not verified physical spacing or official scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.coordinated_lung_all20_score import _affine, _native, _safe_map
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_birl_landmark_score import (
    original_pixel_to_canvas_unit, canvas_unit_to_original_pixel,
)
from tools.digital_compare_appearance import dhr_map_at_unit_queries


PAIRS = ((2, 3), (7, 8), (10, 11))  # moving, fixed; NOT map evaluation order
METHODS = ("common_affine", "analytic", "f2", "dhr")
TERMINAL = {"ok", "failed", "skipped"}


def read_points(path: Path, wh, *, expected_count=124, allow_missing_inf=False):
    """Require every declared label, with no intersect-and-drop policy."""
    values = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"label", "x", "y"}.issubset(reader.fieldnames or []):
            raise ValueError("label,x,y CSV columns required")
        for row in reader:
            label = row["label"].strip()
            if not label or label in values:
                raise ValueError("unique nonempty label required")
            xy = np.asarray([float(row["x"]), float(row["y"])])
            missing = bool(allow_missing_inf and np.isposinf(xy).all())
            if not missing and (not np.isfinite(xy).all() or np.any(xy < -.5) or np.any(xy > np.asarray(wh)-.5)):
                raise ValueError("finite native-image pixel location required")
            values[label] = xy
    if len(values) != expected_count:
        raise ValueError(f"all {expected_count} declared landmarks required")
    return values


def validate_layout(value):
    if value.get("side") != 512:
        raise ValueError("declared 512 canvas required")
    for key in ("fixed", "moving"):
        item = value[key]
        wh = np.asarray(item["original_wh"], dtype=float)
        resized = np.asarray(item["resized_wh"], dtype=float)
        scale = np.asarray(item["effective_original_to_canvas_scale_xy"], dtype=float)
        pad = np.asarray(item["padding_xy"], dtype=float)
        if any(a.shape != (2,) or not np.isfinite(a).all() for a in (wh,resized,scale,pad)):
            raise ValueError("finite two-axis layout required")
        if np.any(wh <= 0) or np.any(resized <= 0) or np.any(pad < 0) or np.any(pad+resized > 512):
            raise ValueError("invalid native/resize/padding sizes")
        if not np.array_equal(scale, resized/wh):
            raise ValueError("exact saved resize scale required")
    return value


def metrics(predicted, target_px, layout, ids):
    target = original_pixel_to_canvas_unit(target_px, layout, 512)
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != target.shape or not np.isfinite(predicted).all():
        raise ValueError("finite prediction for every required label required")
    canvas = np.linalg.norm((predicted-target)*512, axis=1)
    native = np.linalg.norm(canvas_unit_to_original_pixel(predicted,layout,512)-target_px,axis=1)
    def summarize(errors):
        return dict(mean=float(errors.mean()), p90=float(np.percentile(errors,90)),
            maximum=float(errors.max()), per_label=dict(zip(ids,errors.tolist(),strict=True)))
    return dict(canvas_pixels=summarize(canvas), native_moving_pixels=summarize(native))


def load_manifest(path):
    path = Path(path)
    if path.is_dir(): path = path/"predictions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("prediction_complete") is not True or value.get("annotations_read") is not False:
        raise ValueError("completed label-free predictions required BEFORE labels")
    timed=value.get('budget_mode')=='timed_final'
    if timed and (value.get('all50_terminal') is not True or value.get('seconds_per_axis')!=2.
            or value.get('timed_arm') not in ('adam','data_metric')):
        raise ValueError('explicit completed paired timed-final protocol required BEFORE labels')
    rows = value.get("rows")
    if not isinstance(rows,list) or len(rows) != 3:
        raise ValueError("all three predeclared directions required")
    for row,(moving,fixed) in zip(rows,PAIRS,strict=True):
        if (row.get("name"),row.get("moving_section"),row.get("fixed_section")) != (
            f"miit_{moving}_to_{fixed}",moving,fixed):
            raise ValueError("fixed metadata-selected cohort required")
        if row.get("map_direction") != "fixed_canvas_to_moving_canvas":
            raise ValueError("explicit fixed-to-moving map direction required")
        if row.get("status") not in TERMINAL | {"partial_failure"}:
            raise ValueError("nonterminal prediction row")
        if set(row.get("methods",{})) != {"analytic","f2","dhr"} or any(
            v.get("status") not in TERMINAL for v in row["methods"].values()):
            raise ValueError("all method attempts must be terminal")
        analytic=row['methods']['analytic']
        if timed and analytic.get('status')=='ok':
            budget=analytic.get('terminal_budget',{})
            if (budget.get('protocol')!='timed_final' or budget.get('seconds_per_axis')!=2.
                    or budget.get('arm')!=value['timed_arm'] or analytic.get('arm')!=value['timed_arm']):
                raise ValueError('successful timed analytic record must match declared paired arm BEFORE labels')
    return value,path.parent


def _analytic_map(record,directory,a,b):
    """Only an explicitly declared joint analytic export may change its affine."""
    if record.get('terminal_budget',{}).get('protocol')=='timed_final':
        from tools.coordinated_data_metric_batch import validate_timed_budget
        report_path=Path(record['report'])
        if not report_path.is_absolute():report_path=directory/report_path
        report=json.loads(report_path.read_text(encoding='utf-8'))
        validate_timed_budget(report,arm=record.get('arm'))
        if any(record.get(k)!=report.get(k) for k in ('gradient_steps','prefix_gradient_steps',
                'suffix_gradient_steps','terminal_budget','failed_trials')):
            raise ValueError('timed prediction/report accounting mismatch')
    if record.get("pose_mode")!="joint_positive_affine":
        return _safe_map(record,directory,a,b)
    from tools.coordinated_joint_pose import validate_joint_export
    from tools.digital_q1_real_eval import load_effective_vertices
    resolve=lambda value:Path(value) if Path(value).is_absolute() else directory/value
    path=resolve(record["output"]);report_path=resolve(record["report"])
    report=json.loads(report_path.read_text(encoding="utf-8"))
    config=report.get("configuration",{})
    if (config.get("interpolation")!="p1_ac" or config.get("pose_mode")!="joint_positive_affine"
            or config.get("minimum_jacobian")!=.001 or report.get("landmarks_used") is not False):
        raise ValueError("explicit image-only joint P1ac report with original .001 floor required")
    with np.load(path,allow_pickle=False) as archive:
        if (str(archive["interpolation"].item())!="p1_ac"
                or not np.array_equal(archive["original_post_affine_matrix"],a)
                or not np.array_equal(archive["original_post_affine_offset"],b)):
            raise ValueError("joint export must retain the exact original common affine")
    validation=validate_joint_export(path,minimum_jacobian=.001)
    if validation.get("valid") is not True:raise ValueError("joint combined-affine export validation failed")
    vertices,certificate=load_effective_vertices(path)
    if not certificate["composite_representation_valid"] or not np.isfinite(vertices).all():
        raise ValueError("joint saved residual/combined-affine certificate invalid")
    return vertices,dict(path=str(path),report=str(report_path),interpolation="p1_ac",
        pose_mode="joint_positive_affine",certificate=certificate,joint_export_validation=validation,
        certificate_scope="original affine retained; actual joint pose/combined affine/floors validated; independent P1 query evaluation",
        **{key:report.get(key) for key in ("gradient_steps","failed_trials","objective_evaluations",
            "optimize_seconds","output_selection","selected_stage")})


def score(predictions: Path, source_data: Path, *, allow_missing_inf=False):
    manifest,directory = load_manifest(predictions)  # MUST precede coordinate access.
    resolve = lambda name: Path(name) if Path(name).is_absolute() else directory/name
    rows = []
    reference_ids = None
    for source in manifest["rows"]:
        row = dict(name=source["name"], moving_section=source["moving_section"],
            fixed_section=source["fixed_section"], methods={})
        try:
            layout = validate_layout(json.loads(resolve(source["layout"]).read_text(encoding="utf-8")))
            for key in ("fixed","moving"):
                section = source[key+"_section"]
                image_path = Path(source_data)/str(section)/"images"/"image.tif"
                with Image.open(image_path) as image:
                    if list(image.size) != layout[key]["original_wh"] or image.getexif().get(274,1)!=1:
                        raise ValueError("layout must retain original native TIFF dimensions/orientation")
            a,b = _affine(resolve(source["affine"]))
            prepared = {"common_affine": (a,b)}
            for method in ("analytic","f2","dhr"):
                record = source["methods"][method]
                if record["status"] != "ok":
                    row["methods"][method] = dict(status=record["status"],error=record.get("error"))
                    continue
                try:
                    prepared[method] = (_native(record,directory,a,b) if method=="dhr" else
                        _analytic_map(record,directory,a,b) if method=="analytic" else _safe_map(record,directory,a,b))
                except Exception as error:
                    row["methods"][method]=dict(status="failed",error=f"{type(error).__name__}: {error}")
            points = {}
            for key in ("fixed","moving"):
                section = source[key+"_section"]
                points[key] = read_points(Path(source_data)/str(section)/"landmarks"/f"{section:02d}.csv",
                    layout[key]["original_wh"],allow_missing_inf=allow_missing_inf)
            if set(points["fixed"]) != set(points["moving"]):
                raise ValueError("all 124 labels must match; no intersection dropping")
            nominal_ids = sorted(points["fixed"])
            if reference_ids is not None and nominal_ids != reference_ids:
                raise ValueError("all selected sections must retain the same 124 labels")
            reference_ids = nominal_ids
            absent = {key:[label for label in nominal_ids if np.isposinf(points[key][label]).all()]
                for key in ("fixed","moving")}
            absent_any = set(absent["fixed"])|set(absent["moving"])
            ids = [label for label in nominal_ids if label not in absent_any]
            if not ids: raise ValueError("nonempty available paired annotation set required")
            fixed = np.stack([points["fixed"][label] for label in ids])
            moving = np.stack([points["moving"][label] for label in ids])
            query = original_pixel_to_canvas_unit(fixed,layout["fixed"],512)
            if np.any(query<0) or np.any(query>1): raise ValueError("fixed query outside declared domain")
            row.update(nominal_landmarks=124,required_available_landmarks=len(ids),scored_landmarks=len(ids),
                unavailable_fixed_labels=absent["fixed"],unavailable_moving_labels=absent["moving"],
                unavailable_pair_labels=sorted(absent_any),available_pair_labels=ids,
                availability_scope="annotation-only (+inf,+inf) absence; SAME fixed eligible set for every method, no prediction-based dropping")
            for method,data in prepared.items():
                try:
                    metadata = {}
                    if method=="common_affine": mapped = query@a.astype(float).T+b.astype(float)
                    elif method=="dhr":
                        field,params,metadata = data
                        mapped = dhr_map_at_unit_queries(field,params,fixed_size=(512,512),
                            moving_size=(512,512),query=torch.from_numpy(query.astype(np.float32)).reshape(1,1,-1,2)
                        )[0,0].cpu().numpy().astype(float)
                    else:
                        vertices,metadata = data
                        mapped = p1_at_queries_numpy(vertices,query,"ac")
                    row["methods"][method]=dict(status="ok",metadata=metadata,
                        metrics=metrics(mapped,moving,layout["moving"],ids))
                except Exception as error:
                    row["methods"][method]=dict(status="failed",error=f"{type(error).__name__}: {error}")
        except Exception as error:
            row["label_or_input_error"]=f"{type(error).__name__}: {error}"
            for method in METHODS:
                row["methods"][method]=dict(status="failed",error=row["label_or_input_error"])
        rows.append(row)
    aggregate = {}
    for method in METHODS:
        ok=[row for row in rows if row["methods"][method]["status"]=="ok"]
        summary = {unit:dict(mean_pair_mean=float(np.mean([r["methods"][method]["metrics"][unit]["mean"] for r in ok])),
            mean_pair_p90=float(np.mean([r["methods"][method]["metrics"][unit]["p90"] for r in ok])))
            for unit in ("canvas_pixels","native_moving_pixels")} if ok else None
        aggregate[method]=dict(pair_denominator=3,scored_pairs=len(ok),failed_pairs=3-len(ok),
            all_three=summary if len(ok)==3 else None,successful_only=summary)
    return dict(scope="three correlated directions from ONE previously used prostate sample; development transfer, not blind/SOTA",
        units="512canvas and original moving native pixels; no physical spacing claim",
        coordinate_convention="CSV native x,y treated as zero-based pixel centers; publisher exact origin unverified",
        missing_annotation_policy="explicit (+inf,+inf) absence allowed" if allow_missing_inf else "strict all124 finite",
        failure_policy="all124 nominal IDs required; every available finite pair required for ALL methods; missing labels reported; prediction failures retained; no incomplete all-three aggregate or affine fallback",
        prediction_manifest=str(predictions),rows=rows,aggregate=aggregate)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("predictions","source_data","output"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--allow-missing-inf",action="store_true",
        help="explicit released MIIT (+inf,+inf) missing-annotation convention; NaN/partial/nonpositive inf still rejected")
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("new scoring report required")
    result=score(args.predictions,args.source_data,allow_missing_inf=args.allow_missing_inf)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result["aggregate"],allow_nan=False))


if __name__=="__main__": main()
