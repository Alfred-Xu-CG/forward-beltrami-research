"""Read-only accepted-stage MIIT DEVELOPMENT scoring, never TRE selection.

One previously used prostate specimen; six instrumented, unchanged replays.
Snapshot-copy times are diagnostic and cannot replace clean application timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch  # Load the established numerical runtime before NumPy.
import numpy as np
from PIL import Image

from tools.coordinated_miit_score import load_manifest,validate_layout,read_points,metrics
from tools.coordinated_lung_all20_score import _affine,_safe_map
from tools.coordinated_replay_trajectory import score_arrays
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit

METHODS=("analytic","f2")
AVAILABLE=(123,107,98)


def _json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(directory,path):
    value=Path(path)
    return value if value.is_absolute() else directory/value


def _recipe(value):
    result={k:v for k,v in value.items() if k not in ("output","device")}
    # Explicit current logging default is identical to the historical omitted default.
    result.setdefault("trial_diagnostics","existing")
    return result


def _corners(vertices):
    a,b,c,d=vertices[:-1,:-1],vertices[:-1,1:],vertices[1:,1:],vertices[1:,:-1]
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    return np.stack((cross(b-a,d-a),cross(b-a,c-b),cross(c-d,c-b),cross(c-d,d-a)))


def check_snapshots(vertices,reference,eta):
    """Literal actual four-corner determinant and exact boundary checks."""
    if vertices.shape!=(10,1,257,257,2) or vertices.dtype!=np.float64:
        raise ValueError("ten batch1 float64 257-square accepted maps required")
    if reference.shape!=(1,257,257,2) or reference.dtype!=np.float64:
        raise ValueError("actual saved float64 material reference required")
    axis=np.linspace(0.,1.,257);x,y=np.meshgrid(axis,axis)
    identity=np.stack((x,y),-1)[None]
    if not np.array_equal(reference,identity):raise ValueError("declared unit-rectangle reference required")
    qref=_corners(reference[0]);result=[]
    for v in vertices[:,0]:
        if not np.isfinite(v).all():raise ValueError("finite actual snapshot required")
        if any(not np.array_equal(v[index],reference[0][index]) for index in
               ((0,slice(None)),(-1,slice(None)),(slice(None),0),(slice(None),-1))):
            raise ValueError("exact residual boundary required for every snapshot")
        ratios=_corners(v)/qref
        if not np.isfinite(ratios).all() or not (ratios>eta).all():
            raise ValueError("every actual snapshot corner must strictly exceed eta")
        result.append(dict(minimum_normalized_corner=float(ratios.min()),corner_count=ratios.size))
    return result


def selected_prefix(totals,initial):
    """Strict objective comparison, identity first; ties never choose by TRE."""
    best=float(initial);index=None
    if not np.isfinite(best) or not np.isfinite(totals).all():raise ValueError("finite full objectives required")
    for i,value in enumerate(totals):
        if value<best:best,index=float(value),i
    return index,best


def load_replay(source,method,directory,replay_directory,a,b):
    record=source["methods"][method]
    if record["status"]!="ok":raise ValueError("original successful method required for unchanged replay")
    stem=source["name"]+"_"+method+"_replay"
    report=_json(replay_directory/(stem+".json"))
    metadata=_json(replay_directory/(stem+"_trajectory.json"))
    original=_json(_resolve(directory,record["report"]))
    if metadata.get("manual_labels_loaded_during_record") is not False:
        raise ValueError("explicit label-off replay metadata required BEFORE annotations")
    if (Path(metadata.get("configuration_source","")).name!=Path(record["report"]).name
            or Path(metadata.get("output_map","")).name!=stem+".npz"):
        raise ValueError("replay source/output provenance mismatch")
    if _recipe(report["configuration"])!=_recipe(original["configuration"]):
        raise ValueError("replay recipe differs from original beyond output/device")
    config=report["configuration"]
    if (config.get("method")!=method or config.get("grid_side")!=257 or config.get("precision")!="float64"
            or config.get("interpolation")!="p1_ac" or config.get("output_selection")!="best_full"
            or config.get("levels")!=[17,33,65,129,257] or config.get("image_levels")!=[32,64,128,256,512]
            or config.get("inner_steps_by_level",[config.get("inner_steps")]*5)!=[30]*5
            or config.get("cycles")!=(1 if method=="analytic" else 2)):
        raise ValueError("unchanged declared 300-gradient P1ac recipe required")
    for key,wanted in (("gradient_steps",300),("evaluations",310),("objective_evaluations",332),("failed_trials",0)):
        if report.get(key)!=wanted:raise ValueError("replay counter mismatch: "+key)
    if metadata.get("gradient_steps")!=300 or report.get("landmarks_used") is not False:
        raise ValueError("label-free complete replay required")
    stages=metadata["stages"]
    if len(stages)!=10 or stages!=report["stages"] or metadata["initial_objective"]!=report["initial"]:
        raise ValueError("trajectory stage/objective bookkeeping mismatch")
    expected=[(0,level,direction,image) for level,image in zip(config["levels"],config["image_levels"],strict=True)
        for direction in ([1.,0.],[0.,1.])] if method=="analytic" else [
        (cycle,level,None,image) for cycle in range(2) for level,image in zip(config["levels"],config["image_levels"],strict=True)]
    for stage,(cycle,level,direction,image) in zip(stages,expected,strict=True):
        if (stage.get("cycle"),stage.get("level"),stage.get("direction"),stage.get("image_side"),stage.get("inner_steps"))!=(cycle,level,direction,image,30):
            raise ValueError("replay stage allocation/direction mismatch")
    replay_record=dict(output=stem+".npz",report=stem+".json")
    _safe_map(replay_record,replay_directory,a,b)
    _safe_map(record,directory,a,b)
    with np.load(replay_directory/(stem+".npz"),allow_pickle=False) as saved:
        endpoint=saved["vertices"].copy();reference=saved["boundary_reference"].copy()
    with np.load(_resolve(directory,record["output"]),allow_pickle=False) as saved:
        archived=saved["vertices"].copy()
    with np.load(replay_directory/(stem+"_trajectory.npz"),allow_pickle=False) as saved:
        vertices=saved["vertices"].copy();times=saved["optimizer_seconds"].copy()
    geometry=check_snapshots(vertices,reference,config["minimum_jacobian"])
    if times.shape!=(10,) or not np.isfinite(times).all() or times[0]<0 or (np.diff(times)<0).any():
        raise ValueError("ten finite chronological snapshot times required")
    totals=[stage["accepted_full_total"] for stage in stages]
    index,total=selected_prefix(totals,report["initial"]["total"])
    selected=reference if index is None else vertices[index]
    if report.get("selected_stage")!=index or report["final"]["total"]!=total or not np.array_equal(endpoint,selected):
        raise ValueError("output must equal complete-E1-selected prefix including identity")
    difference=endpoint-archived
    return dict(vertices=vertices[:,0],times=times,totals=totals,initial_total=report["initial"]["total"],
        report=report,metadata=metadata,geometry=geometry,
        endpoint_comparison=dict(max_absolute=float(np.abs(difference).max()),rms=float(np.sqrt(np.mean(difference**2))),
            archived_selected_stage=original.get("selected_stage"),replay_selected_stage=index,
            scope="repeat differences reported, not an arbitrary GPU tolerance gate"))


def score(predictions,source_data,replay_directory):
    manifest,directory=load_manifest(predictions)  # Must precede any annotation read.
    rows=[];prepared={}
    # Validate all six label-off recipes and maps BEFORE accessing coordinates.
    for source in manifest["rows"]:
        row=dict(name=source["name"],methods={});rows.append(row)
        try:
            a,b=_affine(_resolve(directory,source["affine"]))
            for method in METHODS:
                try:prepared[(source["name"],method)]=load_replay(source,method,directory,Path(replay_directory),a,b)
                except Exception as error:row["methods"][method]=dict(status="failed",error=f"{type(error).__name__}: {error}")
            prepared[(source["name"],"affine")]=(a.astype(float),b.astype(float))
        except Exception as error:
            row["methods"]={m:dict(status="failed",error=f"{type(error).__name__}: {error}") for m in METHODS}
    global_ids=None
    for pair_index,(source,row) in enumerate(zip(manifest["rows"],rows,strict=True)):
        if all(m in row["methods"] for m in METHODS):continue
        try:
            layout=validate_layout(_json(_resolve(directory,source["layout"])))
            points={}
            for key in ("fixed","moving"):
                section=source[key+"_section"];root=Path(source_data)/str(section)
                with Image.open(root/"images"/"image.tif") as image:
                    if list(image.size)!=layout[key]["original_wh"] or image.getexif().get(274,1)!=1:
                        raise ValueError("native dimensions/orientation differ from layout")
                points[key]=read_points(root/"landmarks"/f"{section:02d}.csv",layout[key]["original_wh"],allow_missing_inf=True)
            ids=sorted(points["fixed"])
            if set(ids)!=set(points["moving"]) or global_ids is not None and ids!=global_ids:
                raise ValueError("exact124 nominal globalIDs required; no intersection dropping")
            global_ids=ids
            absent={key:[i for i in ids if np.isposinf(points[key][i]).all()] for key in points}
            available=[i for i in ids if i not in set(absent["fixed"])|set(absent["moving"])]
            if len(available)!=AVAILABLE[pair_index]:raise ValueError("predeclared annotation-only123/107/98 availability required")
            fixed=np.stack([points["fixed"][i] for i in available]);moving=np.stack([points["moving"][i] for i in available])
            query=original_pixel_to_canvas_unit(fixed,layout["fixed"],512)
            expected=original_pixel_to_canvas_unit(moving,layout["moving"],512)
            a,b=prepared[(source["name"],"affine")]
            row.update(nominal_ids=ids,available_ids=available,unavailable=absent,required_available=len(available))
            for method in METHODS:
                if method in row["methods"]:continue
                data=prepared[(source["name"],method)]
                result=score_arrays(data["vertices"],query,expected,a,b,512,data["totals"],data["initial_total"],data["times"],"ac")
                for i,stage in enumerate(result["stages"]):
                    selected=stage["selected_prefix_stage"]
                    for name,index in (("accepted",i),("image_selected_prefix",selected)):
                        mapped=query if index is None else p1_at_queries_numpy(data["vertices"][index],query,"ac")
                        stage[name+"_errors"]=metrics(mapped@a.T+b,moving,layout["moving"],available)
                    info=data["metadata"]["stages"][i]
                    stage.update(level=info["level"],image_side=info["image_side"],direction=info["direction"],cycle=info["cycle"],
                        cumulative_gradients=30*(i+1),cumulative_trials=31*(i+1),cumulative_objective_evaluations=1+33*(i+1),
                        actual_geometry=data["geometry"][i])
                row["methods"][method]=dict(status="ok",trajectory=result,endpoint_comparison=data["endpoint_comparison"],
                    counters={k:data["report"][k] for k in ("gradient_steps","evaluations","objective_evaluations","failed_trials")})
        except Exception as error:
            for method in METHODS:row["methods"][method]=dict(status="failed",error=f"{type(error).__name__}: {error}")
    aggregate={}
    for method in METHODS:
        ok=[r for r in rows if r["methods"][method]["status"]=="ok"]
        table=[dict(stage=i,cumulative_gradients=30*(i+1),**{
            name+"_equal_pair_"+metric:float(np.mean([r["methods"][method]["trajectory"]["stages"][i][name][metric+"_canvas_px"] for r in ok]))
            for name in ("accepted","image_selected_prefix") for metric in ("mean","p90")}) for i in range(10)] if len(ok)==3 else None
        aggregate[method]=dict(pair_denominator=3,scored_pairs=len(ok),failed_pairs=3-len(ok),all_three_prefix_table=table)
    return dict(scope="three correlated directions from ONE previously used MIIT prostate specimen; development, not blind",
        units="512canvas and original moving native pixels; no physical spacing claim",
        coordinate_convention="CSV native x,y treated as zero-based pixel centers; publisher exact origin unverified",
        posthoc_only=True,predictions=str(predictions),replay_directory=str(replay_directory),rows=rows,aggregate=aggregate,
        selection="strict minimum complete E1 among identity and accepted prefix; NEVER TRE",
        counting="initial E1=1; each stage30gradients/31trials/33E; prefix E=1+33n; final export evaluation adds1 for332",
        timing="instrumented snapshot-copy times ONLY diagnostic, not clean application speed",
        geometry_scope="actual rounded four-corner>eta, fixed boundary, common positive affine; no continuum/reinterpolation claim")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("predictions","source_data","replay_directory","output"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("new posthoc output required")
    result=score(args.predictions,args.source_data,args.replay_directory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result["aggregate"],allow_nan=False))


if __name__=="__main__":main()
