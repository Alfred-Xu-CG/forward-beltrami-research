"""Target-free optimizer replay, then SEPARATE read-only development scoring.

Research question: when does the image-selected prefix reach useful anatomical
accuracy? Record reads ONLY a prior executable configuration and images/matches.
Score reads landmarks afterward; it never changes iterates or selection. A
trajectory is one compact archive, not one artifact directory per trial.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def record(configuration,output,*,device=None):
    from tools.coordinated_real_case import optimize
    import torch
    archive=output.with_name(output.stem+"_trajectory.npz")
    metadata=archive.with_suffix(".json")
    if archive.exists() or metadata.exists():
        raise FileExistsError(archive)
    parameters=json.loads(configuration.read_text(encoding="utf-8"))["configuration"]
    for name in ("fixed","moving","affine","output","matches"):
        if parameters.get(name) is not None:
            parameters[name]=Path(parameters[name])
    parameters["output"]=output
    if device is not None:
        parameters["device"]=device
    snapshots=[]
    def observe(vertices,stage,elapsed):
        snapshots.append((vertices.cpu().clone(),stage,elapsed))
    report=optimize(argparse.Namespace(**parameters),accepted_stage_callback=observe)
    tick=time.perf_counter()
    np.savez_compressed(archive,vertices=torch.stack([v for v,_,_ in snapshots]).numpy(),
                        optimizer_seconds=np.asarray([t for _,_,t in snapshots]))
    payload=dict(configuration_source=str(configuration),output_map=str(output),
        stages=[s for _,s,_ in snapshots],initial_objective=report["initial"],
        optimizer_seconds=report["optimize_seconds"],gradient_steps=report["gradient_steps"],
        snapshot_cpu_bytes=sum(v.numel()*v.element_size() for v,_,_ in snapshots),
        snapshot_copy_overhead_included_in_optimizer_time=True,
        trajectory_serialization_seconds=time.perf_counter()-tick,
        manual_labels_loaded_during_record=False,
        selection="minimum complete full-resolution objective among initial/accepted prefix maps, never TRE")
    metadata.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload


def score_arrays(vertices,query,expected,matrix,offset,side,full_totals,initial_total,times,diagonal):
    from tools.coordinated_score_case import p1_at_queries_numpy
    if len(vertices)!=len(full_totals) or len(vertices)!=len(times):
        raise ValueError("one objective/time per accepted map required")
    def metrics(mapped):
        errors=np.linalg.norm((mapped-expected)*side,axis=-1)
        return dict(mean_canvas_px=float(errors.mean()),p90_canvas_px=float(np.percentile(errors,90)),
                    max_canvas_px=float(errors.max()),required_landmarks=len(errors))
    initial=metrics(query@matrix.T+offset)
    best_total,best_metrics,best_index=initial_total,initial,None
    rows=[]
    for i,v in enumerate(vertices):
        accepted=metrics(p1_at_queries_numpy(v,query,diagonal)@matrix.T+offset)
        if full_totals[i]<best_total:
            best_total,best_metrics,best_index=full_totals[i],accepted,i
        rows.append(dict(stage=i,optimizer_seconds=float(times[i]),accepted=accepted,
            image_selected_prefix=best_metrics,selected_prefix_stage=best_index,
            best_full_total=float(best_total),accepted_full_total=float(full_totals[i])))
    return dict(initial=initial,stages=rows,
                caution="accepted-map TRE and image-selected-prefix TRE differ; neither selects iterates using landmarks")


def score(args):
    from PIL import Image
    from tools.digital_dhr_field_eval import _landmarks
    from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
    if args.output.exists():raise FileExistsError(args.output)
    metadata=json.loads(args.trajectory.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("manual_labels_loaded_during_record") is not False:
        raise ValueError("trajectory must explicitly exclude evaluation labels during recording")
    layout=json.loads(args.layout.read_text(encoding="utf-8"))
    sizes={}
    for name in ("fixed","moving"):
        with Image.open(layout[name]["source"]) as image:sizes[name]=image.size
    fixed=_landmarks(args.fixed_landmarks,sizes["fixed"])
    moving=_landmarks(args.moving_landmarks,sizes["moving"])
    ids=sorted(fixed.keys()&moving.keys())
    if not ids:raise ValueError("no shared required landmark IDs")
    query=original_pixel_to_canvas_unit(np.stack([fixed[i] for i in ids]),layout["fixed"],layout["side"])
    expected=original_pixel_to_canvas_unit(np.stack([moving[i] for i in ids]),layout["moving"],layout["side"])
    with np.load(args.trajectory) as saved:
        vertices=saved["vertices"].copy();times=saved["optimizer_seconds"].copy()
    map_path=args.map if getattr(args,"map",None) is not None else Path(metadata["output_map"])
    with np.load(map_path) as saved:
        mode=str(saved["interpolation"].item())
        matrix=saved["post_affine_matrix"].astype(np.float64)
        offset=saved["post_affine_offset"].astype(np.float64)
    if mode not in ("p1_ac","p1_bd") or vertices.ndim!=5 or vertices.shape[1]!=1:
        raise ValueError("batch1 actual declared P1 trajectories only")
    result=score_arrays(vertices[:,0],query,expected,matrix,offset,layout["side"],
        [s["accepted_full_total"] for s in metadata["stages"]],metadata["initial_objective"]["total"],times,mode[-2:])
    result.update(protocol="reused development specimen, not independent patient confirmation",
        trajectory=str(args.trajectory),layout=str(args.layout),fixed_landmarks=str(args.fixed_landmarks),
        moving_landmarks=str(args.moving_landmarks),shared_ids=ids,
        output_map=str(map_path),
        fixed_only_ids=sorted(fixed.keys()-moving.keys()),moving_only_ids=sorted(moving.keys()-fixed.keys()),
        record_metadata=metadata,posthoc_only=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    rec=sub.add_parser("record")
    rec.add_argument("--configuration",type=Path,required=True);rec.add_argument("--output",type=Path,required=True)
    rec.add_argument("--device")
    scoring=sub.add_parser("score")
    for name in ("trajectory","layout","fixed_landmarks","moving_landmarks","output"):
        scoring.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    scoring.add_argument("--map",type=Path,help="explicit saved output map path after relocating remote artifacts")
    args=parser.parse_args()
    result=record(args.configuration,args.output,device=args.device) if args.mode=="record" else score(args)
    print(json.dumps(result))


if __name__=="__main__":main()
