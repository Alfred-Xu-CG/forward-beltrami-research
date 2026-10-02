"""Offline frozen-affine descriptor ranking on ONE reused MIIT specimen.

No optimizer or map selection. The frozen affine is NOT the true local warp:
this changes orientation/scale, raster blur, and boundary effects together.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from tools.coordinated_miit_score import load_manifest,validate_layout,read_points
from tools.coordinated_miit_trajectory_score import AVAILABLE
from tools.coordinated_lung_all20_score import _affine,_safe_map
from tools.coordinated_real_case import load_registration_evidence
from tools.digital_mind_objective_probe import self_similarity
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit


def _resolve(directory,value):
    path=Path(value)
    return path if path.is_absolute() else directory/path


def footprint_support(points,side):
    """Nominal float64 Offset2+pool1+bilinear footprint, inclusive extrema."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError("finite Nx2 unit queries required")
    if isinstance(side,bool) or not isinstance(side,int) or side<8:raise ValueError("integer side>=8 required")
    lower=np.floor(side*points-.5)-3
    upper=np.ceil(side*points-.5)+3
    return (lower>=0).all(-1)&(upper<=side-1).all(-1),lower,upper


def affine_footprint_support(points,matrix,offset,side):
    """Aligned descriptor footprint AND moving bilinear support at its extrema."""
    valid,lower,upper=footprint_support(points,side)
    corners=np.stack((lower,np.stack((upper[:,0],lower[:,1]),-1),upper,
                      np.stack((lower[:,0],upper[:,1]),-1)),1)
    mapped=((corners+.5)/side)@np.asarray(matrix,dtype=float).T+np.asarray(offset,dtype=float)
    indices=side*mapped-.5
    original=(np.floor(indices)>=0).all((1,2))&(np.ceil(indices)<=side-1).all((1,2))
    return valid&original


def _sample(field,points):
    points=torch.as_tensor(points,dtype=torch.float64,device=field.device)
    grid=(2*points-1).to(field.dtype).reshape(1,1,-1,2)
    return F.grid_sample(field,grid,mode="bilinear",padding_mode="zeros",align_corners=False)[0,:,0].T


@torch.no_grad()
def probe_arrays(fixed,moving,matrix,offset,query,truth,current):
    """Frozen float32 features, float64 coordinate arithmetic; no argmin/VJP."""
    if (fixed.shape!=moving.shape or fixed.ndim!=4 or fixed.shape[:2]!=(1,1)
            or fixed.shape[-2]!=fixed.shape[-1] or fixed.dtype!=torch.float32 or moving.dtype!=torch.float32
            or fixed.device.type!="cpu" or moving.device!=fixed.device
            or not bool(torch.isfinite(fixed).all() and torch.isfinite(moving).all())):
        raise ValueError("finite CPU batch1 square float32 prepared intensities required")
    side=fixed.shape[-1];matrix=np.asarray(matrix,dtype=np.float64);offset=np.asarray(offset,dtype=np.float64)
    query,truth,current=(np.asarray(p,dtype=np.float64) for p in (query,truth,current))
    if matrix.shape!=(2,2) or offset.shape!=(2,) or not np.isfinite(matrix).all() or not np.isfinite(offset).all() or np.linalg.det(matrix)<=0:
        raise ValueError("finite positive affine required")
    if query.shape!=truth.shape or query.shape!=current.shape or len(query)==0:raise ValueError("same nonempty query sets required")
    for p in (query,truth,current):footprint_support(p,side)
    axis=(torch.arange(side,dtype=torch.float64)+.5)/side
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    pixel=torch.stack((x,y),-1)[None]
    aligned_query=pixel@torch.from_numpy(matrix).T+torch.from_numpy(offset)
    aligned=F.grid_sample(moving,(2*aligned_query-1).to(moving.dtype),mode="bilinear",padding_mode="zeros",align_corners=False)
    fixed_descriptor=self_similarity(fixed)[0]
    original_descriptor=self_similarity(moving)[0]
    aligned_descriptor=self_similarity(aligned)[0]
    sampled_fixed=_sample(fixed_descriptor,query)
    costs={};flags={"fixed":footprint_support(query,side)[0]}
    for name,p in (("truth",truth),("current",current)):
        original=p@matrix.T+offset
        costs["raw_"+name]=(sampled_fixed-_sample(original_descriptor,original)).abs().mean(1).numpy()
        costs["affine_"+name]=(sampled_fixed-_sample(aligned_descriptor,p)).abs().mean(1).numpy()
        flags["raw_"+name]=footprint_support(original,side)[0]
        flags["affine_"+name]=affine_footprint_support(p,matrix,offset,side)
    if any(not np.isfinite(v).all() for v in costs.values()):raise ValueError("finite costs for every retained ID required")
    costs["raw_margin"]=costs["raw_current"]-costs["raw_truth"]
    costs["affine_margin"]=costs["affine_current"]-costs["affine_truth"]
    costs["affine_minus_raw_margin"]=costs["affine_margin"]-costs["raw_margin"]
    flags["common"]=np.logical_and.reduce(list(flags.values()))
    return costs,flags


def summarize(costs,mask):
    count=int(np.asarray(mask).sum())
    if not count:return dict(count=0,statistics=None)
    result={key:dict(mean=float(v[mask].mean()),quantiles={str(q):float(np.quantile(v[mask],q))
        for q in (0.,.1,.25,.5,.75,.9,1.)}) for key,v in costs.items()}
    for name in ("raw","affine"):
        values=costs[name+"_margin"][mask]
        result[name+"_truth_preference"]=dict(strict_fraction=float((values>0).mean()),
            strict_count=int((values>0).sum()),tie_count=int((values==0).sum()))
    result["margin_improved_fraction"]=float((costs["affine_minus_raw_margin"][mask]>0).mean())
    return dict(count=count,statistics=result)


def probe(predictions,source_data):
    manifest,directory=load_manifest(predictions)  # MUST precede annotations.
    rows=[];global_ids=None;all_costs=[];all_flags=[]
    for source,count in zip(manifest["rows"],AVAILABLE,strict=True):
        row=dict(name=source["name"],status="failed");rows.append(row)
        try:
            record=source["methods"]["analytic"]
            if record["status"]!="ok":raise ValueError("successful saved analytic prediction required")
            a,b=_affine(_resolve(directory,source["affine"]))
            _safe_map(record,directory,a,b)
            report=json.loads(_resolve(directory,record["report"]).read_text(encoding="utf-8"))
            config=report["configuration"]
            if any(config.get(k)!=v for k,v in dict(grid_side=257,precision="float64",image_precision="float32",interpolation="p1_ac",preprocessing="raw_inverted").items()):
                raise ValueError("original257 P1ac float64 geometry/float32 raw feature recipe required")
            with np.load(_resolve(directory,record["output"]),allow_pickle=False) as archive:
                vertices=archive["vertices"].copy()
            if vertices.shape!=(1,257,257,2) or vertices.dtype!=np.float64:raise ValueError("saved residual float64 batch1 map required")
            layout=validate_layout(json.loads(_resolve(directory,source["layout"]).read_text(encoding="utf-8")))
            points={}
            for key in ("fixed","moving"):
                section=source[key+"_section"];root=Path(source_data)/str(section)
                with Image.open(root/"images"/"image.tif") as image:
                    if list(image.size)!=layout[key]["original_wh"] or image.getexif().get(274,1)!=1:raise ValueError("native dimensions/orientation mismatch")
                points[key]=read_points(root/"landmarks"/f"{section:02d}.csv",layout[key]["original_wh"],allow_missing_inf=True)
            ids=sorted(points["fixed"])
            if set(ids)!=set(points["moving"]) or global_ids is not None and ids!=global_ids:raise ValueError("all124 nominal globalIDs required, no intersection dropping")
            global_ids=ids
            absent={key:[i for i in ids if np.isposinf(points[key][i]).all()] for key in points}
            available=[i for i in ids if i not in set(absent["fixed"])|set(absent["moving"])]
            if len(available)!=count:raise ValueError("same predeclared123/107/98 annotation-only availability required")
            query=original_pixel_to_canvas_unit(np.stack([points["fixed"][i] for i in available]),layout["fixed"],512)
            target=original_pixel_to_canvas_unit(np.stack([points["moving"][i] for i in available]),layout["moving"],512)
            truth=np.linalg.solve(a.astype(float),(target-b.astype(float)).T).T
            current=p1_at_queries_numpy(vertices[0],query,"ac")
            fixed,moving,_,preprocessing=load_registration_evidence(_resolve(directory,source["fixed"]),
                _resolve(directory,source["moving"]),512,preprocessing="raw_inverted",device="cpu",dtype=torch.float32)
            costs,flags=probe_arrays(fixed,moving,a,b,query,truth,current)
            row.update(status="ok",nominal_ids=ids,available_ids=available,unavailable=absent,
                full=summarize(costs,np.ones(count,dtype=bool)),common_support=summarize(costs,flags["common"]),
                per_id={label:dict(query=query[i].tolist(),truth_residual=truth[i].tolist(),current_residual=current[i].tolist(),
                    **{k:float(v[i]) for k,v in costs.items()},support={k:bool(v[i]) for k,v in flags.items()}) for i,label in enumerate(available)},
                shared_affine=dict(matrix=a.tolist(),offset=b.tolist()),preprocessing=preprocessing)
            all_costs.append(costs);all_flags.append(flags["common"])
        except Exception as error:row["error"]=f"{type(error).__name__}: {error}"
    complete=len(all_costs)==3
    pooled={k:np.concatenate([c[k] for c in all_costs]) for k in all_costs[0]} if complete else None
    return dict(scope="ONE previously used MIIT prostate specimen, development diagnostic, not blind/clinical",
        predictions=str(predictions),pair_denominator=3,scored_pairs=len(all_costs),required_available_denominator=328,rows=rows,
        full_cohort=summarize(pooled,np.ones(328,dtype=bool)) if complete else None,
        common_support=summarize(pooled,np.concatenate(all_flags)) if complete else None,
        interpretation="positive margin means strictly prefers truth; pooled IDs are correlated, not328patients; lower absolute cost alone is not evidence of improved ranking",
        support_definition="pixel-index floor(512p-.5)-3 through ceil(512p-.5)+3; aligned footprint AND original moving bilinear support at ALLfour affine-mapped extreme pixel centers, for BOTHtruth/current andfixed",
        support_precision_scope="nominal float64 geometric support flags, NOT an exact certificate of every floating-point interpolation access; actual normalized grids are cast to float32 and boundary-near rounding may alter the effective footprint",
        limitations="shared affine is approximate, not a ground-truth local Jacobian; orientation/scale, interpolation blur and OOB effects change together; no optimizer pilot or TRE selection",
        coordinate_convention="native CSV zero-based pixel centers, exact publisher origin unverified; no physical spacing claim")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("predictions","source_data","output"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("new diagnostic report required")
    result=probe(args.predictions,args.source_data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("scored_pairs","full_cohort","common_support")},allow_nan=False))


if __name__=="__main__":main()
