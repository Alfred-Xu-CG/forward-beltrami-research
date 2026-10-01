"""Image-only raw SuperGlue correspondences; no map truth or manual labels.

Reuses installed DeeperHistReg weights. Unlike global-affine RANSAC, this probe
does not discard a match merely because it follows a spatially nonrigid motion.
Model confidence is NOT an anatomical correctness certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def raw_correspondences(points0,points1,matches0,scores0,*,height,width):
    p0,p1=np.asarray(points0),np.asarray(points1)
    matches,scores=np.asarray(matches0),np.asarray(scores0)
    if p0.ndim!=2 or p1.ndim!=2 or p0.shape[-1]!=2 or p1.shape[-1]!=2:
        raise ValueError("keypoints must be Nx2")
    if matches.shape!=(len(p0),) or scores.shape!=(len(p0),) or min(height,width)<2:
        raise ValueError("one match index/confidence per fixed keypoint required")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all() or not np.isfinite(scores).all():
        raise ValueError("nonfinite keypoint or confidence")
    if not np.issubdtype(matches.dtype,np.integer) or (matches < -1).any() or (matches>=len(p1)).any():
        raise ValueError("match must be -1 or a valid moving-keypoint index")
    if ((scores<0)|(scores>1)).any():
        raise ValueError("confidence must be in[0,1]")
    chosen=matches>=0
    source=(p0[chosen].astype(np.float64)+.5)/np.array([width,height])
    target=(p1[matches[chosen]].astype(np.float64)+.5)/np.array([width,height])
    if ((source<0)|(source>1)).any() or ((target<0)|(target>1)).any():
        raise ValueError("keypoint center outside declared raster domain")
    bins=np.minimum(3,np.floor(source*4).astype(int))
    return dict(status="ok" if chosen.sum()>=8 else "insufficient_matches",
        raw_matches=int(chosen.sum()),occupied_quadrants_4x4=len(set(map(tuple,bins))),
        source_points_unit=source.tolist(),target_points_unit=target.tolist(),
        confidence=scores[chosen].astype(np.float64).tolist(),
        coordinate_convention="fixed unit q -> affine-aligned moving unit p; full moving point is A p+b",
        global_geometric_ransac_used=False)


def extract(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    import deeperhistreg  # noqa: F401: installed package supplies legacy aliases
    import superpoint_superglue as module
    device=torch.device(args.device)
    started=time.perf_counter()
    fixed,_=_read_gray_thumbnail(args.fixed,args.image_side)
    moving,_=_read_gray_thumbnail(args.moving,args.image_side)
    with np.load(args.affine) as archive:
        matrix=torch.from_numpy(archive["post_affine_matrix"].copy())
        offset=torch.from_numpy(archive["post_affine_offset"].copy())
    aligned=warp_moving_to_fixed(moving,matrix,offset,height=args.image_side,width=args.image_side)
    matcher=module.sg.Matching({
        "superpoint":{"nms_radius":4,"keypoint_threshold":.005,"max_keypoints":3000},
        "superglue":{"weights":"outdoor","sinkhorn_iterations":30,"match_threshold":.3},
    }).eval().to(device)
    matcher.superpoint.load_state_dict(torch.load(module.p.superpoint_model_path,map_location=device,weights_only=True))
    matcher.superglue.load_state_dict(torch.load(module.p.superglue_model_path,map_location=device,weights_only=True))
    if device.type=="cuda":
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    setup_seconds=time.perf_counter()-started
    tick=time.perf_counter()
    with torch.no_grad():
        prediction=matcher({"image0":fixed.to(device),"image1":aligned.to(device)})
        prediction={k:v[0].detach().cpu().numpy() for k,v in prediction.items()}
    if device.type=="cuda":
        torch.cuda.synchronize(device)
    report=raw_correspondences(prediction["keypoints0"],prediction["keypoints1"],
        prediction["matches0"],prediction["matching_scores0"],height=args.image_side,width=args.image_side)
    report.update(fixed=str(args.fixed),moving=str(args.moving),affine=str(args.affine),
        post_affine_matrix=matrix.numpy().tolist(),post_affine_offset=offset.numpy().tolist(),
        image_side=args.image_side,fixed_keypoints=len(prediction["keypoints0"]),moving_keypoints=len(prediction["keypoints1"]),
        method="frozen installed DHR SuperPoint/SuperGlue outdoor; threshold.3; no global RANSAC",
        setup_seconds=setup_seconds,inference_seconds=time.perf_counter()-tick,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        targets_manual_landmarks_or_dense_teacher_loaded=False,
        licensing="existing SuperGlue pretrained weights: noncommercial research use; not a new trained model")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_points_unit","target_points_unit","confidence")}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fixed","moving","affine","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--image-side",type=int,default=512)
    p.add_argument("--device",default="cpu")
    args=p.parse_args();torch.set_num_threads(2)
    extract(args)


if __name__=="__main__":
    main()
