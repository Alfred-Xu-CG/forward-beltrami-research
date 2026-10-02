"""Image-only four-quarter-turn SuperGlue positive-similarity initializer.

Selection uses match support and fit only. Original-frame keypoints enter the
existing RANSAC BEFORE fitting. No labels, registration loss or TRE is queried.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.digital_q1_superglue_residual import select_ransac_matches
from tools.digital_superglue_direct_affine import similarity_from_inlier_units


def unrotate_points(points,k,*,width,height):
    """Inverse torch.rot90 pixel-index transform, including non-square tests."""
    points=np.asarray(points,dtype=np.float64)
    if (isinstance(k,bool) or not isinstance(k,int) or k not in range(4)
            or isinstance(width,bool) or isinstance(height,bool)
            or not isinstance(width,int) or not isinstance(height,int) or min(width,height)<2
            or points.ndim!=2 or points.shape[-1]!=2 or not np.isfinite(points).all()):
        raise ValueError("finite Nx2 points, quarter turn0..3 and integer image dimensions required")
    x,y=points.T
    return (points.copy() if k==0 else np.column_stack((width-1-y,x)) if k==1
        else np.column_stack((width-1-x,height-1-y)) if k==2
        else np.column_stack((y,height-1-x)))


def select_candidate(candidates):
    eligible=[row for row in candidates if row.get("status")=="ok"]
    return min(eligible,key=lambda row:(-row["ransac_inliers"],-row["occupied_quadrants_4x4"],
        row["inlier_fit_rmse_512px"],row["quarter_turns"])) if eligible else None


def _installed_matcher(config,device):
    import deeperhistreg  # noqa: F401: existing legacy model aliases.
    import superpoint_superglue as module
    model=module.sg.Matching(config).eval().to(device)
    model.superpoint.load_state_dict(torch.load(module.p.superpoint_model_path,map_location=device,weights_only=False))
    model.superglue.load_state_dict(torch.load(module.p.superglue_model_path,map_location=device,weights_only=False))
    return model


def extract(pairs,report_out,*,device_name,match_threshold=.3,matcher_factory=None,production=True):
    """Drop-in direct-affine extractor signature; model built ONCE per pair.

    production=False only enables tiny square fixtures; production requires
    original512square canvases. Factory injection avoids native model jobs in tests.
    """
    report_out=Path(report_out)
    if (not pairs or report_out.exists() or any(Path(row[3]).exists() for row in pairs)
            or len({Path(row[3]).resolve() for row in pairs})!=len(pairs)
            or report_out.resolve() in {Path(row[3]).resolve() for row in pairs}):
        raise ValueError("nonempty pairs with distinct new output files required")
    if match_threshold!=.3 or isinstance(match_threshold,bool) or not isinstance(production,bool):
        raise ValueError("fixed match threshold.3 and bool production required")
    device=torch.device(device_name)
    factory=_installed_matcher if matcher_factory is None else matcher_factory
    config=dict(superpoint=dict(nms_radius=4,keypoint_threshold=.005,max_keypoints=3000),
        superglue=dict(weights="outdoor",sinkhorn_iterations=30,match_threshold=.3))
    rows=[];model_seconds_total=0.
    for name,fixed_path,moving_path,output in pairs:
        fixed_path,moving_path,output=Path(fixed_path),Path(moving_path),Path(output)
        with Image.open(fixed_path) as image:shape=image.size
        with Image.open(moving_path) as image:moving_shape=image.size
        if shape!=moving_shape or shape[0]!=shape[1] or shape[0]<16 or (production and shape!=(512,512)):
            raise ValueError("matching original square512 canvases required in production")
        width,height=shape
        started=time.perf_counter()
        fixed,_=_read_gray_thumbnail(fixed_path,width)
        moving,_=_read_gray_thumbnail(moving_path,width)
        fixed,moving=fixed.to(device),moving.to(device)
        model_start=time.perf_counter();model=factory(config,device)
        if device.type=="cuda":torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        model_seconds=time.perf_counter()-model_start;model_seconds_total+=model_seconds
        candidates=[]
        for k in range(4):
            tick=time.perf_counter();candidate=dict(quarter_turns=k,status="failed")
            try:
                with torch.no_grad():prediction=model({"image0":fixed,"image1":torch.rot90(moving,k,(-2,-1))})
                prediction={key:value[0].detach().cpu().numpy() for key,value in prediction.items()}
                if device.type=="cuda":torch.cuda.synchronize(device)
                p0=prediction["keypoints0"];p1=unrotate_points(prediction["keypoints1"],k,width=width,height=height)
                matches=prediction["matches0"]
                if (p0.ndim!=2 or p0.shape[-1]!=2 or not np.isfinite(p0).all()
                        or not np.issubdtype(matches.dtype,np.integer) or matches.shape!=(len(p0),)
                        or (matches < -1).any() or (matches>=len(p1)).any()):
                    raise ValueError("finite keypoints and valid original-frame match indices required")
                match=select_ransac_matches(p0,p1,matches,width=width,height=height)
                candidate.update({key:match[key] for key in ("status","raw_matches","ransac_inliers","occupied_quadrants_4x4")},
                    fixed_keypoints=len(p0),moving_keypoints=len(p1))
                if match["status"]=="ok":
                    source=np.asarray(match["source_points_unit"],dtype=np.float64)
                    target=np.asarray(match["target_points_unit"],dtype=np.float64)
                    matrix64,offset64=similarity_from_inlier_units(source,target)
                    with np.errstate(over="ignore"):
                        matrix,offset=matrix64.astype(np.float32),offset64.astype(np.float32)
                    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():raise ValueError("nonfinite rounded similarity")
                    a,b,c,d=(Fraction.from_float(float(value)) for value in matrix.flat)
                    if a*d-b*c<=0:raise ValueError("rounded similarity loses positive exact determinant")
                    residual=source@matrix64.T+offset64-target
                    rmse=float(np.sqrt(np.mean(np.sum((residual*width)**2,axis=1))))
                    if not np.isfinite(rmse):raise ValueError("nonfinite similarity fit")
                    candidate.update(matrix=matrix.tolist(),offset=offset.tolist(),inlier_fit_rmse_512px=rmse,
                        fit_unit="native canvas pixels;512 in production",exact_rounded_affine_positive=True)
            except (ValueError,RuntimeError,KeyError) as error:
                candidate.update(status="failed",error=f"{type(error).__name__}: {error}")
            candidate["inference_unrotate_ransac_fit_seconds"]=time.perf_counter()-tick
            candidates.append(candidate)
        chosen=select_candidate(candidates)
        row=dict(name=name,fixed=str(fixed_path),moving=str(moving_path),output=str(output),
            status="no_eligible_candidate" if chosen is None else "ok",candidates=candidates,
            selected_quarter_turns=None if chosen is None else chosen["quarter_turns"],
            image_side=width,model_build_and_load_seconds=model_seconds,
            decode_match_ransac_fit_seconds=time.perf_counter()-started,
            initializer="quarter_turns",map_direction="fixed_canvas_to_original_moving_canvas",
            keypoint_unrotation_before_ransac=True,matcher_calls=4)
        if chosen is not None:
            matrix=np.asarray(chosen["matrix"],dtype=np.float32);offset=np.asarray(chosen["offset"],dtype=np.float32)
            output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output,post_affine_matrix=matrix,post_affine_offset=offset,
                affine_source="direct_image_superglue",initializer="quarter_turns",selected_quarter_turns=chosen["quarter_turns"])
            row.update({key:chosen[key] for key in ("matrix","offset","raw_matches","ransac_inliers",
                "occupied_quadrants_4x4","inlier_fit_rmse_512px")})
        rows.append(row)
    report=dict(method="four torch.rot90 moving views; original-frame RANSAC6px/min8; positive similarity",
        initializer="quarter_turns",selection="(-inliers,-fixed4x4support,positive-similarity fitRMSE,k); no downstream objective",
        match_threshold=.3,model_build_and_load_seconds=model_seconds_total,rows=rows,device=device_name,
        timing_scope="decode_match_ransac_fit_seconds includes model setup; per-view and model clocks are nested, not additive; model once per pair",
        landmarks_initial_DHR_affine_full_DHR_field_loaded=False,
        orientation_preserving_only_when_status_ok=True,
        peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
    report_out.parent.mkdir(parents=True,exist_ok=True)
    report_out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
