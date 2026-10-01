"""Known-map IMAGE recovery with the shared image-only instance optimizer.

Research card: Can the shared image objective recover an independently legal
warp of real histology texture when oracle correspondence capacity is known?
The target is an analytic shear/rotation/coarse-fine map, never a solver field.
Assume fixed boundary, Q1 interpolation, and a single original moving raster.
Failure of image-selected output to recover held-out queries falsifies recovery
under this objective/budget, NOT expressivity or the possibility of correspondence.
Smallest tests: identity pixels, affine off-grid queries, actual target corners.
Prior work reused: existing analytic capacity targets and shared real-case optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from tools.coordinated_geometry_pilot import reference_grid, target_map
from qcopt.neural_bijection.dense.q1_image_sampling import q1_map_at_pixel_centers


def generic_corner_ratio(vertices, reference):
    """Independent generic oriented triangles, all four corners, actual values."""
    def triangles(v):
        v = v.detach().cpu().numpy().astype(np.float64)
        a,b,c,d = v[:,:-1,:-1],v[:,:-1,1:],v[:,1:,1:],v[:,1:,:-1]
        def det(p,q,r):
            e,f = q-p,r-p
            return e[...,0]*f[...,1]-e[...,1]*f[...,0]
        return np.stack([det(*t) for t in ((a,b,d),(a,b,c),(d,b,c),(a,c,d))],-1)
    return float((triangles(vertices)/triangles(reference)).min())


def generate_fixed(moving, target):
    query = q1_map_at_pixel_centers(target, *moving.shape[-2:])
    return F.grid_sample(moving, 2*query-1, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def sample_q1_queries(vertices, points):
    """Independent direct bilinear evaluation, not the optimizer's sampler."""
    if vertices.shape[0] != 1 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("one map and N by 2 normalized queries required")
    if not bool(((points>=0)&(points<=1)).all()):
        raise ValueError("held-out source queries must lie in the unit rectangle")
    rows,columns = vertices.shape[1:3]
    x,y = points[:,0]*(columns-1),points[:,1]*(rows-1)
    c,r = x.floor().long().clamp_max(columns-2),y.floor().long().clamp_max(rows-2)
    u,v = (x-c)[:,None],(y-r)[:,None]
    return ((1-u)*(1-v)*vertices[0,r,c]+u*(1-v)*vertices[0,r,c+1]
            +u*v*vertices[0,r+1,c+1]+(1-u)*v*vertices[0,r+1,c])


def held_out_map_metrics(estimate, target, image_side, count=4096):
    rng = np.random.default_rng(20261001)
    points = torch.from_numpy(rng.uniform(.00001,.99999,(count,2))).to(target)
    delta = sample_q1_queries(estimate.to(target),points)-sample_q1_queries(target,points)
    errors = delta.square().sum(-1).sqrt()
    rmse = float(errors.square().mean().sqrt())
    return dict(query_count=count,query_seed=20261001,queries_used_for_optimization=False,
                euclidean_query_rmse_normalized=rmse,
                euclidean_query_rmse_canvas_pixels=rmse*image_side,
                p90_query_error_canvas_pixels=float(torch.quantile(errors,.9))*image_side,
                coordinate_component_rmse_normalized=float(delta.square().mean().sqrt()))


def _save_inverted_png(tensor, path):
    array = np.rint(255*(1-tensor[0,0].detach().cpu().numpy())).clip(0,255).astype(np.uint8)
    Image.fromarray(array).save(path)


def prepare(args):
    from tools.digital_q1_real_optimize import _read_gray_thumbnail
    prefix = args.output.with_suffix("")
    manifest_path = args.inputs_from or prefix.with_name(prefix.name+"_inputs.json")
    if args.inputs_from is not None and not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["grid_side"],manifest["image_side"],manifest["target_scale"]) != (
                args.grid_side,args.image_side,args.target_scale):
            raise ValueError("existing prepared inputs have a different geometry/scale")
        return manifest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    moving,_ = _read_gray_thumbnail(args.moving,args.image_side)
    moving = moving.double()
    moving_path = prefix.with_name(prefix.name+"_moving.png")
    affine_path = prefix.with_name(prefix.name+"_identity.npz")
    if moving_path.exists() or affine_path.exists():
        raise FileExistsError("partial prepared inputs exist; use a fresh output prefix")
    _save_inverted_png(moving,moving_path)
    # Reopen the saved raster through exactly the shared interface before generation.
    moving,_ = _read_gray_thumbnail(moving_path,args.image_side)
    moving = moving.double()
    np.savez(affine_path,post_affine_matrix=np.eye(2,dtype=np.float32),
             post_affine_offset=np.zeros(2,dtype=np.float32))
    reference=reference_grid(args.grid_side)
    cases=[]
    for name in args.targets:
        target=reference+args.target_scale*(target_map(reference,name)-reference)
        minimum=generic_corner_ratio(target,reference)
        edges_equal=(torch.equal(target[:,0],reference[:,0]) and
                     torch.equal(target[:,-1],reference[:,-1]) and
                     torch.equal(target[:,:,0],reference[:,:,0]) and
                     torch.equal(target[:,:,-1],reference[:,:,-1]))
        if minimum<=args.minimum_jacobian or not edges_equal:
            raise ValueError("independent target fails declared corner/boundary conditions")
        fixed_float=generate_fixed(moving,target)
        fixed_path=prefix.with_name(prefix.name+"_"+name+"_fixed.png")
        target_path=prefix.with_name(prefix.name+"_"+name+"_target.npz")
        _save_inverted_png(fixed_float,fixed_path)
        fixed,_ = _read_gray_thumbnail(fixed_path,args.image_side)
        fixed=fixed.double()
        np.savez(target_path,vertices=target.numpy(),boundary_reference=reference.numpy(),
                 generated_fixed_float64=fixed_float.numpy())
        displacement=(target-reference).square().sum(-1).sqrt()
        cases.append(dict(target=name,fixed=fixed_path.name,target_archive=target_path.name,
                          target_minimum_normalized_corner=minimum,exact_boundary=True,
                          maximum_target_displacement_canvas_pixels=float(displacement.max())*args.image_side,
                          rms_target_displacement_canvas_pixels=float(displacement.square().mean().sqrt())*args.image_side,
                          fixed_png_quantization_rmse=float((fixed-fixed_float).square().mean().sqrt())))
    manifest=dict(grid_side=args.grid_side,image_side=args.image_side,target_scale=args.target_scale,
                  moving=moving_path.name,identity_affine=affine_path.name,cases=cases,
                  source_moving=str(args.moving),map_direction="fixed source -> moving query",
                  interpolation="Q1; image centers align_corners=False; map nodes align_corners=True",
                  optimizer_inputs="fixed/moving PNG and identity affine only; no target archive",
                  physical_units="normalized unit rectangle and canvas pixels; specimen microns unavailable")
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def image_metrics(fixed,moving,vertices):
    warped=generate_fixed(moving,vertices)
    mask=(fixed>.04).to(fixed)
    difference=(fixed-warped).square()
    return dict(raster_rmse_all_pixels=float(difference.mean().sqrt()),
                raster_rmse_fixed_tissue=float(((difference*mask).sum()/mask.sum()).sqrt()),
                fixed_tissue_pixel_denominator=float(mask.sum()))


def persist_finite_amplitude_attenuation(path, sides=(33,65,129,257)):
    """Append the measured finite-step diagnostic to an existing capacity report."""
    from tools.coordinated_control_capacity import make_layer,decode,f2_coverage
    payload=json.loads(path.read_text(encoding="utf-8"))
    if "finite_amplitude_attenuation" in payload:
        raise ValueError("finite-amplitude diagnostic already recorded")
    u=torch.linspace(0,1,9,dtype=torch.float64)
    field=.02*torch.sin(torch.pi*u)[:,None].square()*torch.sin(torch.pi*u)[None,:].square()
    vector=torch.stack((field[1:-1,1:-1],torch.zeros_like(field[1:-1,1:-1])),0)[None]
    rows=[]
    for side in sides:
        reference=reference_grid(side)
        intended=F.interpolate(F.pad(vector,(1,1,1,1)),size=(side,side),
                               mode="bilinear",align_corners=True).permute(0,2,3,1)
        for method in ("radial","analytic","f1","f2"):
            layer=make_layer(method,side,(1.,0.),.001,8,.75)
            coverage=f2_coverage(layer,side,torch.float64) if method=="f2" else None
            coefficients=vector[:,:1] if method in ("radial","analytic") else vector
            if method=="f2":
                coefficients=coefficients/.75
            candidate=decode(layer,method,reference,coefficients,coverage)
            actual=candidate-reference
            rows.append(dict(side=side,method=method,
                actual_to_intended_displacement_rms=float(actual.square().mean().sqrt()/intended.square().mean().sqrt()),
                actual_peak_x_displacement=float(actual[...,0].max()),
                minimum_normalized_corner=generic_corner_ratio(candidate,reference)))
    payload["finite_amplitude_attenuation"]=dict(
        question="Does identical infinitesimal physical calibration imply identical finite broad-stage transfer?",
        coefficient_level=9,intended_peak_displacement_normalized=.02,
        anchor="identity",dtype="float64",f2_accepted_gain=.75,f2_safety_fraction=.85,
        optimizer_used=False,results=rows,
        conclusion="Finite-amplitude transfer depends on grid size for the existing local controls; not a whole-method capacity bound")
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload["finite_amplitude_attenuation"]


def run(args,manifest):
    from tools.coordinated_real_case import Evidence,optimize
    from tools.digital_q1_real_optimize import _read_gray_thumbnail
    if args.output.exists():
        raise FileExistsError(args.output)
    rows=[]
    folder=args.inputs_from.parent if args.inputs_from is not None else args.output.parent
    fixed_moving,_ = _read_gray_thumbnail(folder/manifest["moving"],args.image_side)
    moving=fixed_moving.double()
    reference=reference_grid(args.grid_side)
    available={case["target"] for case in manifest["cases"]}
    if not set(args.targets)<=available:
        raise ValueError("requested target absent from prepared image inputs")
    for case in manifest["cases"]:
        if case["target"] not in args.targets:
            continue
        fixed,_ = _read_gray_thumbnail(folder/case["fixed"],args.image_side)
        fixed=fixed.double()
        with np.load(folder/case["target_archive"]) as data:
            target=torch.from_numpy(data["vertices"])
        evidence=Evidence(fixed,moving,torch.eye(2,dtype=torch.float64),
                          torch.zeros(2,dtype=torch.float64),args.loss,args.strain_weight,1.)
        truth_total,truth_parts=evidence(target)
        initial=image_metrics(fixed,moving,reference)
        for method in args.methods:
            output=args.output.with_name(args.output.stem+"_"+case["target"]+"_"+method+".npz")
            # Two-component controls use twice as many cycles to match gradient
            # and decoder-trial budgets, not scalar parameters/geometry passes.
            opt=SimpleNamespace(fixed=folder/case["fixed"],moving=folder/manifest["moving"],
                affine=folder/manifest["identity_affine"],output=output,method=method,
                patch_cells=8,f2_accepted_gain=.75,regional_cells=32,regional_min_level=3,
                loss=args.loss,grid_side=args.grid_side,image_side=args.image_side,
                levels=args.levels,inner_steps=args.inner_steps,
                cycles=args.cycles*(2 if method in ("f1","f2") else 1),
                learning_rate=args.learning_rate,lr_calibration="edge",
                strain_weight=args.strain_weight,oob_weight=1.,minimum_jacobian=args.minimum_jacobian,
                precision="float64",image_precision="float64",shape_weight=0.,
                image_levels=args.image_levels,device=args.device,threads=args.threads)
            report=optimize(opt)
            with np.load(output) as data:
                estimated=torch.from_numpy(data["vertices"])
            row=dict(target=case["target"],method=method,target_specification=case,
                held_out=held_out_map_metrics(estimated,target,args.image_side),
                initial_held_out=held_out_map_metrics(reference,target,args.image_side),
                image_error_initial=initial,image_error_final=image_metrics(fixed,moving,estimated),
                target_objective_total=float(truth_total),target_objective_parts={k:float(v) for k,v in truth_parts.items()},
                target_raster_floor=image_metrics(fixed,moving,target),
                actual_minimum_normalized_corner=generic_corner_ratio(estimated,reference),
                optimize_seconds=report["optimize_seconds"],gradient_steps=report["gradient_steps"],
                decoder_trial_evaluations=report["evaluations"],failed_trials=report["failed_trials"],
                objective_evaluations_including_stage_anchors=report["evaluations"]+len(report["stages"])+2,
                median_forward_objective_seconds=report["median_forward_objective_seconds"],
                median_vjp_seconds=report["median_vjp_seconds"],
                peak_allocated_bytes=report["peak_allocated_bytes"],
                initial_objective=report["initial"],final_objective=report["final"],
                accepted_full_resolution_totals=[stage.get("accepted_full_total") for stage in report["stages"]],
                saved_binary_certificate=report["saved_binary_certificate"],output_map=output.name,
                landmarks_or_target_used_by_optimizer=False)
            rows.append(row)
            print(json.dumps(row),flush=True)
    payload=dict(question=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 inputs=manifest,results=rows,selection="shared complete stage-resolution image + cumulative strain + OOB objective; held-out map metrics computed after optimization",
                 timing_scope="shared optimize time excludes input generation/loading and final exact sign certificate; CPU scoring excluded",
                 caution="regularization can bias away from truth; full-resolution objective need not decrease across pyramid levels; capture failure does not establish impossible correspondence")
    args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moving",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs-from",type=Path,help="reuse exactly the previously prepared input manifest and rasters")
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--targets",nargs="+",choices=("wide_shear","local_rotation","coarse_fine"),default=["wide_shear","local_rotation","coarse_fine"])
    p.add_argument("--methods",nargs="+",choices=("radial","analytic","f1","f2"),default=["radial","analytic","f1","f2"])
    p.add_argument("--target-scale",type=float,default=1.)
    p.add_argument("--grid-side",type=int,default=257)
    p.add_argument("--image-side",type=int,default=512)
    p.add_argument("--levels",type=int,nargs="+",default=[17,33,65,129,257])
    p.add_argument("--image-levels",type=int,nargs="+",default=None,
                   help="optional shared original-raster image pyramid, otherwise full image resolution")
    p.add_argument("--cycles",type=int,default=2)
    p.add_argument("--inner-steps",type=int,default=5)
    p.add_argument("--learning-rate",type=float,default=.004)
    p.add_argument("--strain-weight",type=float,default=.05)
    p.add_argument("--minimum-jacobian",type=float,default=.001)
    p.add_argument("--loss",choices=("mind","local_ncc"),default="mind")
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if not 0 < args.target_scale <= 1:
        p.error("target scale must lie in (0,1]")
    torch.set_num_threads(args.threads)
    manifest=prepare(args)
    if args.prepare_only:
        print(json.dumps(manifest),flush=True)
    else:
        run(args,manifest)


if __name__=="__main__":
    main()
