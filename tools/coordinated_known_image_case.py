"""Known-map IMAGE recovery with the shared image-only instance optimizer.

Research card: Can the shared image objective recover an independently legal
warp of real histology texture when oracle correspondence capacity is known?
The target is an analytic shear/rotation/coarse-fine map, never a solver field.
Assume fixed boundary, Q1 generating truth, an explicitly declared Q1/P1
estimated function, and a single original moving raster.
Failure of image-selected output to recover held-out queries falsifies recovery
under this objective/budget, NOT expressivity or the possibility of correspondence.
Smallest tests: identity pixels, affine off-grid queries, actual target corners.
Prior work reused: existing analytic capacity targets and shared real-case optimizer.
"""
from __future__ import annotations

import argparse
import copy
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


def generate_fixed(moving, target, interpolation="q1"):
    if interpolation=="q1":
        query = q1_map_at_pixel_centers(target, *moving.shape[-2:])
    elif interpolation in ("p1_ac","p1_bd"):
        from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_pixel_centers
        query=p1_map_at_pixel_centers(target,*moving.shape[-2:],diagonal=interpolation[-2:])
    else:
        raise ValueError("declare q1, p1_ac or p1_bd")
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


def sample_declared_queries(vertices,points,interpolation="q1"):
    """Independent explicit barycentric evaluation for the declared estimate."""
    if interpolation=="q1":return sample_q1_queries(vertices,points)
    if interpolation not in ("p1_ac","p1_bd"):
        raise ValueError("declare q1, p1_ac or p1_bd")
    if vertices.shape[0]!=1 or points.ndim!=2 or points.shape[1]!=2:
        raise ValueError("one map and N by2 source queries required")
    if not bool(torch.isfinite(points).all() and ((points>=0)&(points<=1)).all()):
        raise ValueError("finite in-domain source queries required")
    rows,columns=vertices.shape[1:3]
    x,y=points[:,0]*(columns-1),points[:,1]*(rows-1)
    c,r=x.floor().long().clamp_max(columns-2),y.floor().long().clamp_max(rows-2)
    u,v=(x-c)[:,None],(y-r)[:,None]
    a,b,d,ne=vertices[0,r,c],vertices[0,r,c+1],vertices[0,r+1,c],vertices[0,r+1,c+1]
    if interpolation=="p1_ac":
        return torch.where(u>=v,(1-u)*a+(u-v)*b+v*ne,(1-v)*a+u*ne+(v-u)*d)
    return torch.where(u+v<=1,(1-u-v)*a+u*b+v*d,(1-v)*b+(u+v-1)*ne+(1-u)*d)


def held_out_map_metrics(estimate, target, image_side, count=4096, *, estimate_interpolation="q1"):
    rng = np.random.default_rng(20261001)
    points = torch.from_numpy(rng.uniform(.00001,.99999,(count,2))).to(target)
    delta = sample_declared_queries(estimate.to(target),points,estimate_interpolation)-sample_q1_queries(target,points)
    errors = delta.square().sum(-1).sqrt()
    rmse = float(errors.square().mean().sqrt())
    return dict(query_count=count,query_seed=20261001,queries_used_for_optimization=False,
                estimate_interpolation=estimate_interpolation,target_interpolation="q1",
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


def image_metrics(fixed,moving,vertices,interpolation="q1"):
    warped=generate_fixed(moving,vertices,interpolation)
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


def objective_call_count(report):
    if "objective_evaluations" in report:
        return report["objective_evaluations"]
    if "objective_evaluations_total" in report:
        return report["objective_evaluations_total"]
    # Older single-resolution reports lack the per-stage full-resolution call.
    stages=report["stages"]
    return report["evaluations"]+len(stages)+sum("accepted_full_total" in s for s in stages)+2


def refresh_objective_counts(path):
    payload=json.loads(path.read_text(encoding="utf-8"))
    for row in payload["results"]:
        method_report=json.loads((path.parent/Path(row["output_map"]).with_suffix(".json")).read_text(encoding="utf-8"))
        row["objective_evaluations_including_stage_anchors"]=objective_call_count(method_report)
        row["objective_call_count_source"]="actual method report: trials + anchors + recorded full-resolution stage evaluations + initial/final"
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


def native_dhr_query_metrics(field,params,target,image_side,count=4096):
    """Read-only native field evaluation; never called by image optimization."""
    from tools.digital_compare_appearance import dhr_map_at_unit_queries
    points=np.random.default_rng(20261001).uniform(.00001,.99999,(count,2))
    query=torch.from_numpy(points).float().reshape(1,1,count,2)
    predicted=dhr_map_at_unit_queries(field,params,fixed_size=(image_side,image_side),
        moving_size=(image_side,image_side),query=query)[0,0].double()
    expected=sample_q1_queries(target.double(),torch.from_numpy(points))
    errors=(predicted-expected).square().sum(-1).sqrt()
    return dict(query_count=count,query_seed=20261001,queries_used_for_optimization=False,
        euclidean_query_rmse_normalized=float(errors.square().mean().sqrt()),
        euclidean_query_rmse_canvas_pixels=float(errors.square().mean().sqrt())*image_side,
        p90_query_error_canvas_pixels=float(torch.quantile(errors,.9))*image_side,
        coordinate_component_rmse_normalized=float((predicted-expected).square().mean().sqrt()),
        out_of_unit_square_predictions=int(((predicted<0)|(predicted>1)).any(-1).sum()),
        denominator=count,queries_dropped=0,native_sampling_dtype="float32",
        maximum_source_query_float32_rounding_canvas_pixels=float(
            np.abs(query.numpy().reshape(count,2).astype(np.float64)-points).max())*image_side)


def score_native_dhr(args):
    import SimpleITK as sitk
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inputs_from is None:
        raise ValueError("native scoring requires the existing known-image input manifest")
    manifest=json.loads(args.inputs_from.read_text(encoding="utf-8"))
    if manifest["image_side"]!=args.image_side:
        raise ValueError("native scoring image units differ from prepared canvas")
    native_root=args.native_dhr_root or args.output.parent
    rows=[]
    for case in manifest["cases"]:
        if case["target"] not in args.targets:
            continue
        native=native_root/("known_dhr_"+case["target"])
        field_path=native/"common_affine_dhr"/"Results_Final"/"displacement_field.mha"
        params_path=field_path.with_name("postprocessing_params.json")
        params=json.loads(params_path.read_text(encoding="utf-8"))
        runtime=json.loads((native/"runtime.json").read_text(encoding="utf-8"))
        config=json.loads((native/"config.json").read_text(encoding="utf-8"))
        if not (np.array_equal(runtime["post_affine_matrix"],np.eye(2)) and
                np.array_equal(runtime["post_affine_offset"],np.zeros(2))):
            raise ValueError("native comparison did not use the declared identity affine")
        field=sitk.GetArrayFromImage(sitk.ReadImage(str(field_path)))
        with np.load(args.inputs_from.parent/case["target_archive"]) as archive:
            target=torch.from_numpy(archive["vertices"])
        row=dict(target=case["target"],held_out=native_dhr_query_metrics(field,params,target,args.image_side),
            common_identity_verified=True,native_field=str(field_path),native_field_shape=list(field.shape),
            runtime_seconds=runtime["runtime_seconds"],nonrigid_seconds=runtime["nonrigid_seconds"],
            preprocessing_seconds=runtime["preprocessing_seconds"],postprocessing=params,
            native_cost=config["nonrigid_registration_params"]["cost_function"],
            native_cost_parameters=config["nonrigid_registration_params"]["cost_function_params"],
            native_regularizer=config["nonrigid_registration_params"]["regularization_function"],
            clahe=config["preprocessing_params"]["clahe"],boundary="native free boundary; not shared fixed-boundary feasible class",
            hard_topology_certificate=None,native_field_used_by_shared_optimizer=False)
        rows.append(row)
        print(json.dumps(row),flush=True)
    payload=dict(question="Posthoc native DHR comparison on the SAME known histology image pairs",
        input_manifest=str(args.inputs_from),target_interpolation="Q1",map_direction="fixed -> moving",
        query_protocol="seed20261001,4096 uniform normalized queries; no dropping or query-based selection",
        differences="native CLAHE/NCC/diffusion-relative regularization and free boundary; no hard topology certificate; different optimization budgets",
        results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload


def annotate_ncc_truth_floor(path):
    """Posthoc only: isolate PNG error from the stabilized local-NCC floor."""
    from tools.coordinated_real_case import Evidence
    from tools.digital_q1_real_optimize import _read_gray_thumbnail
    payload=json.loads(path.read_text(encoding="utf-8"))
    if payload["configuration"]["loss"]!="local_ncc":
        raise ValueError("local-NCC floor annotation needs the local-NCC experiment")
    manifest=payload["inputs"]
    side=manifest["image_side"]
    moving,_=_read_gray_thumbnail(path.parent/manifest["moving"],side)
    moving=moving.double()
    rows=[]
    for case in manifest["cases"]:
        fixed,_=_read_gray_thumbnail(path.parent/case["fixed"],side)
        fixed=fixed.double()
        with np.load(path.parent/case["target_archive"]) as archive:
            target=torch.from_numpy(archive["vertices"])
            perfect=torch.from_numpy(archive["generated_fixed_float64"])
        evidence=lambda f,m:Evidence(f,m,torch.eye(2,dtype=torch.float64),
            torch.zeros(2,dtype=torch.float64),"local_ncc",0.,0.)
        rows.append(dict(target=case["target"],
            quantized_true_warp_ncc_image=float(evidence(fixed,moving)(target)[1]["image"]),
            exact_float_true_warp_ncc_image=float(evidence(perfect,moving)(target)[1]["image"]),
            identical_fixed_self_ncc_image=float(evidence(fixed,fixed)(reference_grid(manifest["grid_side"]))[1]["image"])))
    payload["local_ncc_truth_floor_diagnostic"]=dict(posthoc_only=True,optimizer_changed=False,
        question="Is nonzero true-warp NCC loss caused by quantization or the intrinsic stabilized low-texture floor?",
        stabilizer=1e-5,results=rows)
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload["local_ncc_truth_floor_diagnostic"]


def _validate_capture_prefix(args):
    """Reject unsupported matrices before preparation or any optimizer call."""
    mode=getattr(args,"capture_prefix","none")
    if mode not in ("none","mind_discrete"):
        raise ValueError("capture_prefix must be none or mind_discrete")
    if mode=="mind_discrete":
        image_levels=args.image_levels if args.image_levels is not None else [args.image_side]
        if (not args.methods or any(method!="analytic" for method in args.methods)
                or args.grid_side!=257 or getattr(args,"interpolation","q1")!="p1_ac"
                or getattr(args,"coordinate_mode","alternating")!="alternating"
                or args.loss!="mind" or getattr(args,"mind_order","transport")!="transport"
                or 33 not in args.levels or 128 not in image_levels or args.image_side<128):
            raise ValueError("known-image capture requires analytic fixed257 P1-ac alternating controls, transport MIND, and 33 coefficient/128 image level")
    return mode


def run(args,manifest):
    capture_prefix=_validate_capture_prefix(args)
    from tools.coordinated_real_case import Evidence,optimize,corner_symmetric_dirichlet,load_image_matches,load_registration_evidence
    from tools.digital_q1_real_optimize import _read_gray_thumbnail
    if args.output.exists():
        raise FileExistsError(args.output)
    rows=[]
    interpolation=getattr(args,"interpolation","q1")
    strain_model=getattr(args,"strain_model","displacement_gradient")
    mind_order=getattr(args,"mind_order","transport")
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
        evidence_fixed,evidence_moving,evidence_mask,preprocessing_metadata=load_registration_evidence(
            folder/case["fixed"],folder/manifest["moving"],args.image_side,
            preprocessing=getattr(args,"preprocessing","raw_inverted"),dtype=torch.float64)
        matches,match_path=None,None
        if getattr(args,"match_weight",0.):
            if getattr(args,"matches_dir",None) is None:
                raise ValueError("declare --matches-dir for a positive match weight")
            match_path=args.matches_dir/("known_"+case["target"]+"_sg_raw_matches.json")
            matches,_=load_image_matches(match_path,np.eye(2,dtype=np.float32),np.zeros(2,dtype=np.float32),
                fixed_path=folder/case["fixed"],moving_path=folder/manifest["moving"],image_side=args.image_side,
                device=torch.device("cpu"),dtype=torch.float64,robust_scale=args.match_robust_scale)
        evidence=Evidence(evidence_fixed,evidence_moving,torch.eye(2,dtype=torch.float64),
                          torch.zeros(2,dtype=torch.float64),args.loss,args.strain_weight,1.,
                          shape_weight=args.shape_weight,fixed_mask=evidence_mask,matches=matches,match_weight=getattr(args,"match_weight",0.),
                          interpolation=interpolation if strain_model=="p1_arap" else "q1",strain_model=strain_model,mind_order=mind_order)
        truth_total,truth_parts=evidence(target)
        declared_truth_evidence=copy.copy(evidence)
        declared_truth_evidence.interpolation=interpolation
        declared_truth_total,declared_truth_parts=declared_truth_evidence(target)
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
                precision="float64",image_precision="float64",shape_weight=args.shape_weight,
                image_levels=args.image_levels,device=args.device,threads=args.threads)
            opt.preprocessing=getattr(args,"preprocessing","raw_inverted")
            opt.interpolation=interpolation
            opt.p1_sampling=getattr(args,"p1_sampling","existing")
            opt.coordinate_mode=getattr(args,"coordinate_mode","alternating")
            opt.joint_backend=getattr(args,"joint_backend","cached_manual")
            opt.geometry_backend=getattr(args,"geometry_backend","existing")
            opt.output_selection=getattr(args,"output_selection","last")
            opt.capture_prefix=capture_prefix
            opt.strain_model=strain_model
            opt.mind_order=mind_order
            opt.matches=match_path;opt.match_weight=getattr(args,"match_weight",0.)
            opt.match_robust_scale=getattr(args,"match_robust_scale",8.)
            snapshots=[]
            def observe_stage(vertices,stage,elapsed):
                # No truth/queries are consulted here. Image-only acceptance
                # already occurred in the shared optimizer; copy its observer clone.
                snapshots.append((vertices.cpu(),stage,elapsed))
            report=(optimize(opt,accepted_stage_callback=observe_stage)
                    if args.record_stages else optimize(opt))
            query_history=[dict(stage=stage,optimizer_seconds=elapsed,
                                held_out=held_out_map_metrics(vertices,target,args.image_side,estimate_interpolation=interpolation),
                                actual_minimum_normalized_corner=generic_corner_ratio(vertices,reference))
                           for vertices,stage,elapsed in snapshots]
            initial_query=held_out_map_metrics(reference,target,args.image_side,estimate_interpolation=interpolation)
            times={str(threshold):(0. if initial_query["euclidean_query_rmse_canvas_pixels"]<=threshold
                   else next((item["optimizer_seconds"] for item in query_history
                              if item["held_out"]["euclidean_query_rmse_canvas_pixels"]<=threshold),None))
                   for threshold in args.query_thresholds}
            with np.load(output) as data:
                estimated=torch.from_numpy(data["vertices"])
            row=dict(target=case["target"],method=method,target_specification=case,
                held_out=held_out_map_metrics(estimated,target,args.image_side,estimate_interpolation=interpolation),
                initial_held_out=initial_query,
                image_error_initial=initial,image_error_final=image_metrics(fixed,moving,estimated,interpolation),
                estimate_interpolation=interpolation,target_interpolation="q1",
                target_vertex_interpolation_discrepancy=held_out_map_metrics(target,target,args.image_side,estimate_interpolation=interpolation),
                target_objective_interpolation=("q1 (original generating function, not P1 optimum)" if strain_model=="displacement_gradient"
                    else interpolation+" vertex-table comparator, NOT original Q1 generating function"),
                target_objective_total=float(truth_total),target_objective_parts={k:float(v) for k,v in truth_parts.items()},
                target_declared_objective_total=float(declared_truth_total),
                target_declared_objective_parts={k:float(v) for k,v in declared_truth_parts.items()},
                target_declared_objective_interpolation=interpolation,strain_model=strain_model,
                mind_order=mind_order if args.loss=="mind" else None,
                target_corner_shape=float(corner_symmetric_dirichlet(target)),
                final_corner_shape=float(corner_symmetric_dirichlet(estimated)),
                target_raster_floor=image_metrics(fixed,moving,target),
                target_raster_floor_scope="residual at original Q1 generating map after PNG quantization, not a proved attainable-error lower bound",
                actual_minimum_normalized_corner=generic_corner_ratio(estimated,reference),
                optimize_seconds=report["optimize_seconds"],gradient_steps=report["gradient_steps"],
                decoder_trial_evaluations=report["evaluations"],failed_trials=report["failed_trials"],
                coordinate_mode=opt.coordinate_mode,
                coordinated_substep_evaluations=report.get("coordinated_substep_evaluations"),
                extra_joint_diagnostic_passes=report.get("extra_joint_diagnostic_passes"),
                capture_prefix=report.get("capture_prefix",capture_prefix),
                capture_record=report.get("capture_record"),
                capture_objective_evaluations=report.get("capture_objective_evaluations",0),
                capture_geometry_attempts=(report.get("capture_record") or {}).get("geometry_attempts",0),
                capture_search_seconds=(report.get("capture_record") or {}).get("search_seconds",0.),
                capture_complete_prefix_seconds=(report.get("capture_record") or {}).get("complete_prefix_seconds",0.),
                objective_evaluations_including_stage_anchors=objective_call_count(report),
                median_forward_objective_seconds=report["median_forward_objective_seconds"],
                median_vjp_seconds=report["median_vjp_seconds"],
                peak_allocated_bytes=report["peak_allocated_bytes"],
                initial_objective=report["initial"],final_objective=report["final"],
                accepted_full_resolution_totals=[stage.get("accepted_full_total") for stage in report["stages"]],
                accepted_stage_query_history=query_history,
                posthoc_time_to_query_rmse_canvas_pixels=times if args.record_stages else None,
                stage_snapshot_cpu_bytes=sum(v.numel()*v.element_size() for v,_,_ in snapshots),
                observer_overhead_included_in_optimize_time=bool(args.record_stages),
                saved_binary_certificate=report["saved_binary_certificate"],output_map=output.name,
                landmarks_or_target_used_by_optimizer=False)
            row["image_preprocessing"]=preprocessing_metadata
            rows.append(row)
            print(json.dumps(row),flush=True)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration["capture_prefix"]=capture_prefix
    payload=dict(question=__doc__,configuration=configuration,
                 inputs=manifest,results=rows,selection="shared complete stage-resolution image + cumulative strain + OOB objective; held-out map metrics computed after optimization",
                 timing_scope="shared optimize time excludes input generation/loading and final exact sign certificate; CPU scoring excluded; enabled capture search/safe attempts and extra objective calls are included, separately reported beyond the Adam budget",
                 caution="regularization can bias away from truth; full-resolution objective need not decrease across pyramid levels; capture failure does not establish impossible correspondence")
    args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moving",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs-from",type=Path,help="reuse exactly the previously prepared input manifest and rasters")
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--score-native-dhr",action="store_true",help="separate read-only posthoc mode, no image optimization")
    p.add_argument("--native-dhr-root",type=Path)
    p.add_argument("--record-stages",action="store_true",
                   help="target-free accepted-map snapshots; query evaluation only after optimization; observer overhead included")
    p.add_argument("--query-thresholds",type=float,nargs="+",default=[1.,5.,10.],
                   help="posthoc accepted-stage query RMSE thresholds in canvas pixels, never iterate selection")
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
    p.add_argument("--strain-model",choices=("displacement_gradient","p1_arap"),default="displacement_gradient")
    p.add_argument("--shape-weight",type=float,default=0.,
                   help="same common corner symmetric-Dirichlet weight for truth evaluation and all optimizers")
    p.add_argument("--matches-dir",type=Path)
    p.add_argument("--match-weight",type=float,default=0.)
    p.add_argument("--match-robust-scale",type=float,default=8.)
    p.add_argument("--minimum-jacobian",type=float,default=.001)
    p.add_argument("--loss",choices=("mind","local_ncc"),default="mind")
    p.add_argument("--mind-order",choices=("transport","after_warp"),default="transport")
    p.add_argument("--preprocessing",choices=("raw_inverted","native_dhr"),default="raw_inverted")
    p.add_argument("--interpolation",choices=("q1","p1_ac","p1_bd"),default="q1",
                   help="Estimated function only; prepared known-image truth remains original Q1")
    p.add_argument("--p1-sampling",choices=("existing","frozen"),default="existing")
    p.add_argument("--coordinate-mode",choices=("alternating","joint"),default="alternating")
    p.add_argument("--joint-backend",choices=("ordinary","cached_manual"),default="cached_manual")
    p.add_argument("--geometry-backend",choices=("existing","stage_cache"),default="existing")
    p.add_argument("--output-selection",choices=("last","best_full"),default="last")
    p.add_argument("--capture-prefix",choices=("none","mind_discrete"),default="none",
                   help="optional image-only discrete initializer; analytic fixed257 P1-ac only; extra prefix calls/time reported")
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    try:
        _validate_capture_prefix(args)
    except ValueError as error:
        p.error(str(error))
    if args.coordinate_mode=="joint" and (args.geometry_backend!="existing" or
            any(method not in ("radial","analytic") for method in args.methods)):
        p.error("joint known-image comparison needs only radial/analytic and geometry_backend existing")
    if not 0 < args.target_scale <= 1:
        p.error("target scale must lie in (0,1]")
    if not np.isfinite(args.shape_weight) or args.shape_weight<0:
        p.error("shape weight must be finite and nonnegative")
    torch.set_num_threads(args.threads)
    if args.score_native_dhr:
        score_native_dhr(args)
        return
    manifest=prepare(args)
    if args.prepare_only:
        print(json.dumps(manifest),flush=True)
    else:
        run(args,manifest)


if __name__=="__main__":
    main()
