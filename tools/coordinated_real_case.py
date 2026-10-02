"""Shared-evidence, stage-anchored instance registration. No landmarks are read.

The saved residual is declared Q1 or P1 on the grid; a positive frozen affine is
postcomposed. Queries sample original moving evidence, not a recursively
resampled raster. Selection uses the complete image + cumulative-strain +
out-of-bounds objective, with a fixed foreground denominator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update, interpolate_proposal
from qcopt.neural_bijection.dense.coordinated_patches import CoordinatedPatchQ1Pass
from qcopt.neural_bijection.dense.digital_q1 import AdaptiveSoftRadialQ1Relaxation, StaggeredPatchQ1Layer, q1_corner_determinants, validate_q1_map
from qcopt.neural_bijection.dense.q1_image_sampling import q1_map_at_pixel_centers
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_pixel_centers
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.coordinated_control_capacity import f2_coverage, decode as decode_control


def corner_symmetric_dirichlet(vertices):
    """Four-corner quadrature of ||J||²+||J^-1||²-4, not exact Q1 integral."""
    a,b = vertices[:,:-1,:-1],vertices[:,:-1,1:]
    d,c = vertices[:,1:,:-1],vertices[:,1:,1:]
    dx = torch.stack((b-a,b-a,c-d,c-d),dim=-2)*(vertices.shape[2]-1)
    dy = torch.stack((d-a,c-b,c-b,d-a),dim=-2)*(vertices.shape[1]-1)
    determinant = dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    frobenius = dx.square().sum(-1)+dy.square().sum(-1)
    return (frobenius*(1+determinant.reciprocal().square())-4).mean()


def require_nested_fine_margin(vertices,reference_corners,minimum_jacobian,context):
    """Extra configured margin, not merely positivity/topological validity."""
    margin=(q1_corner_determinants(vertices.double())/reference_corners-minimum_jacobian).amin()
    if not bool(torch.isfinite(margin) and margin>0):
        raise RuntimeError(f"{context}: rounded nested fine-grid margin is not strictly positive")
    return margin


class Evidence:
    """Fixed mask, original evidence, and explicitly penalized out-of-bounds."""

    def __init__(self, fixed, moving, matrix, offset, loss, strain_weight, oob_weight, shape_weight=0., fixed_mask=None, interpolation="q1", matches=None, match_weight=0., strain_model="displacement_gradient", mind_order="transport", joint_prior_backend="eager"):
        self.fixed, self.moving = fixed, moving
        self.matrix, self.offset = matrix, offset
        self.loss, self.strain_weight, self.oob_weight = loss, strain_weight, oob_weight
        self.shape_weight = shape_weight
        if not np.isfinite(match_weight) or match_weight<0 or (match_weight>0 and matches is None):
            raise ValueError("positive match weight requires explicit frozen point evidence")
        self.matches,self.match_weight=matches,float(match_weight)
        if interpolation not in ("q1","p1_ac","p1_bd"):
            raise ValueError("declare q1, p1_ac or p1_bd map interpretation")
        self.interpolation=interpolation
        if strain_model not in ("displacement_gradient","p1_arap") or (strain_model=="p1_arap" and interpolation=="q1"):
            raise ValueError("declare displacement_gradient or actual P1 p1_arap strain model")
        self.strain_model=strain_model
        if joint_prior_backend not in ("eager","inductor"):
            raise ValueError("joint_prior_backend must be eager or inductor")
        if joint_prior_backend=="inductor" and (strain_model!="p1_arap" or interpolation!="p1_ac" or shape_weight<=0):
            raise ValueError("compiled joint priors require P1 ac ARAP and positive shape weight")
        self.joint_prior_backend=joint_prior_backend
        self.joint_p1_priors=None
        if joint_prior_backend=="inductor":
            from tools.coordinated_compiled_priors import make_joint_priors
            self.joint_p1_priors=make_joint_priors("inductor")
        if mind_order not in ("transport","after_warp") or (mind_order=="after_warp" and loss!="mind"):
            raise ValueError("after_warp descriptor order requires MIND; otherwise use transport")
        self.mind_order=mind_order
        self.fixed_p1_evaluator=None
        self.nested_priors=None
        self.mask = (fixed > .04).to(fixed.dtype) if fixed_mask is None else fixed_mask.to(fixed)
        if self.mask.shape != fixed.shape or not bool(torch.isfinite(self.mask).all() and (self.mask>=0).all() and (self.mask<=1).all()):
            raise ValueError("fixed mask must match image and have finite weights in[0,1]")
        if float(self.mask.sum()) == 0:
            raise ValueError("empty fixed foreground")
        self.denominator = self.mask.sum()
        if loss == "mind":
            self.fixed_feature, _ = self_similarity(fixed)
            self.moving_feature = moving if mind_order=="after_warp" else self_similarity(moving)[0]
        else:
            self.fixed_feature, self.moving_feature = fixed, moving

    def prepare_fixed_p1_sampling(self,rows,columns,*,dtype,device):
        """Cache ONLY immutable source pixel-query geometry, before timing trials."""
        if self.interpolation not in ("p1_ac","p1_bd"):
            raise ValueError("frozen P1 sampling requires declared P1 interpolation")
        from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
        from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
        queries=fixed_pixel_centers(*self.fixed.shape[-2:],dtype=dtype,device=device)
        self.fixed_p1_evaluator=FrozenP1Evaluator(rows,columns,queries,self.interpolation[-2:])

    def coarse_nested_evidence(self,coarse_side,fine_side,*,dtype,device):
        """Share frozen inputs; evaluate exactly the nested fine P1 functional.

        Real-arithmetic equivalence, not identical roundedfine evaluation. This
        object never certifies topology. The caller must materialize/check fine
        outputs and use ordinary full Evidence for accepted/output selection.
        """
        if self.interpolation not in ("p1_ac","p1_bd"):
            raise ValueError("coarse nested evidence requires declared P1 interpolation")
        from qcopt.neural_bijection.dense.coordinated_nested_priors import ExactNestedP1Priors
        result=copy.copy(self)
        result.nested_priors=ExactNestedP1Priors(coarse_side,fine_side,
            diagonal=self.interpolation[-2:],dtype=dtype,device=device,strain_model=self.strain_model)
        if (self.fixed_p1_evaluator is None or self.fixed_p1_evaluator.rows!=coarse_side or
                self.fixed_p1_evaluator.columns!=coarse_side):
            result.prepare_fixed_p1_sampling(coarse_side,coarse_side,dtype=dtype,device=device)
        return result

    def __call__(self, vertices):
        if self.interpolation=="q1":
            query = q1_map_at_pixel_centers(vertices,*self.fixed.shape[-2:])
        elif self.fixed_p1_evaluator is None:
            query = p1_map_at_pixel_centers(vertices,*self.fixed.shape[-2:],diagonal=self.interpolation[-2:])
        else:
            query = self.fixed_p1_evaluator(vertices)
        query = query @ self.matrix.T + self.offset
        warped = F.grid_sample(self.moving_feature, (2 * query - 1).to(self.moving_feature.dtype),
                               mode="bilinear", padding_mode="zeros", align_corners=False)
        if self.loss == "mind":
            if self.mind_order=="after_warp":
                warped=self_similarity(warped)[0]
            errors = (self.fixed_feature - warped).abs().mean(1, keepdim=True)
        else:
            count = 49
            sums = lambda image: F.avg_pool2d(image, 7, stride=1, padding=3) * count
            f, w = self.fixed_feature, warped
            sf, sw = sums(f), sums(w)
            cross = sums(f*w) - sf*sw/count
            vf = (sums(f*f) - sf*sf/count).clamp_min(0)
            vw = (sums(w*w) - sw*sw/count).clamp_min(0)
            errors = 1 - cross.square()/(vf*vw + 1e-5)
        image = (errors*self.mask).sum()/self.denominator
        outside = (F.relu(-query) + F.relu(query-1)).square().sum(-1)[:, None]
        oob = (outside*self.mask).sum()/self.denominator
        if self.nested_priors is None:
            if self.joint_p1_priors is not None:
                strain,shape=self.joint_p1_priors(vertices)
            else:
                if self.strain_model=="displacement_gradient":
                    strain = strain_penalty(vertices)
                else:
                    from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
                    strain = p1_arap_energy(vertices,diagonal=self.interpolation[-2:],validate=False)
                shape = corner_symmetric_dirichlet(vertices) if self.shape_weight else None
        else:
            strain,nested_shape = self.nested_priors(vertices)
            shape = nested_shape if self.shape_weight else None
        total = image + self.strain_weight*strain + self.oob_weight*oob
        if shape is not None:
            total = total + self.shape_weight*shape
        match = self.matches(vertices,self.matrix,self.interpolation) if self.match_weight else None
        if match is not None:
            total = total + self.match_weight*match
        outside_fraction = (((query < 0) | (query > 1)).any(-1)[:, None]*self.mask).sum()/self.denominator
        parts = dict(image=image, strain=strain, oob=oob, outside_fraction=outside_fraction)
        if shape is not None:
            parts["shape"] = shape
        if match is not None:
            parts["match"] = match
        return total, parts


def load_image_matches(path,matrix,offset,*,fixed_path,moving_path,image_side,device,dtype,robust_scale):
    record=json.loads(path.read_text(encoding="utf-8"))
    if record.get("targets_manual_landmarks_or_dense_teacher_loaded") is not False:
        raise ValueError("match provenance must explicitly exclude map/manual/dense targets")
    # Records may be prepared on Windows and evaluated on a Linux compute host.
    recorded_basename=lambda value:Path(str(value).replace("\\","/")).name
    if record.get("image_side")!=image_side or recorded_basename(record["fixed"])!=fixed_path.name or recorded_basename(record["moving"])!=moving_path.name:
        raise ValueError("match raster/frame does not agree with this image pair")
    recorded_matrix=np.asarray(record["post_affine_matrix"],dtype=matrix.dtype)
    recorded_offset=np.asarray(record["post_affine_offset"],dtype=offset.dtype)
    if not np.array_equal(recorded_matrix,matrix) or not np.array_equal(recorded_offset,offset):
        raise ValueError("frozen matcher affine does not agree with optimizer affine")
    source=np.asarray(record["source_points_unit"],dtype=np.float64)
    target=np.asarray(record["target_points_unit"],dtype=np.float64)
    confidence=np.asarray(record["confidence"],dtype=np.float64)
    if source.ndim!=2 or source.shape[-1]!=2 or target.shape!=source.shape or confidence.shape!=(len(source),):
        raise ValueError("invalid correspondence table")
    if not np.isfinite(confidence).all() or ((confidence<0)|(confidence>1)).any():
        raise ValueError("invalid original matcher confidence")
    world=target@matrix.T+offset
    eligible=((world>=0)&(world<=1)).all(-1)
    # This exclusion is STATIC original-moving-domain validity, not loss-dependent
    # overlap cropping. Every manual evaluation landmark remains separately scored.
    weights=confidence*eligible
    if int((weights>0).sum())<8:
        raise ValueError("fewer than8 original-moving-domain image matches")
    points=lambda value:torch.as_tensor(value,device=device,dtype=dtype)
    matches=ImageCorrespondences(points(source),points(target),points(weights),
        pixel_scale=image_side,robust_scale=robust_scale)
    metadata=dict(path=str(path),raw_matches=len(source),eligible_matches=int(eligible.sum()),
        static_original_moving_domain_excluded=int((~eligible).sum()),
        confidence_denominator=float(weights.sum()),robust_scale_canvas_px=robust_scale,
        target_frame="affine-aligned moving; loss uses image_side*A(mapped(q)-p)",
        path_check="raster basenames plus side and sampling-affine values; relocated paths allowed, not content identity proof",
        sampling_affine_storage_dtype=str(matrix.dtype),
        mask="fixed target coordinate inside original moving rectangle; no adaptive query dropping")
    for key in ("prediction_side","origin_image_side","origin_match_record","transport_method","transport_factor"):
        if key in record:
            metadata[key]=record[key]
    return matches,metadata


def load_registration_evidence(fixed_path,moving_path,image_side,*,preprocessing="raw_inverted",device="cpu",dtype=torch.float32):
    """Frozen image preprocessing with an ORIGINAL-raster foreground mask.

    Native CLAHE does not redefine the foreground, change coordinates, prewarp
    the moving image, or access matches/labels. The default is the old evidence.
    """
    fixed,_=_read_gray_thumbnail(fixed_path,image_side)
    moving,_=_read_gray_thumbnail(moving_path,image_side)
    mask=(fixed>.04).to(dtype=dtype,device=device)
    metadata=dict(name=preprocessing,mask_source="original inverted grayscale >.04",
                  original_moving_features_no_affine_prewarp=True)
    if preprocessing=="native_dhr":
        from tools.coordinated_native_evidence import load_native_preprocessed_pair
        fixed,moving,native=load_native_preprocessed_pair(fixed_path,moving_path,
                                                        expected_side=image_side,device="cpu")
        metadata["native_capture"]=native
    elif preprocessing!="raw_inverted":
        raise ValueError("declare raw_inverted or native_dhr frozen evidence")
    return fixed.to(device=device,dtype=dtype),moving.to(device=device,dtype=dtype),mask,metadata


def optimize(args, accepted_stage_callback=None):
    overall_start = time.perf_counter()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.grid_side < 3 or min(args.levels) < 3 or max(args.levels) > args.grid_side:
        raise ValueError("coefficient levels must fit output grid")
    if args.inner_steps < 1 or args.cycles < 1 or args.learning_rate <= 0:
        raise ValueError("positive optimization budget required")
    f2_floor_safety_fraction=getattr(args,"f2_floor_safety_fraction",1.)
    if (isinstance(f2_floor_safety_fraction,bool) or
            not isinstance(f2_floor_safety_fraction,(int,float)) or
            not math.isfinite(f2_floor_safety_fraction) or not 0<f2_floor_safety_fraction<=1):
        raise ValueError("f2_floor_safety_fraction must be finite and in (0,1]")
    f2_floor_safety_fraction=float(f2_floor_safety_fraction)
    image_levels=getattr(args,"image_levels",None) or [args.image_side]*len(args.levels)
    if len(image_levels)!=len(args.levels) or min(image_levels)<8 or max(image_levels)>args.image_side or image_levels[-1]!=args.image_side:
        raise ValueError("one image resolution per coefficient level, ending at full image_side")
    continuation_scope=getattr(args,"continuation_scope","all_cycles")
    if continuation_scope not in ("all_cycles","first_cycle"):
        raise ValueError("continuation scope must be all_cycles or first_cycle")
    if min(args.strain_weight,args.oob_weight,getattr(args,"shape_weight",0.))<0:
        raise ValueError("nonnegative objective weights required")
    geometry_backend=getattr(args,"geometry_backend","existing")
    if geometry_backend not in ("existing","stage_cache"):
        raise ValueError("declare existing or fixed-anchor stage_cache geometry backend")
    if geometry_backend=="stage_cache" and args.method not in ("radial","analytic"):
        raise ValueError("stage_cache currently supports global radial/analytic instance stages only")
    control_hierarchy=getattr(args,"control_hierarchy","fixed")
    if control_hierarchy not in ("fixed","nested_p1"):
        raise ValueError("control hierarchy must be fixed or nested_p1")
    nested=control_hierarchy=="nested_p1"
    coordinate_mode=getattr(args,"coordinate_mode","alternating")
    joint_backend=getattr(args,"joint_backend","cached_manual")
    if coordinate_mode not in ("alternating","joint") or joint_backend not in ("ordinary","cached_manual"):
        raise ValueError("declare alternating/joint coordinates and ordinary/cached_manual joint backend")
    joint=coordinate_mode=="joint"
    fine_patch_cells=getattr(args,"fine_patch_cells",0)
    fine_patch_backend=getattr(args,"fine_patch_backend","ordinary")
    if fine_patch_backend not in ("ordinary","manual"):
        raise ValueError("fine_patch_backend must be ordinary or manual")
    if fine_patch_backend=="manual" and not fine_patch_cells:
        raise ValueError("manual fine_patch_backend requires active fine_patch_cells")
    if (isinstance(fine_patch_cells,bool) or not isinstance(fine_patch_cells,int) or fine_patch_cells<0 or
            (fine_patch_cells and (fine_patch_cells<2 or fine_patch_cells%2 or
             args.grid_side-1<2*fine_patch_cells or args.method not in ("radial","analytic") or joint))):
        raise ValueError("fine_patch_cells needs even P>=2, at least2P gridcells, and alternating global radial/analytic")
    filter_steps=getattr(args,"proposal_filter_steps",0)
    filter_min_level=getattr(args,"proposal_filter_min_level",129)
    if (isinstance(filter_steps,bool) or not isinstance(filter_steps,int) or filter_steps<0 or
            isinstance(filter_min_level,bool) or not isinstance(filter_min_level,int) or filter_min_level<3):
        raise ValueError("proposal filter needs nonnegative integer steps and integer minimum level>=3")
    if filter_steps and args.method not in ("radial","analytic"):
        raise ValueError("proposal filter currently supports global radial/analytic only")
    if joint and (args.method not in ("radial","analytic") or geometry_backend!="existing"):
        raise ValueError("joint coordinates require global radial/analytic and geometry_backend existing; joint_backend controls joint geometry")
    nested_evaluation=getattr(args,"nested_evaluation","full_fine")
    if nested_evaluation not in ("full_fine","coarse_exact"):
        raise ValueError("nested evaluation must be full_fine or coarse_exact")
    if nested_evaluation=="coarse_exact" and not nested:
        raise ValueError("coarse_exact evaluation requires nested_p1 control hierarchy")
    strain_model=getattr(args,"strain_model","displacement_gradient")
    mind_order=getattr(args,"mind_order","transport")
    if mind_order not in ("transport","after_warp") or (mind_order=="after_warp" and args.loss!="mind"):
        raise ValueError("after_warp descriptor order requires MIND")
    if strain_model not in ("displacement_gradient","p1_arap") or (strain_model=="p1_arap" and
            getattr(args,"interpolation","q1")=="q1"):
        raise ValueError("p1_arap needs declared P1")
    joint_prior_backend=getattr(args,"joint_prior_backend","eager")
    if joint_prior_backend not in ("eager","inductor"):
        raise ValueError("joint_prior_backend must be eager or inductor")
    if joint_prior_backend=="inductor" and (strain_model!="p1_arap" or
            getattr(args,"interpolation","q1")!="p1_ac" or getattr(args,"shape_weight",0.)<=0 or
            args.precision!="float64" or nested):
        raise ValueError("compiled joint priors currently require fixed P1 ac controls, ARAP, positive shape weight and float64 geometry")
    if nested and (args.method not in ("radial","analytic") or args.cycles!=1 or
            getattr(args,"interpolation","q1") not in ("p1_ac","p1_bd") or
            args.levels[-1]!=args.grid_side or
            any(b<=a or (b-1)%(a-1) for a,b in zip(args.levels,args.levels[1:]))):
        raise ValueError("nested_p1 needs one increasing nested global P1 cycle ending at grid_side")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    image_precision = getattr(args,"image_precision","same")
    image_dtype = dtype if image_precision=="same" else (torch.float64 if image_precision=="float64" else torch.float32)
    with np.load(args.affine) as data:
        a = np.asarray(data["post_affine_matrix"], dtype=np.float32)
        b = np.asarray(data["post_affine_offset"], dtype=np.float32)
    if a.shape != (2, 2) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a.astype(np.float64)) <= 0:
        raise ValueError("finite positive affine required")
    fixed,moving,fixed_mask,preprocessing_metadata=load_registration_evidence(
        args.fixed,args.moving,args.image_side,preprocessing=getattr(args,"preprocessing","raw_inverted"),
        device=device,dtype=image_dtype)
    reference = identity_vertices(args.grid_side, device=device).to(dtype)
    reference_corners = q1_corner_determinants(reference.double())
    current = (identity_vertices(args.levels[0],device=device).to(dtype)
               if nested else reference.clone())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loading_seconds = time.perf_counter()-overall_start
    feature_start = time.perf_counter()
    matches,match_metadata=None,None
    match_weight=getattr(args,"match_weight",0.)
    if match_weight:
        if getattr(args,"matches",None) is None:
            raise ValueError("declare --matches for a positive match weight")
        matches,match_metadata=load_image_matches(args.matches,a,b,fixed_path=args.fixed,moving_path=args.moving,
            image_side=args.image_side,device=device,dtype=dtype,robust_scale=getattr(args,"match_robust_scale",8.))
    evidence = Evidence(fixed, moving, torch.from_numpy(a).to(device=device, dtype=dtype),
                        torch.from_numpy(b).to(device=device, dtype=dtype), args.loss,
                        args.strain_weight, args.oob_weight, getattr(args,"shape_weight",0.),
                        fixed_mask=fixed_mask,interpolation=getattr(args,"interpolation","q1"),matches=matches,match_weight=match_weight,strain_model=strain_model,mind_order=mind_order,joint_prior_backend=joint_prior_backend)
    evidence_by_resolution={args.image_side:evidence}
    for image_resolution in sorted(set(image_levels)):
        if image_resolution not in evidence_by_resolution:
            reduced_fixed=F.interpolate(fixed,size=(image_resolution,image_resolution),mode="area")
            reduced_moving=F.interpolate(moving,size=(image_resolution,image_resolution),mode="area")
            reduced_mask=F.interpolate(evidence.mask,size=(image_resolution,image_resolution),mode="area")
            evidence_by_resolution[image_resolution]=Evidence(reduced_fixed,reduced_moving,evidence.matrix,
                evidence.offset,args.loss,args.strain_weight,args.oob_weight,getattr(args,"shape_weight",0.),
                reduced_mask,interpolation=getattr(args,"interpolation","q1"),matches=matches,match_weight=match_weight,strain_model=strain_model,mind_order=mind_order)
            # The prior sees the SAME control shape, not the raster resolution.
            # Share its callable, never an energy/graph tied to a current map.
            evidence_by_resolution[image_resolution].joint_prior_backend=joint_prior_backend
            evidence_by_resolution[image_resolution].joint_p1_priors=evidence.joint_p1_priors
    p1_sampling=getattr(args,"p1_sampling","existing")
    if p1_sampling not in ("existing","frozen"):
        raise ValueError("P1 sampling must be existing or frozen")
    if p1_sampling=="frozen":
        for item in evidence_by_resolution.values():
            item.prepare_fixed_p1_sampling(args.grid_side,args.grid_side,dtype=dtype,device=device)
    materializers={}
    if nested:
        from qcopt.neural_bijection.dense.coordinated_nested_refinement import FrozenNestedP1Refinement
        for level in args.levels:
            materializers[level]=FrozenNestedP1Refinement(level,args.grid_side,
                diagonal=args.interpolation[-2:],dtype=dtype,device=device)
    def materialize(vertices):
        return materializers[vertices.shape[1]](vertices) if nested else vertices
    coarse_evidence={}
    if nested_evaluation=="coarse_exact":
        for level,resolution in zip(args.levels,image_levels):
            coarse_evidence[(level,resolution)]=evidence_by_resolution[resolution].coarse_nested_evidence(
                level,args.grid_side,dtype=dtype,device=device)
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    synchronize()
    feature_seconds = time.perf_counter()-feature_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stages, trace, failures, forward_seconds, backward_seconds = [], [], [], [], []
    evaluations, gradient_steps, failed_trials = 0, 0, 0
    coordinated_substeps=0
    start = time.perf_counter()
    if nested:
        require_nested_fine_margin(materialize(current),reference_corners,args.minimum_jacobian,"initial anchor")
    initial, parts = evidence(materialize(current))
    initial_record = {key: float(value) for key, value in parts.items()}
    initial_record["total"] = float(initial)
    output_selection = getattr(args,"output_selection","last")
    if output_selection not in ("last","best_full"):
        raise ValueError("output selection must be last or best_full")
    full_best_map, full_best_loss, full_best_stage = materialize(current).detach().clone(), float(initial), None
    for cycle in range(args.cycles):
        for level_index,level in enumerate(args.levels):
            if nested and current.shape[1]!=level:
                from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
                factor=(level-1)//(current.shape[1]-1)
                current=refine_p1_vertices(current,factor,args.interpolation[-2:]).detach()
            control_side=current.shape[1]
            control_reference=(identity_vertices(control_side,device=device).to(dtype)
                               if nested else reference)
            if nested and not validate_q1_map(current,control_reference)["valid"]:
                raise RuntimeError("rounded nested control refinement failed actual map check")
            if nested:
                require_nested_fine_margin(materialize(current),reference_corners,args.minimum_jacobian,"refined anchor")
            stage_resolution=(image_levels[level_index] if cycle==0 or continuation_scope=="all_cycles"
                              else args.image_side)
            stage_evidence=evidence_by_resolution[stage_resolution]
            original_stage_evidence=stage_evidence
            if nested_evaluation=="coarse_exact":
                stage_evidence=coarse_evidence[(control_side,stage_resolution)]
            def trial_evidence(vertices):
                return stage_evidence(vertices if nested_evaluation=="coarse_exact" else materialize(vertices))
            directions = (("joint_xy",) if joint else
                (((1., 0.), (0., 1.)) if args.method not in ("f1","f2") else (None,)))
            for direction in directions:
                anchor = current.detach()
                channels = 2 if joint or direction is None else 1
                coefficients = torch.nn.Parameter(torch.zeros(1, channels, level-2, level-2,
                                                               device=device, dtype=dtype))
                physical_lr = args.learning_rate*(args.levels[0]-1)/(level-1) if args.lr_calibration == "edge" else args.learning_rate
                optimizer = torch.optim.Adam([coefficients], lr=physical_lr)
                if joint:
                    from qcopt.neural_bijection.dense.coordinated_joint_stage import FrozenAnchorJointCoordinatedUpdate
                    layer=FrozenAnchorJointCoordinatedUpdate(anchor,reference=control_reference,
                        mode=args.method,minimum_jacobian=args.minimum_jacobian,theta=.95,backend=joint_backend)
                elif direction is None:
                    if args.method == "f1":
                        layer = AdaptiveSoftRadialQ1Relaxation(args.grid_side, raw_span=8.,
                            safety_fraction=.75, minimum_jacobian=args.minimum_jacobian).to(device)
                        coverage = None
                    else:
                        layer = StaggeredPatchQ1Layer(args.grid_side,args.patch_cells,proposal_mode="fixed_h",
                            raw_span=.5,safety_fraction=.75,minimum_jacobian=args.minimum_jacobian,
                            accepted_gain=args.f2_accepted_gain,
                            floor_safety_fraction=f2_floor_safety_fraction).to(device)
                        coverage = f2_coverage(layer,args.grid_side,dtype)
                else:
                    mode = "analytic" if "analytic" in args.method else "radial"
                    fine_patch_active=bool(fine_patch_cells and level==args.levels[-1])
                    if fine_patch_active:
                        from qcopt.neural_bijection.dense.coordinated_patch_cascade import CoordinatedPatchCascade
                        layer=CoordinatedPatchCascade(control_side,patch_cells=fine_patch_cells,
                            direction=direction,mode=mode,minimum_jacobian=args.minimum_jacobian,theta=.95,
                            backward_backend=fine_patch_backend).to(device)
                    elif geometry_backend=="stage_cache":
                        from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
                        layer=FrozenAnchorCoordinatedUpdate(anchor,direction=direction,mode=mode,
                            minimum_jacobian=args.minimum_jacobian,theta=.95)
                    else:
                        layer = CoordinatedQ1Update(direction, mode=mode,
                            minimum_jacobian=args.minimum_jacobian, theta=.95).to(device)
                    regional_active = args.method.startswith(("regional_","tapered_global_")) and level >= getattr(args,"regional_min_level",3)
                    if regional_active:
                        half=args.regional_cells//2
                        offsets=((0,0),(half,0),(0,half),(half,half))
                        offset=offsets[(cycle*len(args.levels)+level_index)%4]
                        regional=CoordinatedPatchQ1Pass(args.grid_side,patch_cells=args.regional_cells,
                            direction=direction,mode=mode,offset_row=offset[0],offset_column=offset[1],
                            minimum_jacobian=args.minimum_jacobian,theta=.95).to(device)
                best_map, best_loss = anchor, float(trial_evidence(anchor)[0])
                anchor_loss = best_loss
                if nested_evaluation=="coarse_exact":
                    with torch.no_grad():
                        original_anchor_loss=float(original_stage_evidence(materialize(anchor))[0])
                best_diagnostics = {}
                # Include the last optimizer step as an evaluated trial.
                for step in range(args.inner_steps + 1):
                    optimizer.zero_grad(set_to_none=True)
                    synchronize(); tick = time.perf_counter()
                    applied_filter_steps=filter_steps if level>=filter_min_level else 0
                    if applied_filter_steps:
                        from qcopt.neural_bijection.dense.coordinated_proposal_filter import dirichlet_lazy_proposal_filter
                        filtered_coefficients=dirichlet_lazy_proposal_filter(coefficients,steps=applied_filter_steps)
                    else:
                        filtered_coefficients=coefficients
                    coarse = F.pad(filtered_coefficients, (1, 1, 1, 1))
                    if joint:
                        px=interpolate_proposal(coarse[:,0],(control_side,control_side))
                        py=interpolate_proposal(coarse[:,1],(control_side,control_side))
                        result=layer(px,py,validate=False)
                        candidate=result.vertices
                        scales,gauges=result.scales,result.gauges
                        diagnostics=dict(scale=float(scales.min()),mean_scale=float(scales.mean()),
                            gauge=float(gauges.max()),scale_x=float(scales[:,0].min()),scale_y=float(scales[:,1].min()),
                            gauge_x=float(gauges[:,0].max()),gauge_y=float(gauges[:,1].max()),
                            intermediate_margin=float(result.substep_margin_min[:,0].min()),
                            margin=float(result.normalized_margin_min.min()))
                    elif direction is None:
                        physical_coefficients = coefficients/args.f2_accepted_gain if args.method=="f2" else coefficients
                        candidate = decode_control(layer,args.method,anchor,physical_coefficients,coverage)
                        margin = (q1_corner_determinants(candidate.double())/reference_corners-args.minimum_jacobian).amin()
                        diagnostics = dict(margin=float(margin.detach()))
                    else:
                        proposal = interpolate_proposal(coarse[:, 0], (control_side, control_side))
                        if fine_patch_active:
                            result=layer(anchor,proposal,reference=control_reference,validate=False)
                            scales,gauges=result.patch_scales,result.patch_gauges
                        elif regional_active and args.method.startswith("regional_"):
                            result=regional(anchor,proposal,reference=reference,validate=False)
                            scales,gauges=result.patch_scales,result.patch_gauges
                        else:
                            if regional_active and args.method.startswith("tapered_global_"):
                                proposal=regional.windowed_proposal(proposal)
                            result = (layer(proposal,validate=False) if geometry_backend=="stage_cache"
                                      else layer(anchor, proposal, validate=False))
                            scales,gauges=result.scale,result.gauge
                        candidate = result.vertices
                        diagnostics = dict(scale=float(scales.detach().min()),mean_scale=float(scales.detach().mean()),
                                           gauge=float(gauges.detach().max()),
                                           margin=float(result.normalized_margin_min.detach().min()))
                        if fine_patch_active:
                            diagnostics.update(intermediate_margin=float(result.pass_margin_min.detach().min()),
                                intermediate_margins_finite=bool(torch.isfinite(result.pass_margin_min).all()),
                                geometry_pass_count=result.geometry_pass_count)
                    if args.method in ("radial","analytic"):
                        with torch.no_grad():
                            diagnostics.update(proposal_filter_steps=applied_filter_steps,
                                raw_coefficient_rms=float(coefficients.square().mean().sqrt()),
                                filtered_coefficient_rms=float(filtered_coefficients.square().mean().sqrt()),
                                candidate_displacement_rms=float((candidate-anchor).square().sum(-1).mean().sqrt()))
                    fine_candidate=materialize(candidate.detach() if nested_evaluation=="coarse_exact" else candidate)
                    if nested:
                        fine_margin=(q1_corner_determinants(fine_candidate.double())/reference_corners-args.minimum_jacobian).amin()
                        diagnostics["coarse_margin"]=diagnostics["margin"]
                        diagnostics["margin"]=float(fine_margin.detach())
                    total, parts = (stage_evidence(candidate) if nested_evaluation=="coarse_exact"
                                    else stage_evidence(fine_candidate))
                    synchronize(); forward_seconds.append(time.perf_counter()-tick)
                    evaluations += 1
                    if args.method in ("radial","analytic"):
                        coordinated_substeps+=2 if joint else (4 if fine_patch_cells and level==args.levels[-1] else 1)
                    value = float(total.detach())
                    legal = bool(torch.isfinite(fine_candidate).all())
                    legal = legal and np.isfinite(diagnostics["margin"]) and diagnostics["margin"] > 0
                    if joint or (fine_patch_cells and level==args.levels[-1]):
                        legal=legal and np.isfinite(diagnostics["intermediate_margin"]) and diagnostics["intermediate_margin"]>0
                        legal=legal and diagnostics.get("intermediate_margins_finite",True)
                    if not np.isfinite(value) or not legal:
                        failed_trials += 1
                        failures.append(dict(cycle=cycle,level=level,direction=direction,step=step,
                                             reason="nonfinite objective/coordinates or rounded margin",total=value,**diagnostics))
                        break
                    if value < best_loss:
                        best_loss, best_map = value, candidate.detach().clone()
                        best_diagnostics = diagnostics
                    trace.append(dict(cycle=cycle, level=level, control_side=control_side, image_side=stage_resolution, direction=direction, step=step,
                                      total=value, **{k: float(v.detach()) for k, v in parts.items()},
                                      **diagnostics))
                    if step < args.inner_steps:
                        tick = time.perf_counter(); total.backward(); synchronize()
                        backward_seconds.append(time.perf_counter()-tick)
                        if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                            failed_trials += 1
                            failures.append(dict(cycle=cycle,level=level,direction=direction,step=step,
                                                 reason="missing/nonfinite gradient",**diagnostics))
                            break
                        gradient_steps += 1
                        optimizer.step()
                acceptance_details={}
                if nested_evaluation=="coarse_exact":
                    reduced_best_loss=best_loss
                    with torch.no_grad():
                        original_best_loss=float(original_stage_evidence(materialize(best_map))[0])
                    # Rounded coarse/fine equivalence is not bitwise. Accept by
                    # the original complete STAGE objective, never manual TRE.
                    rejected=not np.isfinite(original_best_loss) or original_best_loss>original_anchor_loss
                    acceptance_details=dict(anchor_reduced_total=anchor_loss,
                        proposed_reduced_total=reduced_best_loss,proposed_original_total=original_best_loss,
                        rounded_full_stage_fallback=rejected)
                    if rejected:
                        best_map,best_diagnostics=anchor,{}
                        best_loss=original_anchor_loss
                    else:
                        best_loss=original_best_loss
                    anchor_loss=original_anchor_loss
                if not validate_q1_map(best_map, control_reference)["valid"]:
                    raise RuntimeError("best candidate failed independent accepted-map check")
                current = best_map
                accepted_fine=materialize(current)
                if nested and not validate_q1_map(accepted_fine,reference)["valid"]:
                    raise RuntimeError("accepted nested map failed actual fine-grid check")
                if nested:
                    require_nested_fine_margin(accepted_fine,reference_corners,args.minimum_jacobian,"accepted anchor/fallback")
                accepted_full_loss = float(evidence(accepted_fine)[0])
                stages.append(dict(cycle=cycle, level=level, direction=direction,
                                   control_side=control_side,image_side=stage_resolution,physical_lr=physical_lr,anchor_total=anchor_loss,
                                   accepted_total=best_loss,accepted_full_total=accepted_full_loss,**best_diagnostics,**acceptance_details))
                if accepted_full_loss < full_best_loss:
                    full_best_map = accepted_fine.detach().clone()
                    full_best_loss, full_best_stage = accepted_full_loss, len(stages)-1
                if accepted_stage_callback is not None:
                    snapshot=accepted_fine.detach().clone()
                    synchronize()
                    accepted_stage_callback(snapshot,dict(stages[-1]),time.perf_counter()-start)
    terminal_full_loss = stages[-1]["accepted_full_total"]
    if output_selection == "best_full":
        current = full_best_map
    elif nested:
        current=materialize(current)
    synchronize(); elapsed = time.perf_counter()-start
    final, parts = evidence(current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialization_start = time.perf_counter()
    np.savez(args.output, vertices=current.cpu().numpy(), boundary_reference=reference.cpu().numpy(),
             post_affine_matrix=a, post_affine_offset=b,interpolation=np.asarray(getattr(args,"interpolation","q1")))
    serialization_seconds = time.perf_counter()-serialization_start
    certificate_start = time.perf_counter()
    certificate = certify_q1_binary_map(args.output)
    certification_seconds = time.perf_counter()-certificate_start
    base_query_buffers={id(buffer) for item in evidence_by_resolution.values()
        if item.fixed_p1_evaluator is not None for buffer in item.fixed_p1_evaluator.buffers()}
    new_query_buffers={id(buffer):buffer for item in coarse_evidence.values()
        for buffer in item.fixed_p1_evaluator.buffers() if id(buffer) not in base_query_buffers}
    nested_prior_bytes=sum(item.nested_priors.resident_bytes for item in coarse_evidence.values())
    report = dict(configuration={**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                 "f2_floor_safety_fraction":f2_floor_safety_fraction,
                                 "joint_prior_backend":joint_prior_backend},
                  representation=evidence.interpolation+" residual then frozen positive affine; exact declared interpretation",
                  initial=initial_record, final=dict(total=float(final), **{k: float(v) for k,v in parts.items()}),
                  control_vertices=args.grid_side**2, cells=(args.grid_side-1)**2,
                  corner_constraints=4*(args.grid_side-1)**2, query_count=args.image_side**2,
                  evaluations=evaluations, gradient_steps=gradient_steps, failed_trials=failed_trials,
                  objective_evaluations=evaluations+2*len(stages)+2+(2*len(stages) if nested_evaluation=="coarse_exact" else 0),
                  optimize_seconds=elapsed, median_forward_objective_seconds=float(np.median(forward_seconds)),
                  loading_seconds=loading_seconds,feature_seconds=feature_seconds,
                  serialization_seconds=serialization_seconds,certification_seconds=certification_seconds,
                  end_to_end_seconds=time.perf_counter()-overall_start,
                  median_vjp_seconds=float(np.median(backward_seconds)) if backward_seconds else None,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  stages=stages, trace=trace, failures=failures,saved_binary_certificate=certificate,
                  landmarks_used=False, mask="fixed grayscale inversion > .04, constant denominator",
                  final_corner_shape=float(corner_symmetric_dirichlet(current.double())),
                  geometry_dtype=str(dtype),evidence_dtype=str(image_dtype),
                  strain_model=strain_model,
                  joint_prior_backend=joint_prior_backend,
                  f2_floor_safety_fraction=f2_floor_safety_fraction,
                  mind_order=mind_order if args.loss=="mind" else None,
                  image_levels=image_levels,
                  continuation_scope=continuation_scope,
                  output_selection=output_selection,selected_stage=full_best_stage if output_selection=="best_full" else len(stages)-1,
                  image_match_evidence=match_metadata,
                  image_preprocessing=preprocessing_metadata,
                  p1_sampling=p1_sampling,
                  geometry_backend=geometry_backend,
                  geometry_backend_scope=("configured global-stage backend; final patch cascade uses connected "+
                    ("first-order full-Y/proposal manual VJP; auxiliary geometry diagnostics are non-differentiable" if fine_patch_backend=="manual" else "ordinary AD")) if fine_patch_cells else "configured global-stage backend",
                  coordinate_mode=coordinate_mode,joint_backend=joint_backend if joint else None,
                  proposal_filter_steps=filter_steps,proposal_filter_min_level=filter_min_level,
                  fine_patch_cells=fine_patch_cells,
                  fine_patch_backend=fine_patch_backend if fine_patch_cells else None,
                  coordinated_substep_evaluations=coordinated_substeps if args.method in ("radial","analytic") else None,
                  extra_joint_diagnostic_passes=evaluations if joint and joint_backend=="cached_manual" else 0,
                  control_hierarchy=control_hierarchy,
                  nested_evaluation=nested_evaluation,
                  nested_prior_cache_bytes=nested_prior_bytes,
                  nested_evidence_cache_bytes=nested_prior_bytes+
                    sum(buffer.numel()*buffer.element_size() for buffer in new_query_buffers.values()),
                  control_sizes=args.levels if nested else [args.grid_side]*len(args.levels),
                  nested_p1_cache_bytes=sum(m.resident_bytes for m in materializers.values()),
                  fixed_p1_cache_bytes=sum(buffer.numel()*buffer.element_size()
                    for item in evidence_by_resolution.values() if item.fixed_p1_evaluator is not None
                    for buffer in item.fixed_p1_evaluator.buffers()),
                  terminal_full_total=terminal_full_loss,best_accepted_full_total=full_best_loss,
                  acceptance_objective="original complete objective at current image resolution; full-resolution trajectory may not be monotone",
                  oob="zero padding, fixed denominator, explicit quadratic excess penalty; no query dropping")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "affine", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--method", choices=("radial", "analytic", "f1", "f2", "regional_radial", "regional_analytic", "tapered_global_radial", "tapered_global_analytic"), default="radial")
    p.add_argument("--patch-cells",type=int,default=8)
    p.add_argument("--fine-patch-cells",type=int,default=0,
                   help="Optional final-level four-pass coordinated patch support; zero preserves global updates")
    p.add_argument("--fine-patch-backend",choices=("ordinary","manual"),default="ordinary",
                   help="Final patch cascade only: ordinary AD or connected first-order full-Y/proposal VJP; manual diagnostics are non-differentiable")
    p.add_argument("--f2-accepted-gain",type=float,default=1.)
    p.add_argument("--f2-floor-safety-fraction",type=float,default=1.,
                   help="fraction of remaining F2 area-floor slack; historical1 permits contact, <1 reserves strict real-arithmetic slack")
    p.add_argument("--regional-cells",type=int,default=32)
    p.add_argument("--regional-min-level",type=int,default=3,
                   help="Use unwindowed global updates below this coefficient level")
    p.add_argument("--loss", choices=("mind", "local_ncc"), default="mind")
    p.add_argument("--mind-order",choices=("transport","after_warp"),default="transport")
    p.add_argument("--preprocessing",choices=("raw_inverted","native_dhr"),default="raw_inverted",
                   help="Frozen native PIL/normalization/grayscale/CLAHE option; mask stays original, not native optimizer equivalence")
    p.add_argument("--interpolation",choices=("q1","p1_ac","p1_bd"),default="q1")
    p.add_argument("--p1-sampling",choices=("existing","frozen"),default="existing",
                   help="Optional immutable fixed-source query cache; no moving-query gradients")
    p.add_argument("--geometry-backend",choices=("existing","stage_cache"),default="existing",
                   help="Optional constant-anchor global radial/analytic instance-stage cache; not trainable-anchor neural API")
    p.add_argument("--coordinate-mode",choices=("alternating","joint"),default="alternating",
                   help="Optional joint x-then-y latent optimization; global radial/analytic only, geometry-backend existing")
    p.add_argument("--joint-backend",choices=("ordinary","cached_manual"),default="cached_manual",
                   help="Joint stage ordinary AD or cached first step/manual full-Y second adjoint plus explicitly counted no-grad diagnostics")
    p.add_argument("--proposal-filter-steps",type=int,default=0,help="zero-ghost passes on RAW interior coefficients before safe decoding")
    p.add_argument("--proposal-filter-min-level",type=int,default=129)
    p.add_argument("--control-hierarchy",choices=("fixed","nested_p1"),default="fixed",
                   help="Actual increasing P1 control meshes; objective always uses final fine mesh, one global cycle only")
    p.add_argument("--nested-evaluation",choices=("full_fine","coarse_exact"),default="full_fine",
                   help="Optional same fine functional via direct coarse P1 queries/count quadrature; fresh rounded fine checks and original stage acceptance retained")
    p.add_argument("--grid-side", type=int, default=257)
    p.add_argument("--image-side", type=int, default=512)
    p.add_argument("--image-levels", type=int, nargs="+",
                   help="image continuation resolutions matching --levels, e.g.32 64 128 256 512")
    p.add_argument("--continuation-scope",choices=("all_cycles","first_cycle"),default="all_cycles",
                   help="Optional full-image objective for all coefficient levels after initial continuation cycle")
    p.add_argument("--levels", type=int, nargs="+", default=[17,33,65,129,257])
    p.add_argument("--output-selection",choices=("last","best_full"),default="last",
                   help="Select accepted output using full-resolution complete objective only, never landmarks")
    p.add_argument("--matches",type=Path,help="frozen image-only raw matcher JSON in the identical affine frame")
    p.add_argument("--match-weight",type=float,default=0.)
    p.add_argument("--match-robust-scale",type=float,default=8.,help="pseudohuber error scale in full moving-canvas pixels")
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=.004)
    p.add_argument("--lr-calibration", choices=("physical", "edge"), default="physical")
    p.add_argument("--strain-weight", type=float, default=.05)
    p.add_argument("--strain-model",choices=("displacement_gradient","p1_arap"),default="displacement_gradient")
    p.add_argument("--joint-prior-backend",choices=("eager","inductor"),default="eager",
                   help="Optional fullgraph default Inductor joint ARAP+shape prior; fixed P1 ac/float64 only. First-call compilation is part of execution time.")
    p.add_argument("--shape-weight", type=float, default=0.)
    p.add_argument("--oob-weight", type=float, default=1.)
    p.add_argument("--minimum-jacobian", type=float, default=.001)
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--image-precision", choices=("same","float32","float64"), default="same")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    report = optimize(p.parse_args())
    print(json.dumps({k: report[k] for k in ("initial", "final", "optimize_seconds", "evaluations", "gradient_steps")}))


if __name__ == "__main__":
    main()
