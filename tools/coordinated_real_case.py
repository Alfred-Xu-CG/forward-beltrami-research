"""Shared-evidence, stage-anchored instance registration. No landmarks are read.

The saved residual is declared Q1 or P1 on the grid; a positive frozen affine is
postcomposed. Queries sample original moving evidence, not a recursively
resampled raster. Selection uses the complete image + cumulative-strain +
out-of-bounds objective, with a fixed foreground denominator.
"""
from __future__ import annotations

import argparse
import json
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


class Evidence:
    """Fixed mask, original evidence, and explicitly penalized out-of-bounds."""

    def __init__(self, fixed, moving, matrix, offset, loss, strain_weight, oob_weight, shape_weight=0., fixed_mask=None, interpolation="q1"):
        self.fixed, self.moving = fixed, moving
        self.matrix, self.offset = matrix, offset
        self.loss, self.strain_weight, self.oob_weight = loss, strain_weight, oob_weight
        self.shape_weight = shape_weight
        if interpolation not in ("q1","p1_ac","p1_bd"):
            raise ValueError("declare q1, p1_ac or p1_bd map interpretation")
        self.interpolation=interpolation
        self.mask = (fixed > .04).to(fixed.dtype) if fixed_mask is None else fixed_mask.to(fixed)
        if self.mask.shape != fixed.shape or not bool(torch.isfinite(self.mask).all() and (self.mask>=0).all() and (self.mask<=1).all()):
            raise ValueError("fixed mask must match image and have finite weights in[0,1]")
        if float(self.mask.sum()) == 0:
            raise ValueError("empty fixed foreground")
        self.denominator = self.mask.sum()
        if loss == "mind":
            self.fixed_feature, _ = self_similarity(fixed)
            self.moving_feature, _ = self_similarity(moving)
        else:
            self.fixed_feature, self.moving_feature = fixed, moving

    def __call__(self, vertices):
        if self.interpolation=="q1":
            query = q1_map_at_pixel_centers(vertices,*self.fixed.shape[-2:])
        else:
            query = p1_map_at_pixel_centers(vertices,*self.fixed.shape[-2:],diagonal=self.interpolation[-2:])
        query = query @ self.matrix.T + self.offset
        warped = F.grid_sample(self.moving_feature, (2 * query - 1).to(self.moving_feature.dtype),
                               mode="bilinear", padding_mode="zeros", align_corners=False)
        if self.loss == "mind":
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
        strain = strain_penalty(vertices)
        total = image + self.strain_weight*strain + self.oob_weight*oob
        shape = corner_symmetric_dirichlet(vertices) if self.shape_weight else None
        if shape is not None:
            total = total + self.shape_weight*shape
        outside_fraction = (((query < 0) | (query > 1)).any(-1)[:, None]*self.mask).sum()/self.denominator
        parts = dict(image=image, strain=strain, oob=oob, outside_fraction=outside_fraction)
        if shape is not None:
            parts["shape"] = shape
        return total, parts


def optimize(args, accepted_stage_callback=None):
    overall_start = time.perf_counter()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.grid_side < 3 or min(args.levels) < 3 or max(args.levels) > args.grid_side:
        raise ValueError("coefficient levels must fit output grid")
    if args.inner_steps < 1 or args.cycles < 1 or args.learning_rate <= 0:
        raise ValueError("positive optimization budget required")
    image_levels=getattr(args,"image_levels",None) or [args.image_side]*len(args.levels)
    if len(image_levels)!=len(args.levels) or min(image_levels)<8 or max(image_levels)>args.image_side or image_levels[-1]!=args.image_side:
        raise ValueError("one image resolution per coefficient level, ending at full image_side")
    if min(args.strain_weight,args.oob_weight,getattr(args,"shape_weight",0.))<0:
        raise ValueError("nonnegative objective weights required")
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
    fixed, _ = _read_gray_thumbnail(args.fixed, args.image_side)
    moving, _ = _read_gray_thumbnail(args.moving, args.image_side)
    fixed, moving = fixed.to(device=device, dtype=image_dtype), moving.to(device=device, dtype=image_dtype)
    reference = identity_vertices(args.grid_side, device=device).to(dtype)
    reference_corners = q1_corner_determinants(reference.double())
    current = reference.clone()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loading_seconds = time.perf_counter()-overall_start
    feature_start = time.perf_counter()
    evidence = Evidence(fixed, moving, torch.from_numpy(a).to(device=device, dtype=dtype),
                        torch.from_numpy(b).to(device=device, dtype=dtype), args.loss,
                        args.strain_weight, args.oob_weight, getattr(args,"shape_weight",0.),
                        interpolation=getattr(args,"interpolation","q1"))
    evidence_by_resolution={args.image_side:evidence}
    for image_resolution in sorted(set(image_levels)):
        if image_resolution not in evidence_by_resolution:
            reduced_fixed=F.interpolate(fixed,size=(image_resolution,image_resolution),mode="area")
            reduced_moving=F.interpolate(moving,size=(image_resolution,image_resolution),mode="area")
            reduced_mask=F.interpolate(evidence.mask,size=(image_resolution,image_resolution),mode="area")
            evidence_by_resolution[image_resolution]=Evidence(reduced_fixed,reduced_moving,evidence.matrix,
                evidence.offset,args.loss,args.strain_weight,args.oob_weight,getattr(args,"shape_weight",0.),
                reduced_mask,interpolation=getattr(args,"interpolation","q1"))
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    synchronize()
    feature_seconds = time.perf_counter()-feature_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stages, trace, failures, forward_seconds, backward_seconds = [], [], [], [], []
    evaluations, gradient_steps, failed_trials = 0, 0, 0
    start = time.perf_counter()
    initial, parts = evidence(current)
    initial_record = {key: float(value) for key, value in parts.items()}
    initial_record["total"] = float(initial)
    output_selection = getattr(args,"output_selection","last")
    if output_selection not in ("last","best_full"):
        raise ValueError("output selection must be last or best_full")
    full_best_map, full_best_loss, full_best_stage = current.detach().clone(), float(initial), None
    for cycle in range(args.cycles):
        for level_index,level in enumerate(args.levels):
            stage_evidence=evidence_by_resolution[image_levels[level_index]]
            directions = ((1., 0.), (0., 1.)) if args.method not in ("f1","f2") else (None,)
            for direction in directions:
                anchor = current.detach()
                channels = 2 if direction is None else 1
                coefficients = torch.nn.Parameter(torch.zeros(1, channels, level-2, level-2,
                                                               device=device, dtype=dtype))
                physical_lr = args.learning_rate*(args.levels[0]-1)/(level-1) if args.lr_calibration == "edge" else args.learning_rate
                optimizer = torch.optim.Adam([coefficients], lr=physical_lr)
                if direction is None:
                    if args.method == "f1":
                        layer = AdaptiveSoftRadialQ1Relaxation(args.grid_side, raw_span=8.,
                            safety_fraction=.75, minimum_jacobian=args.minimum_jacobian).to(device)
                        coverage = None
                    else:
                        layer = StaggeredPatchQ1Layer(args.grid_side,args.patch_cells,proposal_mode="fixed_h",
                            raw_span=.5,safety_fraction=.75,minimum_jacobian=args.minimum_jacobian,
                            accepted_gain=args.f2_accepted_gain).to(device)
                        coverage = f2_coverage(layer,args.grid_side,dtype)
                else:
                    mode = "analytic" if "analytic" in args.method else "radial"
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
                best_map, best_loss = anchor, float(stage_evidence(anchor)[0])
                anchor_loss = best_loss
                best_diagnostics = {}
                # Include the last optimizer step as an evaluated trial.
                for step in range(args.inner_steps + 1):
                    optimizer.zero_grad(set_to_none=True)
                    synchronize(); tick = time.perf_counter()
                    coarse = F.pad(coefficients, (1, 1, 1, 1))
                    if direction is None:
                        physical_coefficients = coefficients/args.f2_accepted_gain if args.method=="f2" else coefficients
                        candidate = decode_control(layer,args.method,anchor,physical_coefficients,coverage)
                        margin = (q1_corner_determinants(candidate.double())/reference_corners-args.minimum_jacobian).amin()
                        diagnostics = dict(margin=float(margin.detach()))
                    else:
                        proposal = interpolate_proposal(coarse[:, 0], (args.grid_side, args.grid_side))
                        if regional_active and args.method.startswith("regional_"):
                            result=regional(anchor,proposal,reference=reference,validate=False)
                            scales,gauges=result.patch_scales,result.patch_gauges
                        else:
                            if regional_active and args.method.startswith("tapered_global_"):
                                proposal=regional.windowed_proposal(proposal)
                            result = layer(anchor, proposal, validate=False)
                            scales,gauges=result.scale,result.gauge
                        candidate = result.vertices
                        diagnostics = dict(scale=float(scales.detach().min()),mean_scale=float(scales.detach().mean()),
                                           gauge=float(gauges.detach().max()),
                                           margin=float(result.normalized_margin_min.detach().min()))
                    total, parts = stage_evidence(candidate)
                    synchronize(); forward_seconds.append(time.perf_counter()-tick)
                    evaluations += 1
                    value = float(total.detach())
                    legal = bool(torch.isfinite(candidate).all())
                    legal = legal and diagnostics["margin"] > 0
                    if not np.isfinite(value) or not legal:
                        failed_trials += 1
                        failures.append(dict(cycle=cycle,level=level,direction=direction,step=step,
                                             reason="nonfinite objective/coordinates or rounded margin",total=value,**diagnostics))
                        break
                    if value < best_loss:
                        best_loss, best_map = value, candidate.detach().clone()
                        best_diagnostics = diagnostics
                    trace.append(dict(cycle=cycle, level=level, image_side=image_levels[level_index], direction=direction, step=step,
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
                if not validate_q1_map(best_map, reference)["valid"]:
                    raise RuntimeError("best candidate failed independent accepted-map check")
                current = best_map
                accepted_full_loss = float(evidence(current)[0])
                stages.append(dict(cycle=cycle, level=level, direction=direction,
                                   image_side=image_levels[level_index],physical_lr=physical_lr,anchor_total=anchor_loss,
                                   accepted_total=best_loss,accepted_full_total=accepted_full_loss,**best_diagnostics))
                if accepted_full_loss < full_best_loss:
                    full_best_map = current.detach().clone()
                    full_best_loss, full_best_stage = accepted_full_loss, len(stages)-1
                if accepted_stage_callback is not None:
                    snapshot=current.detach().clone()
                    synchronize()
                    accepted_stage_callback(snapshot,dict(stages[-1]),time.perf_counter()-start)
    terminal_full_loss = stages[-1]["accepted_full_total"]
    if output_selection == "best_full":
        current = full_best_map
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
    report = dict(configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  representation=evidence.interpolation+" residual then frozen positive affine; exact declared interpretation",
                  initial=initial_record, final=dict(total=float(final), **{k: float(v) for k,v in parts.items()}),
                  control_vertices=args.grid_side**2, cells=(args.grid_side-1)**2,
                  corner_constraints=4*(args.grid_side-1)**2, query_count=args.image_side**2,
                  evaluations=evaluations, gradient_steps=gradient_steps, failed_trials=failed_trials,
                  objective_evaluations=evaluations+2*len(stages)+2,
                  optimize_seconds=elapsed, median_forward_objective_seconds=float(np.median(forward_seconds)),
                  loading_seconds=loading_seconds,feature_seconds=feature_seconds,
                  serialization_seconds=serialization_seconds,certification_seconds=certification_seconds,
                  end_to_end_seconds=time.perf_counter()-overall_start,
                  median_vjp_seconds=float(np.median(backward_seconds)),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  stages=stages, trace=trace, failures=failures,saved_binary_certificate=certificate,
                  landmarks_used=False, mask="fixed grayscale inversion > .04, constant denominator",
                  final_corner_shape=float(corner_symmetric_dirichlet(current.double())),
                  geometry_dtype=str(dtype),evidence_dtype=str(image_dtype),
                  image_levels=image_levels,
                  output_selection=output_selection,selected_stage=full_best_stage if output_selection=="best_full" else len(stages)-1,
                  terminal_full_total=terminal_full_loss,best_accepted_full_total=full_best_loss,
                  acceptance_objective="complete objective at current image resolution; full-resolution trajectory may not be monotone",
                  oob="zero padding, fixed denominator, explicit quadratic excess penalty; no query dropping")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "affine", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--method", choices=("radial", "analytic", "f1", "f2", "regional_radial", "regional_analytic", "tapered_global_radial", "tapered_global_analytic"), default="radial")
    p.add_argument("--patch-cells",type=int,default=8)
    p.add_argument("--f2-accepted-gain",type=float,default=1.)
    p.add_argument("--regional-cells",type=int,default=32)
    p.add_argument("--regional-min-level",type=int,default=3,
                   help="Use unwindowed global updates below this coefficient level")
    p.add_argument("--loss", choices=("mind", "local_ncc"), default="mind")
    p.add_argument("--interpolation",choices=("q1","p1_ac","p1_bd"),default="q1")
    p.add_argument("--grid-side", type=int, default=257)
    p.add_argument("--image-side", type=int, default=512)
    p.add_argument("--image-levels", type=int, nargs="+",
                   help="image continuation resolutions matching --levels, e.g.32 64 128 256 512")
    p.add_argument("--levels", type=int, nargs="+", default=[17,33,65,129,257])
    p.add_argument("--output-selection",choices=("last","best_full"),default="last",
                   help="Select accepted output using full-resolution complete objective only, never landmarks")
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=.004)
    p.add_argument("--lr-calibration", choices=("physical", "edge"), default="physical")
    p.add_argument("--strain-weight", type=float, default=.05)
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
