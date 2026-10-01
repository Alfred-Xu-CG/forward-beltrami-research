"""Shared-evidence, stage-anchored instance registration. No landmarks are read.

The saved residual is Q1 on the declared grid; a positive frozen affine is
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
from qcopt.neural_bijection.dense.digital_q1 import AdaptiveSoftRadialQ1Relaxation, StaggeredPatchQ1Layer, q1_corner_determinants, validate_q1_map
from qcopt.neural_bijection.dense.q1_image_sampling import q1_map_at_pixel_centers
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.coordinated_control_capacity import f2_coverage, decode as decode_control


class Evidence:
    """Fixed mask, original evidence, and explicitly penalized out-of-bounds."""

    def __init__(self, fixed, moving, matrix, offset, loss, strain_weight, oob_weight):
        self.fixed, self.moving = fixed, moving
        self.matrix, self.offset = matrix, offset
        self.loss, self.strain_weight, self.oob_weight = loss, strain_weight, oob_weight
        self.mask = (fixed > .04).to(fixed.dtype)
        if float(self.mask.sum()) == 0:
            raise ValueError("empty fixed foreground")
        self.denominator = self.mask.sum()
        if loss == "mind":
            self.fixed_feature, _ = self_similarity(fixed)
            self.moving_feature, _ = self_similarity(moving)
        else:
            self.fixed_feature, self.moving_feature = fixed, moving

    def __call__(self, vertices):
        query = q1_map_at_pixel_centers(vertices, *self.fixed.shape[-2:])
        query = query @ self.matrix.T + self.offset
        warped = F.grid_sample(self.moving_feature, 2 * query - 1,
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
        outside_fraction = (((query < 0) | (query > 1)).any(-1)[:, None]*self.mask).sum()/self.denominator
        return total, dict(image=image, strain=strain, oob=oob, outside_fraction=outside_fraction)


def optimize(args):
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.grid_side < 3 or min(args.levels) < 3 or max(args.levels) > args.grid_side:
        raise ValueError("coefficient levels must fit output grid")
    if args.inner_steps < 1 or args.cycles < 1 or args.learning_rate <= 0:
        raise ValueError("positive optimization budget required")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    with np.load(args.affine) as data:
        a = np.asarray(data["post_affine_matrix"], dtype=np.float32)
        b = np.asarray(data["post_affine_offset"], dtype=np.float32)
    if a.shape != (2, 2) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a.astype(np.float64)) <= 0:
        raise ValueError("finite positive affine required")
    fixed, _ = _read_gray_thumbnail(args.fixed, args.image_side)
    moving, _ = _read_gray_thumbnail(args.moving, args.image_side)
    fixed, moving = fixed.to(device=device, dtype=dtype), moving.to(device=device, dtype=dtype)
    reference = identity_vertices(args.grid_side, device=device).to(dtype)
    reference_corners = q1_corner_determinants(reference.double())
    current = reference.clone()
    evidence = Evidence(fixed, moving, torch.from_numpy(a).to(device=device, dtype=dtype),
                        torch.from_numpy(b).to(device=device, dtype=dtype), args.loss,
                        args.strain_weight, args.oob_weight)
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stages, trace, failures, forward_seconds, backward_seconds = [], [], [], [], []
    evaluations, gradient_steps, failed_trials = 0, 0, 0
    start = time.perf_counter()
    initial, parts = evidence(current)
    initial_record = {key: float(value) for key, value in parts.items()}
    initial_record["total"] = float(initial)
    for cycle in range(args.cycles):
        for level in args.levels:
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
                    layer = CoordinatedQ1Update(direction, mode=args.method,
                        minimum_jacobian=args.minimum_jacobian, theta=.95).to(device)
                best_map, best_loss = anchor, float(evidence(anchor)[0])
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
                        result = layer(anchor, proposal, validate=False)
                        candidate = result.vertices
                        diagnostics = dict(scale=float(result.scale.detach().min()),
                                           gauge=float(result.gauge.detach().max()),
                                           margin=float(result.normalized_margin_min.detach().min()))
                    total, parts = evidence(candidate)
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
                    trace.append(dict(cycle=cycle, level=level, direction=direction, step=step,
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
                stages.append(dict(cycle=cycle, level=level, direction=direction,
                                   physical_lr=physical_lr,anchor_total=anchor_loss, accepted_total=best_loss, **best_diagnostics))
    synchronize(); elapsed = time.perf_counter()-start
    final, parts = evidence(current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, vertices=current.cpu().numpy(), boundary_reference=reference.cpu().numpy(),
             post_affine_matrix=a, post_affine_offset=b)
    certificate = certify_q1_binary_map(args.output)
    report = dict(configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  representation="Q1 residual then frozen positive affine; P1 not used for image objective",
                  initial=initial_record, final=dict(total=float(final), **{k: float(v) for k,v in parts.items()}),
                  control_vertices=args.grid_side**2, cells=(args.grid_side-1)**2,
                  corner_constraints=4*(args.grid_side-1)**2, query_count=args.image_side**2,
                  evaluations=evaluations, gradient_steps=gradient_steps, failed_trials=failed_trials,
                  optimize_seconds=elapsed, median_forward_objective_seconds=float(np.median(forward_seconds)),
                  median_vjp_seconds=float(np.median(backward_seconds)),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  stages=stages, trace=trace, failures=failures,saved_binary_certificate=certificate,
                  landmarks_used=False, mask="fixed grayscale inversion > .04, constant denominator",
                  oob="zero padding, fixed denominator, explicit quadratic excess penalty; no query dropping")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "affine", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--method", choices=("radial", "analytic", "f1", "f2"), default="radial")
    p.add_argument("--patch-cells",type=int,default=8)
    p.add_argument("--f2-accepted-gain",type=float,default=1.)
    p.add_argument("--loss", choices=("mind", "local_ncc"), default="mind")
    p.add_argument("--grid-side", type=int, default=257)
    p.add_argument("--image-side", type=int, default=512)
    p.add_argument("--levels", type=int, nargs="+", default=[17,33,65,129,257])
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=.004)
    p.add_argument("--lr-calibration", choices=("physical", "edge"), default="physical")
    p.add_argument("--strain-weight", type=float, default=.05)
    p.add_argument("--oob-weight", type=float, default=1.)
    p.add_argument("--minimum-jacobian", type=float, default=.001)
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    report = optimize(p.parse_args())
    print(json.dumps({k: report[k] for k in ("initial", "final", "optimize_seconds", "evaluations", "gradient_steps")}))


if __name__ == "__main__":
    main()
