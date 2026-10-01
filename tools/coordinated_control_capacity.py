"""Bounded CPU R1/R2 controls on the SAME independently generated known maps.

All methods start at identity and optimize target MSE plus cumulative strain.
The trial-evaluation budget is matched, not the number of scalar parameters or
the number of geometry passes: F1/F2 update two components together, whereas
coordinated stages alternate axes. Targets are oracle evidence, not inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update, interpolate_proposal
from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, StaggeredPatchQ1Layer,
    q1_corner_determinants, validate_q1_map,
)
from tools.coordinated_geometry_pilot import reference_grid, target_map, derivative_rmse


def cumulative_strain(vertices, reference):
    residual = vertices-reference
    side = vertices.shape[1]
    return .5*sum((torch.diff(residual, dim=axis)*(side-1)).square().sum(-1).mean()
                  for axis in (1, 2))


def objective(vertices, reference, target, weight):
    mse = (vertices-target).square().mean()
    strain = cumulative_strain(vertices, reference)
    return mse+weight*strain, mse, strain


def f2_coverage(layer, side, dtype):
    """Count actual staggered patch interiors, including seams and edge rows."""
    coverage = torch.zeros(side*side, dtype=dtype, device=layer.passes[0].interior_ids.device)
    for patch_pass in layer.passes:
        coverage.index_add_(0, patch_pass.interior_ids, torch.ones_like(
            patch_pass.interior_ids, dtype=dtype))
    coverage = coverage.reshape(side, side)
    if not bool((coverage[1:-1, 1:-1] > 0).all()):
        raise RuntimeError("staggered F2 leaves an uncalibratable interior vertex")
    return coverage


def decode(layer, method, anchor, coefficients, coverage=None):
    side = anchor.shape[1]
    coarse = F.pad(coefficients, (1, 1, 1, 1))
    if method in ("radial", "analytic"):
        return layer(anchor, interpolate_proposal(coarse[:, 0], (side, side)), validate=False).vertices
    dense = F.interpolate(coarse, size=(side, side), mode="bilinear", align_corners=True)
    if method == "f1":
        # Identity JVP = 8*h*logit. Current-edge units may change at later
        # anchors; that difference belongs to the existing F1 mechanism.
        logits = dense*(side-1)/8
        return layer(anchor, logits.permute(0, 2, 3, 1)[:, 1:-1, 1:-1])
    nominal = .5*layer.passes[0].patch_cells/(side-1)
    logits = dense/(nominal*coverage.clamp_min(1)[None, None])
    logits = logits.permute(0, 2, 3, 1)[:, 1:-1, 1:-1]
    return layer(anchor, tuple(logits for _ in layer.passes))


def make_layer(method, side, direction, margin, patch_cells, f2_accepted_gain=1.):
    if method in ("radial", "analytic"):
        return CoordinatedQ1Update(direction, mode=method, minimum_jacobian=margin)
    if method == "f1":
        return AdaptiveSoftRadialQ1Relaxation(side, raw_span=8., safety_fraction=.75,
                                            minimum_jacobian=margin)
    return StaggeredPatchQ1Layer(side, patch_cells, raw_span=.5,
                               minimum_jacobian=margin, safety_fraction=.85,
                               accepted_gain=f2_accepted_gain)


def check_identity_calibration(reference, level, patch_cells, f2_accepted_gain=1.):
    side = reference.shape[1]
    torch.manual_seed(177+side)
    tangent = torch.randn(1, 2, level-2, level-2, dtype=reference.dtype)
    dense = F.interpolate(F.pad(tangent, (1,1,1,1)), size=(side, side),
                          mode="bilinear", align_corners=True).permute(0,2,3,1)
    findings = {}
    for method in ("f1", "f2"):
        layer = make_layer(method, side, None, .001, patch_cells, f2_accepted_gain)
        coverage = f2_coverage(layer, side, reference.dtype) if method == "f2" else None
        # Keep decode's nominal unchanged for external callers: accepted_gain
        # multiplies the identity JVP, so its input coefficients compensate here.
        def physical_decode(c):
            return decode(layer, method, reference,
                          c/f2_accepted_gain if method=="f2" else c, coverage)
        zero = torch.zeros_like(tangent)
        _, jvp = torch.autograd.functional.jvp(
            physical_decode, zero, tangent)
        # Separate two-sided numerical probe catches errors in autograd scaling.
        error = float((jvp-dense).abs().max())
        numerical_errors = []
        for step in (1e-8,1e-9,1e-10):
            numerical = (physical_decode(step*tangent)
                         -physical_decode(-step*tangent))/(2*step)
            numerical_errors.append({"step":step,"maximum_error":float((numerical-dense).abs().max())})
        numerical_error = numerical_errors[-1]["maximum_error"]
        if error > 1e-12 or numerical_error > 5e-6:
            raise RuntimeError(f"{method} identity calibration failed: {error}, {numerical_error}")
        if method=="f1" and numerical_errors[-1]["maximum_error"]>=numerical_errors[0]["maximum_error"]:
            raise RuntimeError("F1 physical finite-difference probe did not converge before roundoff")
        findings[method] = {"jvp_maximum_error": error,
                            "numerical_maximum_error": numerical_error,
                            "numerical_convergence":numerical_errors,
                            "identity_physical_proposal_rms": float(dense.square().mean().sqrt())}
        if coverage is not None:
            findings[method]["accepted_gain"] = f2_accepted_gain
            interior = coverage[1:-1, 1:-1]
            findings[method]["coverage_counts"] = {str(int(count)): int((interior==count).sum())
                                                    for count in torch.unique(interior)}
            findings[method]["sequential_passes_per_evaluation"] = len(layer.passes)
    return findings


def fit(reference, target, method, weight, levels, args):
    current = reference.clone()
    initial_total, initial_mse, _ = objective(current, reference, target, weight)
    accepted_total = float(initial_total)
    best_rmse = float(initial_mse.sqrt())
    initial_rmse = best_rmse
    qref = q1_corner_determinants(reference)
    evaluations, gradients, rejected_trials, stage_count = 0, 0, 0, 0
    optimizer_setup_seconds = []
    trace, accepted = [], []
    thresholds = {str(value): None for value in args.rmse_thresholds}
    started = time.perf_counter()
    components = 1 if method in ("radial", "analytic") else 2
    while evaluations < args.evaluations:
        if time.perf_counter()-started > args.case_seconds:
            break
        level_index = (stage_count//2 if components==1 else stage_count)%len(levels)
        level = levels[level_index]
        direction = ((1.,0.), (0.,1.))[stage_count%2] if components==1 else None
        anchor = current.detach()
        anchor_total = accepted_total
        coefficients = torch.nn.Parameter(torch.zeros(1, components, level-2, level-2,
                                                       dtype=reference.dtype))
        physical_rate = args.learning_rate*(levels[0]-1)/(level-1)
        setup_tick = time.perf_counter()
        optimizer = torch.optim.Adam([coefficients], lr=physical_rate)
        optimizer_setup_seconds.append(time.perf_counter()-setup_tick)
        layer = make_layer(method, reference.shape[1], direction, args.minimum_jacobian,
                           args.patch_cells, args.f2_accepted_gain)
        coverage = f2_coverage(layer, reference.shape[1], reference.dtype) if method=="f2" else None
        best_map, best_total = anchor, anchor_total
        remaining = args.evaluations-evaluations
        stage_gradients = min(args.inner_steps, remaining-1)
        stage_failures = 0
        stage_step0_error = None
        for step in range(stage_gradients+1):
            optimizer.zero_grad(set_to_none=True)
            physical_coefficients = coefficients/args.f2_accepted_gain if method=="f2" else coefficients
            candidate = decode(layer, method, anchor, physical_coefficients, coverage)
            total, mse, strain = objective(candidate, reference, target, weight)
            evaluations += 1
            minimum = (q1_corner_determinants(candidate.double())/qref.double()).min()
            legal = bool(torch.isfinite(candidate).all() and minimum>args.minimum_jacobian
                         and torch.isfinite(total))
            value = float(total.detach())
            if step==0:
                stage_step0_error = abs(value-anchor_total)
            if not legal:
                rejected_trials += 1
                stage_failures += 1
                break
            if value < best_total:
                best_map, best_total = candidate.detach().clone(), value
            trace.append({"trial": evaluations, "stage": stage_count, "level": level,
                          "direction": direction, "step": step, "total": value,
                          "rmse": float(mse.detach().sqrt()), "strain": float(strain.detach()),
                          "minimum_normalized_corner": float(minimum.detach())})
            if step < stage_gradients:
                total.backward()
                if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                    rejected_trials += 1
                    stage_failures += 1
                    break
                optimizer.step()
                gradients += 1
        if not validate_q1_map(best_map, reference)["valid"]:
            raise RuntimeError("accepted full-objective map failed output check")
        current, accepted_total = best_map, best_total
        rmse = float((current-target).square().mean().sqrt())
        best_rmse = min(best_rmse, rmse)
        seconds = time.perf_counter()-started
        for threshold in args.rmse_thresholds:
            if thresholds[str(threshold)] is None and rmse <= threshold:
                thresholds[str(threshold)] = seconds
        accepted.append({"stage": stage_count, "level": level, "direction": direction,
                         "anchor_total": anchor_total, "accepted_total": accepted_total,
                         "accepted_rmse": rmse, "physical_learning_rate": physical_rate,
                         "step0_anchor_error": stage_step0_error, "failures": stage_failures,
                         "evaluated_final_optimizer_update": stage_failures==0,
                         "seconds": seconds})
        stage_count += 1
    elapsed = time.perf_counter()-started
    final_total, final_mse, final_strain = objective(current, reference, target, weight)
    return {"initial_rmse": initial_rmse, "initial_total": float(initial_total),
            "rmse": float(final_mse.sqrt()), "best_accepted_rmse": best_rmse,
            "h1_derivative_rmse": derivative_rmse(current,target),
            "final_total": float(final_total), "final_strain": float(final_strain),
            "trial_evaluations": evaluations, "objective_evaluations_including_initial_final": evaluations+2,
            "gradient_updates": gradients, "stages": stage_count,
            "parameter_components_per_stage": components, "rejected_trials": rejected_trials,
            "seconds": elapsed, "budget_exhausted": evaluations==args.evaluations,
            "optimizer_setup_seconds":sum(optimizer_setup_seconds),
            "first_optimizer_setup_seconds":optimizer_setup_seconds[0] if optimizer_setup_seconds else None,
            "seconds_excluding_optimizer_setup":elapsed-sum(optimizer_setup_seconds),
            "time_to_rmse": thresholds, "valid": validate_q1_map(current,reference)["valid"],
            "minimum_normalized_corner": float((q1_corner_determinants(current)/qref).min()),
            "maximum_step0_anchor_error": max(item["step0_anchor_error"] for item in accepted),
            "accepted_objective_increases": sum(item["accepted_total"]>item["anchor_total"] for item in accepted),
            "accepted_stages": accepted, "trace": trace}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides",type=int,nargs="+",default=[33,65])
    parser.add_argument("--methods",nargs="+",choices=["radial","analytic","f1","f2"],default=["radial","analytic","f1","f2"])
    parser.add_argument("--strain-weights",type=float,nargs="+",default=[.05])
    parser.add_argument("--evaluations",type=int,default=120)
    parser.add_argument("--inner-steps",type=int,default=5)
    parser.add_argument("--learning-rate",type=float,default=.004)
    parser.add_argument("--minimum-jacobian",type=float,default=.001)
    parser.add_argument("--patch-cells",type=int,default=8)
    parser.add_argument("--f2-accepted-gain",type=float,default=1.,
                        help="F2 accepted step gain; coefficients are identity-JVP calibrated")
    parser.add_argument("--rmse-thresholds",type=float,nargs="+",default=[.01,.005,.001])
    parser.add_argument("--case-seconds",type=float,default=15.)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not 0 < args.f2_accepted_gain <= 1:
        parser.error("--f2-accepted-gain must be in (0,1]")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    rows,calibrations=[],{}
    for side in args.sides:
        reference=reference_grid(side)
        levels=[9,17,side,17,9]
        calibrations[str(side)]=check_identity_calibration(reference,9,args.patch_cells,args.f2_accepted_gain)
        for name in ("wide_shear","local_rotation","coarse_fine"):
            target=target_map(reference,name)
            if not validate_q1_map(target,reference)["valid"]:
                rows.append({"side":side,"target":name,"target_valid":False,"reason":"outside declared digital space"})
                continue
            for weight in args.strain_weights:
                for method in args.methods:
                    report=fit(reference,target,method,weight,levels,args)
                    row={"side":side,"target":name,"method":method,"strain_weight":weight,
                         "target_valid":True,"levels":levels,**report}
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k not in ("accepted_stages","trace")}),flush=True)
    payload={"question":"R1/R2 matched trial budget and common cumulative objective; not real registration",
             "dtype":"float64","interpolation":"Q1","boundary":"fixed",
             "budget_scope":"same trial evaluations, differing components and geometry passes; initial/final included separately",
             "timing_scope":"CPU case optimization includes all stage setup; optimizer setup also reported separately; import/loading and identity calibration excluded",
             "runtime":{"torch_version":torch.__version__,"torch_threads":torch.get_num_threads(),
                        "device":"cpu","cuda_used":False},
             "configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             "identity_unit_calibration":calibrations,"results":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
