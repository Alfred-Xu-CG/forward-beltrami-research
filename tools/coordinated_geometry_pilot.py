"""R1/R2 CPU capacity pilot; target vertices are oracle evidence, not registration.

Run: PYTHONPATH=src python tools/coordinated_geometry_pilot.py --sides 33 65
Targets are sampled analytic maps generated independently of the decoder.
No case filtering: invalid targets are recorded and excluded from fitting with
an explicit reason, while every valid target/method run stays in the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update, interpolate_proposal
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants, validate_q1_map


def reference_grid(side, dtype=torch.float64):
    axis = torch.linspace(0, 1, side, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1)[None]


def target_map(reference, name):
    x, y = reference[..., 0], reference[..., 1]
    envelope = (torch.sin(torch.pi * x) * torch.sin(torch.pi * y)).square()
    if name == "wide_shear":
        displacement = torch.stack((0.12 * envelope, torch.zeros_like(x)), dim=-1)
    elif name == "local_rotation":
        dx, dy = x - 0.5, y - 0.5
        radius2 = dx.square() + dy.square()
        angle = 0.7 * (1 - radius2 / 0.44**2).clamp_min(0).pow(3)
        tx = torch.cos(angle) * dx - torch.sin(angle) * dy
        ty = torch.sin(angle) * dx + torch.cos(angle) * dy
        displacement = torch.stack((tx - dx, ty - dy), dim=-1)
    elif name == "coarse_fine":
        displacement = torch.stack((0.09 * envelope,
                                    0.04 * torch.sin(2 * torch.pi * x) * envelope
                                    + 0.01 * torch.sin(8 * torch.pi * x) * envelope), dim=-1)
    else:
        raise ValueError(name)
    # Mathematical zero boundary written exactly in stored coordinates.
    displacement[:, 0] = displacement[:, -1] = 0
    displacement[:, :, 0] = displacement[:, :, -1] = 0
    return reference + displacement


def derivative_rmse(current, target):
    side = current.shape[1]
    differences = [(torch.diff(current - target, dim=axis) * (side - 1)).square().mean()
                   for axis in (1, 2)]
    return float(torch.stack(differences).mean().sqrt())


def fit_target(reference, target, mode, levels, cycles, inner_steps, learning_rate):
    current = reference.clone()
    initial = float((current - target).square().mean().sqrt())
    started = time.perf_counter()
    calls, stages, rejected = 0, 0, 0
    scales, active = [], []
    history = []
    for cycle in range(cycles):
        for level in levels:
            for direction in ((1., 0.), (0., 1.)):
                anchor = current.detach()
                coefficients = torch.zeros(1, level, level, dtype=current.dtype, requires_grad=True)
                optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
                layer = CoordinatedQ1Update(direction, mode=mode, minimum_jacobian=.001)
                best, best_loss = None, float((anchor - target).square().mean())
                for _ in range(inner_steps):
                    optimizer.zero_grad(set_to_none=True)
                    raw = interpolate_proposal(coefficients, current.shape[1:3], boundary="fixed")
                    result = layer(anchor, raw, reference=reference, validate=False)
                    loss = (result.vertices - target).square().mean()
                    loss.backward()
                    optimizer.step()
                    calls += 1
                    if float(loss.detach()) < best_loss:
                        best, best_loss = result, float(loss.detach())
                stages += 1
                if best is not None and bool((best.normalized_margin_min > 0).all()):
                    candidate = best.vertices.detach()
                    if validate_q1_map(candidate, reference)["valid"]:
                        current = candidate
                        scales.append(float(best.scale.detach().mean()))
                        active.append(float(best.gauge.detach().mean()))
                    else:
                        rejected += 1
                else:
                    rejected += 1
        history.append({"cycle": cycle + 1, "rmse": float((current-target).square().mean().sqrt()),
                        "seconds": time.perf_counter() - started})
    report = validate_q1_map(current, reference)
    qref = q1_corner_determinants(reference)
    return {"initial_rmse": initial, "rmse": float((current-target).square().mean().sqrt()),
            "h1_derivative_rmse": derivative_rmse(current, target),
            "seconds": time.perf_counter() - started, "evaluations": calls,
            "stages": stages, "rejected_stages": rejected, "valid": report["valid"],
            "minimum_normalized_corner": float((q1_corner_determinants(current)/qref).min()),
            "mean_scale": sum(scales)/len(scales) if scales else 1.,
            "maximum_gauge": max(active) if active else 0., "history": history}


def conditioning(reference):
    direction = (1., 0.)
    x, y = reference[..., 0], reference[..., 1]
    pattern = (torch.sin(torch.pi*x)*torch.sin(torch.pi*y)).square()
    pattern[:, 0] = pattern[:, -1] = 0
    pattern[:, :, 0] = pattern[:, :, -1] = 0
    unit_gauge = CoordinatedQ1Update(direction)(reference, pattern).gauge.item()
    rows = []
    for gauge in (0.1, 1., 10., 100.):
        for mode in ("radial", "analytic"):
            multiplier = torch.tensor(gauge/unit_gauge, dtype=reference.dtype, requires_grad=True)
            result = CoordinatedQ1Update(direction, mode=mode)(reference, multiplier*pattern)
            response = (result.amplitude*pattern).sum()/pattern.square().sum()
            gain = torch.autograd.grad(response, multiplier)[0].item()
            rows.append({"mode": mode, "proposal_gauge": gauge,
                         "scale": result.scale.item(), "radial_direction_gain": gain})
    return rows


def scale_calibration(reference):
    # Same physical coefficient perturbation at every level, evaluated through
    # the boundary mask. Also expose gradient support aggregation explicitly.
    rows = []
    for level in (5, 9, 17, reference.shape[1]):
        coefficients = torch.full((1, level, level), .001, dtype=reference.dtype, requires_grad=True)
        raw = interpolate_proposal(coefficients, reference.shape[1:3], boundary="fixed")
        layer = CoordinatedQ1Update()
        result = layer(reference, raw)
        objective = result.vertices[..., 0].mean()
        gradient = torch.autograd.grad(objective, coefficients)[0]
        rows.append({"level": level, "physical_proposal_rms": float(layer._mask(raw).square().mean().sqrt()),
                     "coefficient_gradient_l2": float(gradient.norm()),
                     "coefficient_gradient_sum": float(gradient.sum()), "scale": result.scale.item()})
    return rows


def oracle_and_margin_restriction(reference):
    target = target_map(reference, "wide_shear")
    amplitude = target[..., 0] - reference[..., 0]
    layer = CoordinatedQ1Update(minimum_jacobian=.001)
    gauge = layer(reference, amplitude).gauge
    inverse = amplitude / (1-gauge[:, None, None])
    encoded = layer(reference, inverse).vertices
    x, y = reference[..., 0], reference[..., 1]
    compression = reference.clone()
    compression[..., 0] += .154 * torch.sin(2*torch.pi*x) * torch.sin(torch.pi*y).square()
    compression[:, 0] = reference[:, 0]
    compression[:, -1] = reference[:, -1]
    compression[:, :, 0] = reference[:, :, 0]
    compression[:, :, -1] = reference[:, :, -1]
    minimum = float((q1_corner_determinants(compression)/q1_corner_determinants(reference)).min())
    return {"oracle_shear_maximum_coordinate_error": float((encoded-target).abs().max()),
            "compression_target_valid": validate_q1_map(compression, reference)["valid"],
            "compression_minimum_normalized_corner": minimum,
            "extra_margin_exclusion": {str(eta): minimum <= eta for eta in (0., .001, .05, .1)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", nargs="+", type=int, default=[33, 65])
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--inner-steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=.025)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows, calibration, gains, oracle = [], {}, {}, {}
    for side in args.sides:
        reference = reference_grid(side)
        gains[str(side)] = conditioning(reference)
        calibration[str(side)] = scale_calibration(reference)
        oracle[str(side)] = oracle_and_margin_restriction(reference)
        for name in ("wide_shear", "local_rotation", "coarse_fine"):
            target = target_map(reference, name)
            valid_target = validate_q1_map(target, reference)["valid"]
            target_minimum = float((q1_corner_determinants(target)/q1_corner_determinants(reference)).min())
            if not valid_target:
                rows.append({"side": side, "target": name, "target_valid": False,
                             "reason": "analytic target outside declared digital space"})
                continue
            for mode in ("radial", "analytic"):
                for schedule, levels in (("single", [side]*5), ("revisit", [9, 17, side, 17, 9])):
                    report = fit_target(reference, target, mode, levels, args.cycles,
                                        args.inner_steps, args.learning_rate)
                    row = {"side": side, "target": name, "mode": mode, "schedule": schedule,
                           "target_valid": True, "target_minimum_normalized_corner": target_minimum,
                           "coefficient_learning_rate_physical": args.learning_rate, **report}
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k != "history"}), flush=True)
    payload = {"question": "R1/R2 known-map capacity and entry conditioning; no real-image claim",
               "dtype": "float64", "boundary": "fixed", "interpolation": "Q1",
               "minimum_jacobian": .001, "results": rows,
               "conditioning": gains, "scale_calibration": calibration, "oracle": oracle}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"conditioning": gains, "scale_calibration": calibration, "oracle": oracle}), flush=True)


if __name__ == "__main__":
    main()
