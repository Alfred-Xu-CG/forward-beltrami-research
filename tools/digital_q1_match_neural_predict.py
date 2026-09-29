"""Predict one saved positive-factor P1 map from image matches, no labels."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_match_neural_decoder import MatchSpatialDecoder, match_rmse


def predict(checkpoint: Path, matches: Path, affine: Path, output: Path,
            *, device_name: str, repeats: int = 10) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("prediction output already exists")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    device = torch.device(device_name)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MatchSpatialDecoder(update_sides=tuple(saved["update_sides"]),
                                final_side=int(saved["final_side"])).to(device)
    model.net.load_state_dict(saved["state_dict"])
    model.eval()
    data = json.loads(matches.read_text(encoding="utf-8"))
    if data["status"] != "ok" or data["ransac_inliers"] < 8:
        raise ValueError("need at least eight image-derived matches")
    declared_affine = data.get("affine_map")
    if declared_affine is not None and Path(declared_affine).name != affine.name:
        raise ValueError("match coordinates and affine archive use different declared frames")
    source = torch.tensor(data["source_points_unit"], device=device, dtype=torch.float32)
    target = torch.tensor(data["target_points_unit"], device=device, dtype=torch.float32)
    with np.load(affine) as factor:
        matrix = np.asarray(factor["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(factor["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("invalid affine factor")
    a, b, c, d = (Fraction.from_float(float(item)) for item in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("affine factor must preserve orientation")
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    with torch.no_grad():
        for _ in range(3):
            predicted = model(source, target)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        forward = []
        for _ in range(repeats):
            synchronize()
            started = time.perf_counter()
            predicted = model(source, target)
            synchronize()
            forward.append(time.perf_counter() - started)
        check = validate_q1_map(predicted, identity_vertices(model.final_side, device=device))
        match_fit = float(match_rmse(predicted, source, target))
        identity_fit = float(match_rmse(identity_vertices(model.final_side, device=device), source, target))
        vertices = predicted.detach().cpu().numpy()
    forward_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = identity_vertices(model.final_side, device=device).cpu().numpy()
    np.savez_compressed(output, vertices=vertices, boundary_reference=reference,
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"] or not check["valid"]:
        raise RuntimeError("saved map did not pass topology audit")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vjp_times = []
    finite_vjp = True
    for _ in range(repeats):
        initial = source.detach().clone().requires_grad_(True)
        destination = target.detach().clone().requires_grad_(True)
        model.net.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        output_map = model(initial, destination)
        loss = match_rmse(output_map, initial, destination)
        loss.backward()
        synchronize()
        vjp_times.append(time.perf_counter() - started)
        finite_vjp &= (initial.grad is not None and destination.grad is not None
                       and bool(torch.isfinite(initial.grad).all())
                       and bool(torch.isfinite(destination.grad).all())
                       and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                               for p in model.net.parameters()))
    vjp_peak = None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    report = {
        "checkpoint": str(checkpoint), "matches": str(matches), "affine": str(affine),
        "match_declared_affine": declared_affine,
        "map": str(output), "device": device_name, "source_matches": len(source),
        "update_sides": model.update_sides, "final_side": model.final_side,
        "initial_affine_was_DHR_derived": True,
        "full_DHR_displacement_or_anatomical_landmarks_used_at_prediction": False,
        "train_case_count": len(saved["train_case_ids"]),
        "identity_match_rmse": identity_fit, "predicted_match_rmse": match_fit,
        "saved_binary_certificate": binary,
        "in_memory_validity": check,
        "forward_seconds_median": statistics.median(forward),
        "forward_plus_vjp_seconds_median": statistics.median(vjp_times),
        "forward_peak_allocated_bytes": forward_peak,
        "vjp_peak_allocated_bytes": vjp_peak,
        "finite_match_and_parameter_vjp": finite_vjp,
        "timing_excludes_external_matcher_affine_generation_io_certificate": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.matches, args.affine, args.output,
                     device_name=args.device, repeats=args.repeats)
    print(json.dumps({key: result[key] for key in (
        "source_matches", "identity_match_rmse", "predicted_match_rmse",
        "forward_seconds_median", "forward_plus_vjp_seconds_median",
        "finite_match_and_parameter_vjp",
    )}))


if __name__ == "__main__":
    main()
