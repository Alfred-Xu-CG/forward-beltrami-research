"""Frozen A6 image layer with optional spectral safe residual-feedback passes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_image_to_latent import _minimum_area_ratio
from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    HierarchicalConvexQuadLocalLayer, SafeColoredVertexRelaxation,
    SafeColoredQCRadialRelaxation,
    evaluate_structured_p1_with_jacobian, local_photometric_logits,
    spectralize_bounded_logits,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--extra-passes", default="0,1,2")
    parser.add_argument("--extra-gains", default="0.25,0.5,1")
    parser.add_argument("--extra-sine-modes", type=int, default=16)
    parser.add_argument("--floor-fraction", type=float, default=0.2)
    parser.add_argument("--qc-cap", type=float, default=None)
    parser.add_argument("--min-train-area", type=float, default=0.05)
    parser.add_argument("--max-train-mu", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    extra_passes = tuple(int(value) for value in args.extra_passes.split(","))
    extra_gains = tuple(float(value) for value in args.extra_gains.split(","))
    if not extra_passes or min(extra_passes) < 0 or not extra_gains or min(extra_gains) <= 0:
        raise ValueError("extra pass counts and gains must be nonnegative/positive")
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    settings = state["args"]
    if settings["method"] != "A6" or settings["target_family"] != "high32":
        raise ValueError("expected high32 A6 checkpoint")
    side, image_side = settings["side"], settings["image_side"]
    encoder = ConvexQuadLocalImageEncoder(
        side, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"], body_mode=settings["a2_body_mode"],
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    decoder = HierarchicalConvexQuadLocalLayer(side, motion_mode="radial").to(device)
    refiner = (
        SafeColoredVertexRelaxation(side, motion_mode="radial",
                                    floor_fraction=args.floor_fraction)
        if args.qc_cap is None else SafeColoredQCRadialRelaxation(
            side, qc_cap=args.qc_cap, floor_fraction=args.floor_fraction)
    ).to(device)
    mesh = structured_rectangle(side - 1, side - 1)
    vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces.copy(), dtype=torch.int64, device=device)
    centroids = vertices[faces].mean(dim=1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=image_side, width=image_side)
    table.prepare(device=device, dtype=torch.float32)
    train = tuple(value.to(device) for value in make_dataset(
        args.train_count, image_side, 55101, target_family="high32", return_coefficients=True,
    ))
    test = tuple(value.to(device) for value in make_dataset(
        args.test_count, image_side, args.test_seed, target_family="high32", return_coefficients=True,
    ))
    source_grid = vertices.reshape(side, side, 2)
    high_basis = torch.sin(64 * math.pi * source_grid[..., 0]) * torch.sin(64 * math.pi * source_grid[..., 1])
    def decode(fixed, moving, extra_count: int, gain: float):
        latent = encoder(torch.cat((fixed, moving), dim=1))
        base = decoder.base(latent[0], latent[1])
        hint = local_photometric_logits(
            fixed, moving, base,
            window=settings["hint_window"], ridge=settings["hint_ridge"],
            raw_span=decoder.local.raw_span,
        )
        hint = spectralize_bounded_logits(
            hint, side=side, raw_span=decoder.local.raw_span,
            count=settings["hint_sine_modes"],
        )
        mapped = decoder.local(base, latent[2] + settings["hint_gain"] * hint)
        stages = [mapped]
        floor = refiner.compute_area_floor(mapped) if extra_count and refiner.floor_fraction else None
        for _ in range(extra_count):
            next_hint = local_photometric_logits(
                fixed, moving, mapped, window=settings["hint_window"],
                ridge=settings["hint_ridge"], raw_span=refiner.raw_span,
            )
            next_hint = spectralize_bounded_logits(
                next_hint, side=side, raw_span=refiner.raw_span,
                count=args.extra_sine_modes,
            )
            mapped = refiner(mapped, gain * next_hint, area_floor=floor)
            stages.append(mapped)
        query = table.interpolate(mapped.reshape(len(fixed), -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return mapped, query, warped, stages
    @torch.no_grad()
    def evaluate(dataset, extra_count: int, gain: float, *, face_geometry: bool) -> dict[str, object]:
        fixed_all, moving_all, target_all, coefficients_all = dataset
        image_total, map_total, mu_total = 0.0, 0.0, 0.0
        minimum = float("inf"); maximum_mu = 0.0
        projections = []
        times = []
        for start in range(0, len(fixed_all), args.batch):
            stop = min(start + args.batch, len(fixed_all))
            if device.type == "cuda": torch.cuda.synchronize(device)
            began = time.perf_counter()
            mapped, query, warped, stages = decode(fixed_all[start:stop], moving_all[start:stop], extra_count, gain)
            if device.type == "cuda": torch.cuda.synchronize(device)
            times.append(time.perf_counter() - began)
            current = stop - start
            image_total += (warped - fixed_all[start:stop]).square().mean().item() * current
            map_total += (query - target_all[start:stop]).square().mean().item() * current
            for stage in stages:
                minimum = min(minimum, _minimum_area_ratio(stage))
            projected = ((mapped - source_grid) * high_basis[None, :, :, None]).sum(dim=(1, 2)) / high_basis.square().sum()
            projections.extend(projected.tolist())
            face_query = centroids[None].expand(current, -1, -1)
            _, jacobian = evaluate_structured_p1_with_jacobian(mapped, face_query)
            predicted_mu = _mu(jacobian)
            maximum_mu = max(maximum_mu, float(predicted_mu.abs().max()))
            if face_geometry:
                _, target_jacobian = _target_on_faces(face_query, coefficients_all[start:stop], 32)
                mu_total += (predicted_mu - _mu(target_jacobian)).abs().square().mean().item() * current
        result = {
            "image_mse": image_total / len(fixed_all),
            "query_map_rmse": math.sqrt(map_total / len(fixed_all)),
            "minimum_all_stage_signed_area_ratio": minimum,
            "maximum_predicted_beltrami_modulus": maximum_mu,
            "projected_fine_amplitudes": projections,
            "true_fine_amplitudes": coefficients_all[:, 2].tolist(),
            "mean_batch_forward_seconds": statistics.mean(times),
        }
        if face_geometry:
            result["face_beltrami_rmse"] = math.sqrt(mu_total / len(fixed_all))
        return result
    candidates = []
    for count in extra_passes:
        for gain in ((extra_gains[0],) if count == 0 else extra_gains):
            result = evaluate(train, count, gain, face_geometry=False)
            candidates.append({"extra_passes": count, "gain": gain,
                               "train": {key: value for key, value in result.items()
                                         if key not in ("projected_fine_amplitudes", "true_fine_amplitudes")}})
    eligible = [candidate for candidate in candidates
                if candidate["train"]["minimum_all_stage_signed_area_ratio"] >= args.min_train_area
                and candidate["train"]["maximum_predicted_beltrami_modulus"] <= args.max_train_mu]
    selected = min(eligible, key=lambda value: value["train"]["image_mse"]) if eligible else None
    baseline_test = evaluate(test, 0, extra_gains[0], face_geometry=True)
    selected_test = evaluate(test, selected["extra_passes"], selected["gain"], face_geometry=True) if selected else None
    vjp = None
    if selected:
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        forwards, backwards = [], []
        for repeat in range(4):
            encoder.zero_grad(set_to_none=True)
            if device.type == "cuda": torch.cuda.synchronize(device)
            began = time.perf_counter()
            _, _, warped, _ = decode(test[0][:args.batch], test[1][:args.batch], selected["extra_passes"], selected["gain"])
            loss = (warped - test[0][:args.batch]).square().mean()
            if device.type == "cuda": torch.cuda.synchronize(device)
            middle = time.perf_counter()
            loss.backward()
            if device.type == "cuda": torch.cuda.synchronize(device)
            ended = time.perf_counter()
            if repeat:
                forwards.append(middle - began)
                backwards.append(ended - middle)
        vjp = {
            "median_full_forward_seconds": statistics.median(forwards),
            "median_full_vjp_seconds": statistics.median(backwards),
            "maximum_encoder_gradient": max(float(parameter.grad.abs().max()) for parameter in encoder.parameters() if parameter.grad is not None),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
    print(json.dumps({
        "method": "frozen_A6_with_spectral_qc_safe_feedback" if args.qc_cap is not None else
                  "frozen_A6_with_spectral_safe_residual_feedback",
        "checkpoint": args.checkpoint,
        "control_side": side,
        "control_vertices": side**2,
        "control_faces": 2 * (side - 1)**2,
        "image_side": image_side,
        "image_queries": image_side**2,
        "train_count": args.train_count,
        "train_seed": 55101,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "batch": args.batch,
        "device": str(device),
        "dtype": "float32",
        "extra_sine_modes": args.extra_sine_modes,
        "floor_fraction": args.floor_fraction,
        "qc_cap": args.qc_cap,
        "minimum_train_area_constraint": args.min_train_area,
        "maximum_train_beltrami_modulus_constraint": args.max_train_mu,
        "selection": "minimum_train_image_mse_among_area_and_mu_eligible_candidates",
        "candidates": candidates,
        "selected": selected,
        "baseline_test": baseline_test,
        "selected_test": selected_test,
        "selected_full_vjp": vjp,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
