"""Held-out A4 refinement using basis-free local brightness-constancy hints.

The pretrained encoder is frozen. Hyperparameters are chosen only by train
image MSE, then tested on a disjoint image split. This is not new training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_image_to_latent import _minimum_area_ratio
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import HierarchicalConvexQuadLocalLayer, evaluate_structured_p1_with_jacobian, local_photometric_logits, spectralize_bounded_logits
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--min-train-area", type=float, default=0.0)
    parser.add_argument("--max-passes", type=int, default=1)
    parser.add_argument("--fixed-window", type=int, default=None)
    parser.add_argument("--fixed-ridge", type=float, default=None)
    parser.add_argument("--fixed-gain", type=float, default=None)
    parser.add_argument("--floor-fraction", type=float, default=0.0)
    parser.add_argument("--sine-mode-counts", default="", help="comma-separated retained joint sine-mode counts")
    parser.add_argument("--max-train-mu", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prior = state["args"]
    if prior["method"] != "A4" or prior["side"] != 257 or prior["target_family"] != "high32":
        raise ValueError("requires the image-only A4 high32 257 checkpoint")
    side = prior["side"]
    image_side = prior["image_side"]
    encoder = ConvexQuadLocalImageEncoder(
        side, width=prior["a2_width"], head_mode=prior["a2_head_mode"], body_mode=prior["a2_body_mode"]
    )
    encoder.load_state_dict(state["encoder"])
    encoder = encoder.to(device).eval()
    decoder = HierarchicalConvexQuadLocalLayer(
        side, motion_mode="radial", floor_fraction=args.floor_fraction
    ).to(device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(side - 1, side - 1), height=image_side, width=image_side
    )
    table.prepare(device=device, dtype=torch.float32)
    train = tuple(value.to(device) for value in make_dataset(
        args.train_count, image_side, 55101, return_coefficients=True, target_family="high32"
    ))
    test = tuple(value.to(device) for value in make_dataset(
        args.test_count, image_side, 99317, return_coefficients=True, target_family="high32"
    ))
    line = torch.linspace(0.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    high_basis = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
    projection_denominator = high_basis.square().sum()
    source = torch.stack((xx, yy), dim=-1)
    mesh = structured_rectangle(side - 1, side - 1)
    source_vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32, device=device)
    face_centroids = source_vertices[
        torch.tensor(mesh.faces.copy(), dtype=torch.long, device=device)
    ].mean(dim=1)

    @torch.no_grad()
    def spectralized_hint(hint: torch.Tensor, mode_count: int | None) -> torch.Tensor:
        if mode_count is None:
            return hint
        return spectralize_bounded_logits(
            hint, side=side, raw_span=decoder.local.raw_span, count=mode_count
        )

    def evaluate(
        dataset, window: int, ridge: float, gain: float, passes: int = 1,
        mode_count: int | None = None, face_geometry: bool = False,
        intrinsic_geometry: bool = False,
    ) -> dict[str, object]:
        image_losses = []
        map_losses = []
        area = float("inf")
        projected = []
        timings = []
        mu_squared_sum = 0.0
        mu_sampled_squared_sum = 0.0
        maximum_mu = 0.0
        fixed_all, moving_all, target_all, coefficients_all = dataset
        for start in range(0, len(fixed_all), args.batch):
            stop = min(start + args.batch, len(fixed_all))
            fixed = fixed_all[start:stop]
            moving = moving_all[start:stop]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            root, levels, learned_local = encoder(torch.cat((fixed, moving), dim=1))
            base = decoder.base(root, levels)
            area_floor = decoder.local.compute_area_floor(base) if args.floor_fraction else None
            hint = local_photometric_logits(
                fixed, moving, base, window=window, ridge=ridge, raw_span=decoder.local.raw_span
            )
            hint = spectralized_hint(hint, mode_count)
            mapped = decoder.local(base, learned_local + gain * hint, area_floor=area_floor)
            current_area = _minimum_area_ratio(mapped)
            for _ in range(1, passes):
                hint = local_photometric_logits(
                    fixed, moving, mapped, window=window, ridge=ridge, raw_span=decoder.local.raw_span
                )
                hint = spectralized_hint(hint, mode_count)
                mapped = decoder.local(mapped, gain * hint, area_floor=area_floor)
                current_area = min(current_area, _minimum_area_ratio(mapped))
            dense = table.interpolate(mapped.reshape(stop - start, -1, 2))
            warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - began)
            image_losses.append((warped - fixed).square().mean(dim=(1, 2, 3)))
            map_losses.append((dense - target_all[start:stop]).square().mean(dim=(1, 2, 3)))
            area = min(area, current_area)
            projection = ((mapped - source[None]) * high_basis[None, :, :, None]).sum(dim=(1, 2)) / projection_denominator
            projected.extend(projection.tolist())
            if face_geometry or intrinsic_geometry:
                centroids = face_centroids[None].expand(stop - start, -1, -1)
                _, predicted_jacobian = evaluate_structured_p1_with_jacobian(mapped, centroids)
                predicted_mu = _mu(predicted_jacobian)
                maximum_mu = max(maximum_mu, float(predicted_mu.abs().max()))
            if face_geometry:
                _, target_jacobian = _target_on_faces(centroids, coefficients_all[start:stop], 32)
                sampled_target, _ = _target_on_faces(
                    source_vertices[None].expand(stop - start, -1, -1), coefficients_all[start:stop], 32
                )
                _, sampled_jacobian = evaluate_structured_p1_with_jacobian(
                    sampled_target.reshape(stop - start, side, side, 2), centroids
                )
                target_mu, sampled_mu = (_mu(jacobian) for jacobian in (target_jacobian, sampled_jacobian))
                mu_squared_sum += float((predicted_mu - target_mu).abs().square().sum())
                mu_sampled_squared_sum += float((sampled_mu - target_mu).abs().square().sum())
        images = torch.cat(image_losses)
        maps = torch.cat(map_losses)
        query_map_rmse = maps.mean().sqrt().item()
        result = {
            "image_mse": images.mean().item(),
            "query_map_rmse": query_map_rmse if math.isfinite(query_map_rmse) else None,
            "minimum_signed_area_ratio": area,
            "projected_fine_amplitudes": projected,
            "true_fine_amplitudes": coefficients_all[:, 2].tolist(),
            "mean_batch_forward_seconds": statistics.mean(timings),
        }
        if face_geometry:
            total_faces = len(fixed_all) * len(face_centroids)
            result.update({
                "face_beltrami_rmse": math.sqrt(mu_squared_sum / total_faces),
                "sampled_target_P1_beltrami_discretization_rmse": math.sqrt(mu_sampled_squared_sum / total_faces),
            })
        if face_geometry or intrinsic_geometry:
            result["maximum_predicted_beltrami_modulus"] = maximum_mu
        return result

    began = time.perf_counter()
    candidates = []
    windows = (args.fixed_window,) if args.fixed_window is not None else (1, 3, 5)
    ridges = (args.fixed_ridge,) if args.fixed_ridge is not None else (0.1, 1.0, 10.0)
    gains = (args.fixed_gain,) if args.fixed_gain is not None else (0.25, 0.5, 1.0)
    mode_counts = (
        tuple(int(value) for value in args.sine_mode_counts.split(",")) if args.sine_mode_counts else (None,)
    )
    for window in windows:
        for ridge in ridges:
            for gain in gains:
                for mode_count in mode_counts:
                    for passes in range(1, args.max_passes + 1):
                        measured = evaluate(
                            train, window, ridge, gain, passes=passes, mode_count=mode_count,
                            intrinsic_geometry=args.max_train_mu is not None,
                        )
                        candidates.append({
                            "window": window, "ridge": ridge, "gain": gain,
                            "sine_modes": mode_count, "passes": passes, "train": {
                                key: value for key, value in measured.items()
                                if key not in ("projected_fine_amplitudes", "true_fine_amplitudes")
                            }
                        })
    baseline_train = evaluate(train, 1, 1.0, 0.0)
    baseline_test = evaluate(test, 1, 1.0, 0.0)
    eligible = [
        candidate for candidate in candidates
        if candidate["train"]["minimum_signed_area_ratio"] >= args.min_train_area
        and (
            args.max_train_mu is None
            or candidate["train"]["maximum_predicted_beltrami_modulus"] <= args.max_train_mu
        )
    ]
    if not eligible:
        raise RuntimeError("no training candidate met the declared area threshold")
    chosen = min(eligible, key=lambda item: item["train"]["image_mse"])
    heldout = evaluate(
        test, chosen["window"], chosen["ridge"], chosen["gain"],
        passes=chosen["passes"], mode_count=chosen["sine_modes"], face_geometry=True,
    )
    print(json.dumps({
        "question": "basis_free_photometric_hint_on_frozen_a4_encoder",
        "checkpoint": args.checkpoint,
        "selection": "minimum training image MSE only",
        "minimum_train_area_constraint": args.min_train_area,
        "maximum_train_beltrami_modulus_constraint": args.max_train_mu,
        "floor_fraction": args.floor_fraction,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "batch": args.batch,
        "control_side": side,
        "image_side": image_side,
        "device": str(device),
        "dtype": "float32",
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "selected": chosen,
        "selected_heldout": heldout,
        "candidate_train_summary": candidates,
        "total_seconds": time.perf_counter() - began,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
