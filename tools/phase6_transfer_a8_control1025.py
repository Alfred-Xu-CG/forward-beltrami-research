"""Transfer trained 257-side A8 encoder heads to a real 1025-side P1 layer."""

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
    SpectralSafeFeedbackLayer, evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-refiner", action="store_true")
    parser.add_argument("--feature-side", type=int, default=None)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    span = args.side - 1
    if args.side <= 257 or span & (span - 1):
        raise ValueError("expected a dyadic finer grid, e.g. 513 or 1025")
    device = torch.device(args.device)
    torch.manual_seed(20260923)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    settings = state["args"]
    if (settings["method"] != "A8" or settings["side"] != 257
            or settings["target_family"] != "high32"):
        raise ValueError("requires the trained 257-side high32 A8 encoder")
    encoder = ConvexQuadLocalImageEncoder(
        args.side, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"],
        body_mode=settings["a2_body_mode"],
    ).to(device)
    old_levels = len(ConvexQuadLocalImageEncoder(
        257, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"],
        body_mode=settings["a2_body_mode"]).base.latent_sides)
    loaded = encoder.load_state_dict(state["encoder"], strict=False)
    if loaded.unexpected_keys or any(
            not (".7." in key or ".8." in key) for key in loaded.missing_keys):
        raise RuntimeError(f"unexpected state transfer: {loaded}")
    decoder = SpectralSafeFeedbackLayer(
        args.side, initial_window=settings["hint_window"],
        initial_ridge=settings["hint_ridge"],
        initial_gain=settings["hint_gain"],
        initial_modes=settings["hint_sine_modes"],
        extra_passes=settings["extra_passes"],
        extra_gain=settings["extra_gain"],
        extra_modes=16,
        extra_qc_cap=settings["extra_qc_cap"],
        floor_fraction=settings["floor_fraction"],
        extra_checkpoint=args.checkpoint_refiner,
    ).to(device)
    image_side = 512
    train = tuple(value.to(device) for value in make_dataset(
        args.train_count, image_side, 55101, target_family="high32",
        return_coefficients=True))
    test = tuple(value.to(device) for value in make_dataset(
        args.test_count, image_side, 99317, target_family="high32",
        return_coefficients=True))
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=image_side, width=image_side)
    table.prepare(device=device, dtype=torch.float32)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    centroids = vertices[torch.tensor(mesh.faces.copy(), device=device)].mean(dim=1)
    feature_side = args.feature_side or args.side
    if feature_side not in (257,args.side):
        raise ValueError("feature-side must be the original 257 or target control side")

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def encode(fixed: torch.Tensor, moving: torch.Tensor):
        pair = torch.cat((fixed,moving),dim=1)
        if feature_side == args.side:
            return encoder(pair)
        coarse = encoder.base.body(F.interpolate(
            pair,size=(feature_side,feature_side),
            mode="bilinear",align_corners=True))
        root,levels = encoder.base.latents_from_features(coarse)
        local = F.interpolate(
            encoder.local_head(coarse),size=(args.side,args.side),
            mode="bilinear",align_corners=True)[:,:,1:-1,1:-1]
        return root,levels,local.permute(0,2,3,1)

    def forward(fixed: torch.Tensor, moving: torch.Tensor):
        latent = encode(fixed,moving)
        mapped = decoder(fixed, moving, latent)
        query = table.interpolate(mapped.reshape(mapped.shape[0], -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), mapped, query

    coarse_level_max_difference = None
    if feature_side == 257:
        reference_encoder = ConvexQuadLocalImageEncoder(
            257,width=settings["a2_width"],
            head_mode=settings["a2_head_mode"],
            body_mode=settings["a2_body_mode"]).to(device)
        reference_encoder.load_state_dict(state["encoder"])
        with torch.no_grad():
            reference = reference_encoder(torch.cat(
                (test[0][:1],test[1][:1]),dim=1))
            transferred = encode(test[0][:1],test[1][:1])
            errors = [(reference[0]-transferred[0]).abs().max()]
            for previous,current in zip(reference[1],transferred[1]):
                errors.extend((left-right).abs().max()
                              for left,right in zip(previous,current))
            coarse_level_max_difference = float(torch.stack(errors).max())
        del reference_encoder

    @torch.no_grad()
    def evaluate(dataset, count: int):
        rows = []
        for index in range(count):
            fixed, moving, target, coefficient = (
                item[index:index + 1] for item in dataset)
            loss, mapped, query = forward(fixed, moving)
            _, jacobian = evaluate_structured_p1_with_jacobian(
                mapped, centroids[None])
            _, target_jacobian = _target_on_faces(
                centroids[None], coefficient, 32)
            predicted_mu = _mu(jacobian)
            initial = decoder.initial_map(
                fixed, moving, encode(fixed,moving))
            _, initial_jacobian = evaluate_structured_p1_with_jacobian(
                initial, centroids[None])
            rows.append({
                "image_mse": float(loss),
                "query_map_mse": float((query - target).square().mean()),
                "face_beltrami_mse": float(
                    (predicted_mu - _mu(target_jacobian)).abs().square().mean()),
                "minimum_signed_area_ratio": _minimum_area_ratio(mapped),
                "maximum_beltrami_modulus": float(predicted_mu.abs().amax()),
                "initial_maximum_beltrami_modulus": float(_mu(initial_jacobian).abs().amax()),
            })
        return {
            "count": count,
            "image_mse": statistics.mean(row["image_mse"] for row in rows),
            "query_map_rmse": math.sqrt(statistics.mean(
                row["query_map_mse"] for row in rows)),
            "face_beltrami_rmse": math.sqrt(statistics.mean(
                row["face_beltrami_mse"] for row in rows)),
            "minimum_signed_area_ratio": min(
                row["minimum_signed_area_ratio"] for row in rows),
            "maximum_beltrami_modulus": max(
                row["maximum_beltrami_modulus"] for row in rows),
            "initial_cap_eligible_count": sum(
                row["initial_maximum_beltrami_modulus"] < settings["extra_qc_cap"]
                for row in rows),
            "samples": rows,
        }

    initial = evaluate(test, args.test_count)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(38819)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times, records = [], [], []
    minimum_train_area = math.inf
    start_all = time.perf_counter()
    repeats = max(3,args.steps) if args.steps == 0 else args.steps
    for step in range(repeats):
        draw = torch.randint(args.train_count, (args.batch,),
                             generator=generator).to(device)
        fixed, moving = train[0][draw], train[1][draw]
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        start = time.perf_counter()
        loss, mapped, _ = forward(fixed, moving)
        minimum_train_area = min(minimum_train_area, _minimum_area_ratio(mapped))
        synchronize()
        middle = time.perf_counter()
        loss.backward()
        synchronize()
        end = time.perf_counter()
        if not all(parameter.grad is None or torch.isfinite(
                parameter.grad).all() for parameter in encoder.parameters()):
            raise RuntimeError("nonfinite transferred A8 encoder VJP")
        if args.steps:
            optimizer.step()
        if step:
            forward_times.append(middle - start)
            backward_times.append(end - middle)
        if args.steps and (step == 0 or (step + 1) % 25 == 0
                           or step + 1 == args.steps):
            records.append({
                "step": step + 1,
                "sampled_image_mse_before_update": float(loss),
                "elapsed_seconds": time.perf_counter() - start_all,
            })
    training_seconds = time.perf_counter() - start_all
    peak = (torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None)
    final = evaluate(test,args.test_count) if args.steps else initial
    if args.save_state:
        saved_args = dict(settings)
        saved_args["side"] = args.side
        torch.save({"encoder": encoder.state_dict(), "args": saved_args},
                   args.save_state)
    print(json.dumps({
        "method": "A8_trained257_to_million_control_transfer",
        "control_side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": image_side,
        "image_queries": image_side ** 2,
        "old_level_count": old_levels,
        "new_level_count": len(encoder.base.latent_sides),
        "feature_side": feature_side,
        "coarse_level_max_difference": coarse_level_max_difference,
        "missing_state_keys": list(loaded.missing_keys),
        "steps": args.steps,
        "batch": args.batch,
        "device": str(device),
        "checkpoint_refiner": args.checkpoint_refiner,
        "initial_heldout": initial,
        "final_heldout": final,
        "minimum_training_signed_area_ratio": minimum_train_area,
        "median_full_forward_seconds_after_first": statistics.median(forward_times),
        "median_full_vjp_seconds_after_first": statistics.median(backward_times),
        "training_seconds": training_seconds,
        "peak_cuda_allocated_bytes": peak,
        "records": records,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
