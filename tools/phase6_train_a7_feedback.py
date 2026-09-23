"""Image-to-latent training through one safe spectral residual-feedback pass."""

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
    structured_p1_qc_tail_penalty,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-state", required=True)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--qc-weight", type=float, default=0.5)
    parser.add_argument("--qc-threshold", type=float, default=0.45)
    parser.add_argument("--extra-passes", type=int, default=1)
    parser.add_argument("--extra-gain", type=float, default=0.25)
    parser.add_argument("--extra-qc-cap", type=float, default=None)
    parser.add_argument("--floor-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 0 or args.qc_weight < 0 or not 0 <= args.qc_threshold < 1:
        raise ValueError("steps or QC configuration invalid")
    device = torch.device(args.device)
    state = torch.load(args.load_state, map_location=device, weights_only=False)
    settings = state["args"]
    if settings["method"] not in ("A6", "A7", "A8") or settings["target_family"] != "high32":
        raise ValueError("requires high32 A6/A7/A8 encoder state")
    side, image_side = settings["side"], settings["image_side"]
    encoder = ConvexQuadLocalImageEncoder(
        side, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"], body_mode=settings["a2_body_mode"],
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    decoder = SpectralSafeFeedbackLayer(
        side, initial_window=settings["hint_window"], initial_ridge=settings["hint_ridge"],
        initial_gain=settings["hint_gain"], initial_modes=settings["hint_sine_modes"],
        extra_passes=args.extra_passes, extra_gain=args.extra_gain, extra_modes=16,
        floor_fraction=args.floor_fraction, extra_qc_cap=args.extra_qc_cap,
    ).to(device)
    train = tuple(value.to(device) for value in make_dataset(
        args.train_count, image_side, 55101, target_family="high32", return_coefficients=True,
    ))
    test = tuple(value.to(device) for value in make_dataset(
        args.test_count, image_side, args.test_seed, target_family="high32", return_coefficients=True,
    ))
    mesh = structured_rectangle(side - 1, side - 1)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    source_grid = vertices.reshape(side, side, 2)
    high_basis = torch.sin(64 * math.pi * source_grid[..., 0]) * torch.sin(64 * math.pi * source_grid[..., 1])
    table = StructuredDenseQueryTable.from_mesh(mesh, height=image_side, width=image_side)
    table.prepare(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    def synchronize() -> None:
        if device.type == "cuda": torch.cuda.synchronize(device)
    def forward(dataset, indices):
        fixed, moving = dataset[0][indices], dataset[1][indices]
        control = decoder(fixed, moving, encoder(torch.cat((fixed, moving), dim=1)))
        query = table.interpolate(control.reshape(len(indices), -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1,
                               mode="bilinear", padding_mode="border", align_corners=True)
        image_mse = (warped - fixed).square().mean()
        return image_mse, control, query
    @torch.no_grad()
    def evaluate(dataset):
        image_sum, map_sum, mu_sum = 0.0, 0.0, 0.0
        minimum = float("inf"); maximum_mu = 0.0
        projected = []
        sample_maximum_mu, sample_minimum_area = [], []
        initial_sample_maximum_mu = []
        count = len(dataset[0])
        for start in range(0, count, args.batch):
            stop = min(start + args.batch, count)
            indices = torch.arange(start, stop, device=device)
            image_mse, control, query = forward(dataset, indices)
            current = stop - start
            image_sum += image_mse.item() * current
            map_sum += (query - dataset[2][indices]).square().mean().item() * current
            minimum = min(minimum, _minimum_area_ratio(control))
            face_query = centroids[None].expand(current, -1, -1)
            _, jacobian = evaluate_structured_p1_with_jacobian(control, face_query)
            if args.extra_qc_cap is not None:
                fixed, moving = dataset[0][indices], dataset[1][indices]
                initial = decoder.initial_map(
                    fixed, moving, encoder(torch.cat((fixed, moving), dim=1)))
                _, initial_jacobian = evaluate_structured_p1_with_jacobian(initial, face_query)
                initial_sample_maximum_mu.extend(
                    _mu(initial_jacobian).abs().reshape(current, -1).amax(dim=1).tolist()
                )
            _, target_jacobian = _target_on_faces(face_query, dataset[3][indices], 32)
            predicted_mu = _mu(jacobian)
            mu_sum += (predicted_mu - _mu(target_jacobian)).abs().square().mean().item() * current
            maximum_mu = max(maximum_mu, float(predicted_mu.abs().max()))
            sample_maximum_mu.extend(predicted_mu.abs().reshape(current, -1).amax(dim=1).tolist())
            a, b = control[:, :-1, :-1], control[:, :-1, 1:]
            c, d = control[:, 1:, 1:], control[:, 1:, :-1]
            lower = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
                b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
            upper = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (
                c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
            sample_minimum_area.extend(
                (torch.minimum(lower.reshape(current, -1).amin(dim=1),
                               upper.reshape(current, -1).amin(dim=1)) * (side - 1) ** 2).tolist()
            )
            projection = ((control - source_grid) * high_basis[None, :, :, None]).sum(dim=(1, 2)) / high_basis.square().sum()
            projected.extend(projection.tolist())
        return {
            "image_mse": image_sum / count,
            "query_map_rmse": math.sqrt(map_sum / count),
            "face_beltrami_rmse": math.sqrt(mu_sum / count),
            "maximum_predicted_beltrami_modulus": maximum_mu,
            "minimum_signed_area_ratio": minimum,
            "projected_fine_amplitudes": projected,
            "true_fine_amplitudes": dataset[3][:, 2].tolist(),
            "per_sample_maximum_predicted_beltrami_modulus": sample_maximum_mu,
            "per_sample_minimum_signed_area_ratio": sample_minimum_area,
            "per_sample_initial_maximum_beltrami_modulus": initial_sample_maximum_mu,
            "initial_qc_cap_eligible_count": sum(
                value < args.extra_qc_cap for value in initial_sample_maximum_mu
            ) if args.extra_qc_cap is not None else None,
        }
    initial_train, initial_test = evaluate(train), evaluate(test)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(38819)
    forward_times, backward_times, records = [], [], []
    began_all = time.perf_counter()
    for step in range(args.steps):
        draw = torch.randint(args.train_count, (args.batch,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        synchronize(); began = time.perf_counter()
        image_mse, control, _ = forward(train, draw)
        qc_penalty = (structured_p1_qc_tail_penalty(control, args.qc_threshold)
                      if args.qc_weight else image_mse.new_zeros(()))
        objective = image_mse + args.qc_weight * qc_penalty
        synchronize(); middle = time.perf_counter()
        objective.backward()
        synchronize(); after_backward = time.perf_counter()
        optimizer.step()
        forward_times.append(middle - began)
        backward_times.append(after_backward - middle)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            records.append({"step": step + 1, "sampled_image_mse_before_update": image_mse.item(),
                            "sampled_qc_penalty_before_update": qc_penalty.item(),
                            "sampled_objective_before_update": objective.item(),
                            "elapsed_seconds": time.perf_counter() - began_all})
    training_seconds = time.perf_counter() - began_all
    final_train, final_test = evaluate(train), evaluate(test)
    if args.save_state:
        saved_args = dict(settings)
        saved_args.update(vars(args))
        saved_args["method"] = "A8" if args.extra_qc_cap is not None else "A7"
        saved_args["target_family"] = "high32"
        torch.save({"encoder": encoder.state_dict(), "args": saved_args}, args.save_state)
    print(json.dumps({
        "method": ("A8_spectral_qc_safe_feedback_image_to_latent"
                   if args.extra_qc_cap is not None else
                   "A7_spectral_safe_feedback_image_to_latent"),
        "representation": "original_grid_P1",
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
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "qc_weight": args.qc_weight,
        "qc_threshold": args.qc_threshold,
        "extra_passes": args.extra_passes,
        "extra_gain": args.extra_gain,
        "extra_qc_cap": args.extra_qc_cap,
        "extra_sine_modes": 16,
        "floor_fraction": args.floor_fraction,
        "loaded_state": args.load_state,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "initial_train": initial_train,
        "initial_test": initial_test,
        "final_train": final_train,
        "final_test": final_test,
        "median_full_forward_seconds_after_first": statistics.median(forward_times[1:] or forward_times) if forward_times else None,
        "median_full_vjp_seconds_after_first": statistics.median(backward_times[1:] or backward_times) if backward_times else None,
        "training_seconds": training_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
