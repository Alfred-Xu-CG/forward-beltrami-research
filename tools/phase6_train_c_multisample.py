"""Multi-sample high32 image training of the all-edge sine-PCG Tutte layer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_image_to_latent import _minimum_area_ratio
from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from phase6_train_conductance_image_to_latent import MultiscaleEdgeImageEncoder
from phase6_spectral_edge_encoder import SpectralEdgeImageEncoder
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    SinePreconditionedTutteLayer, SpectralSafeFeedbackLayer,
    evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--image-gradient-weight", type=float, default=0.0)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-gradient-weight", type=float, default=0.0)
    parser.add_argument("--teacher-coordinate-weight", type=float, default=0.0)
    parser.add_argument("--encoder-body", choices=("local", "context"), default="context")
    parser.add_argument("--spectral-frequencies", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-state", default=None)
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    spectral_frequencies = tuple(
        int(value) for value in args.spectral_frequencies.split(",") if value)
    if (args.steps < 0 or args.train_count < 1 or args.test_count < 1
            or args.batch < 1 or args.image_gradient_weight < 0
            or args.teacher_gradient_weight < 0 or args.teacher_coordinate_weight < 0
            or ((args.teacher_gradient_weight or args.teacher_coordinate_weight)
                and not args.teacher_checkpoint)):
        raise ValueError("invalid dataset or step count")
    device = torch.device(args.device)
    torch.manual_seed(20260923)
    side = args.side
    mesh = structured_rectangle(side - 1, side - 1)
    grid = np.arange(side**2, dtype=np.int64).reshape(side, side)
    edges = np.concatenate((
        np.stack((grid[:, :-1].ravel(), grid[:, 1:].ravel()), axis=1),
        np.stack((grid[:-1].ravel(), grid[1:].ravel()), axis=1),
        np.stack((grid[:-1, :-1].ravel(), grid[1:, 1:].ravel()), axis=1),
    ))
    midpoints = mesh.vertices[edges].mean(axis=1)
    encoder = (
        SpectralEdgeImageEncoder(
            side, midpoints, body_mode=args.encoder_body,
            frequencies=spectral_frequencies,
        ) if spectral_frequencies else
        MultiscaleEdgeImageEncoder(
            side, midpoints, body_mode=args.encoder_body,
        )
    ).to(device)
    if args.load_state:
        state = torch.load(args.load_state, map_location=device, weights_only=False)
        previous = state["args"]
        for key in ("side", "image_side", "encoder_body", "spectral_frequencies"):
            if previous.get(key, "") != getattr(args, key):
                raise ValueError(f"loaded state does not match {key}")
        encoder.load_state_dict(state["encoder"])
    solver = SinePreconditionedTutteLayer(
        side, minimum_conductance=1, maximum_conductance=16,
        tolerance=1e-10, max_iterations=120,
    ).to(device)
    train = tuple(value.to(device) for value in make_dataset(
        args.train_count, args.image_side, 55101,
        target_family="high32", return_coefficients=True,
    ))
    test = tuple(value.to(device) for value in make_dataset(
        args.test_count, args.image_side, args.test_seed,
        target_family="high32", return_coefficients=True,
    ))
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    source_grid = vertices.reshape(side, side, 2)
    high_basis = (torch.sin(64 * math.pi * source_grid[..., 0])
                  * torch.sin(64 * math.pi * source_grid[..., 1]))
    teacher_control = None
    if args.teacher_checkpoint:
        teacher_state = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        teacher_settings = teacher_state["args"]
        if teacher_settings["side"] != side or teacher_settings["image_side"] != args.image_side:
            raise ValueError("teacher grid or image size differs")
        teacher_encoder = ConvexQuadLocalImageEncoder(
            side, width=teacher_settings["a2_width"],
            head_mode=teacher_settings["a2_head_mode"],
            body_mode=teacher_settings["a2_body_mode"],
        ).to(device)
        teacher_encoder.load_state_dict(teacher_state["encoder"])
        teacher_encoder.eval()
        teacher_decoder = SpectralSafeFeedbackLayer(
            side, initial_window=teacher_settings["hint_window"],
            initial_ridge=teacher_settings["hint_ridge"],
            initial_gain=teacher_settings["hint_gain"],
            initial_modes=teacher_settings["hint_sine_modes"],
            extra_passes=teacher_settings["extra_passes"],
            extra_gain=teacher_settings["extra_gain"],
            extra_modes=16, extra_qc_cap=teacher_settings["extra_qc_cap"],
            floor_fraction=teacher_settings["floor_fraction"],
        ).to(device)
        teacher_decoder.eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, args.train_count, args.batch):
                fixed = train[0][start:start + args.batch]
                moving = train[1][start:start + args.batch]
                latent = teacher_encoder(torch.cat((fixed, moving), dim=1))
                pieces.append(teacher_decoder(fixed, moving, latent))
        teacher_control = torch.cat(pieces)
        del teacher_encoder, teacher_decoder
    h_count, v_count, d_count = side * (side - 1), side * (side - 1), (side - 1)**2

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(dataset, indices):
        fixed, moving = dataset[0][indices], dataset[1][indices]
        logits = encoder(torch.cat((fixed, moving), dim=1)).double()
        horizontal, vertical, diagonal = torch.split(
            logits, (h_count, v_count, d_count), dim=1)
        control = solver(
            horizontal.reshape(len(indices), side, side - 1),
            vertical.reshape(len(indices), side - 1, side),
            diagonal.reshape(len(indices), side - 1, side - 1),
        ).float()
        query = table.interpolate(control.reshape(len(indices), -1, 2))
        warped = F.grid_sample(
            moving, 2 * query - 1, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        residual = warped - fixed
        image_mse = residual.square().mean()
        if args.image_gradient_weight:
            gradient_mse = 0.5 * (
                (residual[..., 1:] - residual[..., :-1]).square().mean()
                + (residual[..., 1:, :] - residual[..., :-1, :]).square().mean()
            )
            objective = image_mse + args.image_gradient_weight * gradient_mse
        else:
            objective = image_mse
        return objective, image_mse, control, query

    @torch.no_grad()
    def evaluate(dataset):
        count = len(dataset[0])
        image_sum, map_sum, mu_sum = 0.0, 0.0, 0.0
        maximum_mu, minimum_area, max_residual = 0.0, float("inf"), 0.0
        forward_iterations, projected = [], []
        for start in range(0, count, args.batch):
            stop = min(start + args.batch, count)
            indices = torch.arange(start, stop, device=device)
            _, image_mse, control, query = forward(dataset, indices)
            current = stop - start
            image_sum += image_mse.item() * current
            map_sum += (query - dataset[2][indices]).square().mean().item() * current
            minimum_area = min(minimum_area, _minimum_area_ratio(control))
            max_residual = max(max_residual, solver.last_forward_stats["true_relative_residual"])
            forward_iterations.append(solver.last_forward_stats["iterations"])
            face_query = centroids[None].expand(current, -1, -1)
            _, jacobian = evaluate_structured_p1_with_jacobian(control, face_query)
            _, target_jacobian = _target_on_faces(face_query, dataset[3][indices], 32)
            predicted_mu = _mu(jacobian)
            maximum_mu = max(maximum_mu, float(predicted_mu.abs().max()))
            mu_sum += (predicted_mu - _mu(target_jacobian)).abs().square().mean().item() * current
            if float(high_basis.square().sum()) > 1e-10:
                projection = ((control - source_grid) * high_basis[None, :, :, None]).sum(
                    dim=(1, 2)) / high_basis.square().sum()
                projected.extend(projection.tolist())
        return {
            "image_mse": image_sum / count,
            "query_map_rmse": math.sqrt(map_sum / count),
            "face_beltrami_rmse": math.sqrt(mu_sum / count),
            "maximum_predicted_beltrami_modulus": maximum_mu,
            "minimum_signed_area_ratio": minimum_area,
            "maximum_true_relative_forward_residual": max_residual,
            "maximum_forward_iterations": max(forward_iterations),
            "projected_fine_amplitudes": projected if projected else None,
            "true_fine_amplitudes": dataset[3][:, 2].tolist(),
        }

    initial_train, initial_test = evaluate(train), evaluate(test)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(38819)
    times_forward, times_vjp, records = [], [], []
    max_forward_residual, max_backward_residual = 0.0, 0.0
    began_training = time.perf_counter()
    for step in range(args.steps):
        indices = torch.randint(
            args.train_count, (args.batch,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        objective, image_mse, control, _ = forward(train, indices)
        teacher_gradient_mse = None
        teacher_coordinate_mse = None
        if teacher_control is not None:
            difference = control - teacher_control[indices]
            teacher_coordinate_mse = difference.square().mean()
            teacher_gradient_mse = 0.5 * (
                ((side - 1) * (difference[:, :, 1:] - difference[:, :, :-1])).square().mean()
                + ((side - 1) * (difference[:, 1:] - difference[:, :-1])).square().mean()
            )
            objective = (objective
                         + args.teacher_coordinate_weight * teacher_coordinate_mse
                         + args.teacher_gradient_weight * teacher_gradient_mse)
        max_forward_residual = max(
            max_forward_residual, solver.last_forward_stats["true_relative_residual"])
        synchronize()
        middle = time.perf_counter()
        objective.backward()
        max_backward_residual = max(
            max_backward_residual, solver.last_backward_stats["true_relative_residual"])
        synchronize()
        ended = time.perf_counter()
        optimizer.step()
        times_forward.append(middle - started)
        times_vjp.append(ended - middle)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            records.append({
                "step": step + 1, "sampled_image_mse_before_update": image_mse.item(),
                "sampled_objective_before_update": objective.item(),
                "sampled_teacher_gradient_mse_before_update": (
                    teacher_gradient_mse.item() if teacher_gradient_mse is not None else None),
                "sampled_teacher_coordinate_mse_before_update": (
                    teacher_coordinate_mse.item() if teacher_coordinate_mse is not None else None),
                "elapsed_seconds": time.perf_counter() - began_training,
                "forward_iterations": solver.last_forward_stats["iterations"],
                "backward_iterations": solver.last_backward_stats["iterations"],
            })
    training_seconds = time.perf_counter() - began_training
    final_train, final_test = evaluate(train), evaluate(test)
    if args.save_state:
        torch.save({"encoder": encoder.state_dict(), "args": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "C_all_edge_sine_pcg_multisample_image_to_latent",
        "representation": "original_grid_P1_positive_symmetric_conductance",
        "side": side,
        "control_vertices": side**2,
        "control_faces": 2 * (side - 1)**2,
        "active_edges": h_count + v_count + d_count,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "train_count": args.train_count, "train_seed": 55101,
        "test_count": args.test_count, "test_seed": args.test_seed,
        "batch": args.batch, "steps": args.steps,
        "learning_rate": args.learning_rate,
        "image_gradient_weight": args.image_gradient_weight,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_gradient_weight": args.teacher_gradient_weight,
        "teacher_coordinate_weight": args.teacher_coordinate_weight,
        "encoder_body": args.encoder_body,
        "spectral_frequencies": list(spectral_frequencies),
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "device": str(device), "solver_dtype": "float64", "image_dtype": "float32",
        "minimum_conductance": 1, "maximum_conductance": 16,
        "solver_tolerance": 1e-10,
        "initial_train": initial_train, "initial_test": initial_test,
        "final_train": final_train, "final_test": final_test,
        "median_full_forward_seconds_after_first": (
            statistics.median(times_forward[1:] or times_forward) if times_forward else None),
        "median_full_vjp_seconds_after_first": (
            statistics.median(times_vjp[1:] or times_vjp) if times_vjp else None),
        "maximum_training_true_relative_forward_residual": max_forward_residual,
        "maximum_training_true_relative_adjoint_residual": max_backward_residual,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        "training_seconds": training_seconds,
        "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
