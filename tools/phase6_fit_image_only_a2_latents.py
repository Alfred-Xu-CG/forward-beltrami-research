"""Diagnostic: optimize A2 latents for one pair using only image MSE.

The true map and coefficients are used after optimization for evaluation only.
This is not amortized image-to-latent inference or a clinical experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    HierarchicalConvexQuadFreeCenterLayer,
    certify_convex_quad_output,
    evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    if args.side < 5 or args.steps < 1 or not 0 <= args.sample_index < 8:
        raise ValueError("invalid side, steps or held-out sample index")
    torch.manual_seed(20260923)
    fixed_all, moving_all, true_map_all, coefficients_all = make_dataset(
        8, args.image_side, 99317, return_coefficients=True, target_family="high32"
    )
    selection = slice(args.sample_index, args.sample_index + 1)
    fixed, moving, true_map, coefficients = (
        value[selection] for value in (fixed_all, moving_all, true_map_all, coefficients_all)
    )
    decoder = HierarchicalConvexQuadFreeCenterLayer(args.side)
    root = torch.nn.Parameter(torch.zeros(1, 1, 1, 2))
    horizontal = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(1, side, side - 1)) for side in decoder.latent_sides)
    vertical = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(1, side - 1, side)) for side in decoder.latent_sides)
    centers = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(1, side - 1, side - 1, 2)) for side in decoder.latent_sides)
    parameters = [root, *horizontal, *vertical, *centers]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=torch.device("cpu"), dtype=torch.float32)

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        control = decoder(root, tuple(zip(horizontal, vertical, centers)))
        dense = table.interpolate(control.reshape(1, -1, 2))
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), control, dense

    with torch.no_grad():
        initial_image_mse = forward()[0].item()
    forward_times = []
    backward_times = []
    records = []
    began_all = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        loss, _, _ = forward()
        middle = time.perf_counter()
        loss.backward()
        ended = time.perf_counter()
        optimizer.step()
        forward_times.append(middle - began)
        backward_times.append(ended - middle)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            records.append({"step": step + 1, "image_mse_before_update": loss.item(), "cumulative_seconds": time.perf_counter() - began_all})
    total_seconds = time.perf_counter() - began_all
    with torch.no_grad():
        final_loss, control, dense = forward()
        minimum_area = certify_convex_quad_output(control)
        vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32)
        faces = torch.tensor(mesh.faces.copy())
        source_grid = vertices.reshape(args.side, args.side, 2)
        basis = torch.sin(64 * math.pi * source_grid[..., 0]) * torch.sin(64 * math.pi * source_grid[..., 1])
        basis_norm = basis.square().sum()
        projected = (
            ((control[0] - source_grid) * basis[..., None]).sum(dim=(0, 1)) / basis_norm
            if basis_norm.item() > 1e-8 else None
        )
        centroids = vertices[faces].mean(dim=1)[None]
        _, jacobian = evaluate_structured_p1_with_jacobian(control, centroids)
        _, target_jacobian = _target_on_faces(centroids, coefficients, 32)
        sampled_target, _ = _target_on_faces(vertices[None], coefficients, 32)
        _, sampled_jacobian = evaluate_structured_p1_with_jacobian(sampled_target.reshape(1, args.side, args.side, 2), centroids)
        predicted_mu, target_mu, sampled_mu = (_mu(j) for j in (jacobian, target_jacobian, sampled_jacobian))
    if args.save_state:
        torch.save({
            "method": "A2_image_only_direct_latents",
            "side": args.side,
            "image_side": args.image_side,
            "sample_index": args.sample_index,
            "parameters": [parameter.detach().clone() for parameter in parameters],
        }, args.save_state)
    print(json.dumps({
        "task": "image_only_direct_latent_optimization_not_encoder_training",
        "target_family": "high32",
        "heldout_seed": 99317,
        "sample_index": args.sample_index,
        "true_fine_amplitude_evaluation_only": coefficients[0, 2].item(),
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "latent_values": sum(parameter.numel() for parameter in parameters),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_image_mse": initial_image_mse,
        "final_image_mse": final_loss.item(),
        "final_query_map_rmse": (dense - true_map).square().mean().sqrt().item(),
        "projected_fine_x_amplitude": projected[0].item() if projected is not None else None,
        "projected_fine_y_amplitude": projected[1].item() if projected is not None else None,
        "face_beltrami_rmse": (predicted_mu - target_mu).abs().square().mean().sqrt().item(),
        "sampled_target_P1_beltrami_discretization_rmse": (sampled_mu - target_mu).abs().square().mean().sqrt().item(),
        "maximum_predicted_beltrami_modulus": predicted_mu.abs().max().item(),
        "minimum_source_normalized_face_area": minimum_area,
        "median_forward_seconds": statistics.median(forward_times[1:] or forward_times),
        "median_backward_seconds": statistics.median(backward_times[1:] or backward_times),
        "train_seconds": total_seconds,
        "records": records,
        "saved_state": args.save_state,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
