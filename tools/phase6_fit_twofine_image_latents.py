"""One-pair image-only latent oracle for two exact fine-grid PL factors."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import CoarseFineConvexQuadComposition, certify_convex_quad_output, evaluate_structured_p1_with_jacobian


def _latent_parameters(factor):
    root = torch.nn.Parameter(torch.zeros(1, 1, 1, 2))
    levels = tuple(
        (torch.nn.Parameter(torch.zeros(1, n, n - 1)),
         torch.nn.Parameter(torch.zeros(1, n - 1, n)),
         torch.nn.Parameter(torch.zeros(1, n - 1, n - 1, 2)))
        for n in factor.latent_sides
    )
    return root, levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--save-state")
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.sample_index < 8:
        raise ValueError("invalid steps or sample index")
    torch.manual_seed(20260923)
    fixed_all, moving_all, true_map_all, coefficients_all = make_dataset(
        8, args.image_side, 99317, return_coefficients=True, target_family="high32"
    )
    selection = slice(args.sample_index, args.sample_index + 1)
    fixed, moving, true_map, coefficients = (
        value[selection] for value in (fixed_all, moving_all, true_map_all, coefficients_all)
    )
    decoder = CoarseFineConvexQuadComposition(args.side, args.side, args.image_side)
    decoder.prepare(device="cpu", dtype=torch.float32)
    first = _latent_parameters(decoder.coarse)
    second = _latent_parameters(decoder.fine)
    parameters = [first[0], *(p for level in first[1] for p in level), second[0], *(p for level in second[1] for p in level)]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    def forward():
        result = decoder(first[0], first[1], second[0], second[1])
        warped = F.grid_sample(moving, 2 * result.dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), result

    with torch.no_grad():
        initial = forward()[0].item()
    records = []
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            records.append({"step": step + 1, "image_mse_before_update": loss.item(), "cumulative_seconds": time.perf_counter() - started})
    duration = time.perf_counter() - started
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float32)
    faces = torch.as_tensor(mesh.faces.copy(), dtype=torch.long)
    with torch.no_grad():
        final, result = forward()
        first_vertices, _ = evaluate_structured_p1_with_jacobian(result.controls[0], vertices[None])
        mapped_vertices, _ = evaluate_structured_p1_with_jacobian(result.controls[1], first_vertices)
        basis = torch.sin(64 * math.pi * vertices[:, 0]) * torch.sin(64 * math.pi * vertices[:, 1])
        denominator = basis.square().sum()
        projection = (
            ((mapped_vertices - vertices[None]) * basis[None, :, None]).sum(dim=1) / denominator
            if denominator.item() > 1e-8 else None
        )
        centroids = vertices[faces].mean(dim=1)[None]
        first_face, first_jacobian = evaluate_structured_p1_with_jacobian(result.controls[0], centroids)
        _, second_jacobian = evaluate_structured_p1_with_jacobian(result.controls[1], first_face)
        jacobian = second_jacobian @ first_jacobian
        _, target_jacobian = _target_on_faces(centroids, coefficients, 32)
        mu, target_mu = _mu(jacobian), _mu(target_jacobian)
        determinant = torch.linalg.det(jacobian)
    if args.save_state:
        torch.save({
            "method": "twofine_direct_image_latents",
            "side": args.side,
            "image_side": args.image_side,
            "sample_index": args.sample_index,
            "parameters": [parameter.detach().cpu().clone() for parameter in parameters],
        }, args.save_state)
    print(json.dumps({
        "task": "one_pair_image_only_direct_latent_optimization_not_amortized_encoder",
        "target_family": "high32",
        "sample_index": args.sample_index,
        "heldout_seed": 99317,
        "true_fine_amplitude_evaluation_only": coefficients[0, 2].item(),
        "control_sides": [args.side, args.side],
        "control_vertices_per_factor": args.side**2,
        "control_faces_per_factor": mesh.n_faces,
        "image_queries": args.image_side**2,
        "latent_values": sum(p.numel() for p in parameters),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_image_mse": initial,
        "final_image_mse": final.item(),
        "final_query_map_rmse": (result.dense - true_map).square().mean().sqrt().item(),
        "projected_fine_xy_amplitude": projection[0].tolist() if projection is not None else None,
        "source_centroid_beltrami_rmse": (mu - target_mu).abs().square().mean().sqrt().item(),
        "maximum_predicted_beltrami_modulus_at_centroids": mu.abs().max().item(),
        "minimum_chain_jacobian_determinant_at_centroids": determinant.min().item(),
        "minimum_factor_signed_area_ratios": [certify_convex_quad_output(control) for control in result.controls],
        "training_seconds": duration,
        "records": records,
        "saved_state": args.save_state,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
