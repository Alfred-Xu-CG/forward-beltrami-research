"""Held-out geometry of image-trained exact coarse-fine PL compositions."""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_image_to_latent import _minimum_area_ratio
from phase6_train_multisample_image import CoarseFineConvexQuadImageEncoder, make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import CoarseFineConvexQuadComposition, evaluate_structured_p1_with_jacobian


def evaluate(checkpoint: str, batch: int, device: str) -> dict:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    args = state["args"]
    if args["method"] != "CF2":
        raise ValueError("expected an image-trained CF2 checkpoint")
    side = args["side"]
    coarse_side = args["coarse_side"]
    image_side = args["image_side"]
    family = args["target_family"]
    cycles = 8 if family == "base" else 32
    encoder = CoarseFineConvexQuadImageEncoder(
        coarse_side, side,
        head_mode=args.get("a2_head_mode", "multilevel"),
        body_mode=args.get("a2_body_mode", "local"),
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    decoder = CoarseFineConvexQuadComposition(coarse_side, side, image_side)
    decoder.prepare(device=device, dtype=torch.float32)
    fixed, moving, true_map, coefficients = (
        value.to(device) for value in make_dataset(8, image_side, 99317, return_coefficients=True, target_family=family)
    )
    mesh = structured_rectangle(side - 1, side - 1)
    vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces.copy(), dtype=torch.long, device=device)
    centroids = vertices[faces].mean(dim=1)
    image_sum = pixel_map_sum = face_map_sum = mu_sum = floor_sum = 0.0
    minimum_areas = [float("inf"), float("inf")]
    maximum_mu = 0.0
    minimum_chain_det = float("inf")
    fine_projections = []
    count = fixed.shape[0]
    with torch.no_grad():
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            current = stop - start
            pair = torch.cat((fixed[start:stop], moving[start:stop]), dim=1)
            coarse_latent, fine_latent = encoder(pair)
            result = decoder(*coarse_latent, *fine_latent)
            for index, control in enumerate(result.controls):
                minimum_areas[index] = min(minimum_areas[index], _minimum_area_ratio(control))
            warped = F.grid_sample(moving[start:stop], 2 * result.dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
            image_sum += (warped - fixed[start:stop]).square().mean().item() * current
            pixel_map_sum += (result.dense - true_map[start:stop]).square().mean().item() * current
            query = centroids[None].expand(current, -1, -1)
            first, first_jacobian = evaluate_structured_p1_with_jacobian(result.controls[0], query)
            predicted, second_jacobian = evaluate_structured_p1_with_jacobian(result.controls[1], first)
            jacobian = second_jacobian @ first_jacobian
            target, target_jacobian = _target_on_faces(query, coefficients[start:stop], cycles)
            sampled_vertices, _ = _target_on_faces(vertices[None].expand(current, -1, -1), coefficients[start:stop], cycles)
            _, sampled_jacobian = evaluate_structured_p1_with_jacobian(
                sampled_vertices.reshape(current, side, side, 2), query
            )
            mu, target_mu, sampled_mu = (_mu(value) for value in (jacobian, target_jacobian, sampled_jacobian))
            face_map_sum += (predicted - target).square().mean().item() * current
            mu_sum += (mu - target_mu).abs().square().mean().item() * current
            floor_sum += (sampled_mu - target_mu).abs().square().mean().item() * current
            maximum_mu = max(maximum_mu, mu.abs().max().item())
            determinant = jacobian[..., 0, 0] * jacobian[..., 1, 1] - jacobian[..., 0, 1] * jacobian[..., 1, 0]
            minimum_chain_det = min(minimum_chain_det, determinant.min().item())
            if family == "high32":
                first_vertices, _ = evaluate_structured_p1_with_jacobian(
                    result.controls[0], vertices[None].expand(current, -1, -1)
                )
                composed_vertices, _ = evaluate_structured_p1_with_jacobian(result.controls[1], first_vertices)
                source = vertices.reshape(side, side, 2)
                basis = torch.sin(64 * math.pi * source[..., 0]) * torch.sin(64 * math.pi * source[..., 1])
                displacement = composed_vertices.reshape(current, side, side, 2) - source
                projected = (displacement * basis[None, ..., None]).sum(dim=(1, 2)) / basis.square().sum()
                for index in range(current):
                    fine_projections.append({
                        "true_amplitude": coefficients[start + index, 2].item(),
                        "projected_x_amplitude": projected[index, 0].item(),
                        "projected_y_amplitude": projected[index, 1].item(),
                    })
    return {
        "checkpoint": checkpoint,
        "method": "CF2_image_only_training",
        "representation": "exact_PL_composition_not_original_grid_P1",
        "target_family": family,
        "coarse_control_side": coarse_side,
        "fine_control_side": side,
        "coarse_control_vertices": coarse_side**2,
        "fine_control_vertices": side**2,
        "coarse_control_faces": 2 * (coarse_side - 1)**2,
        "fine_control_faces": len(faces),
        "image_queries": image_side**2,
        "test_count": count,
        "test_seed": 99317,
        "batch": batch,
        "device": device,
        "heldout_image_mse": image_sum / count,
        "heldout_pixel_query_map_rmse": math.sqrt(pixel_map_sum / count),
        "heldout_source_centroid_map_rmse": math.sqrt(face_map_sum / count),
        "heldout_source_centroid_beltrami_rmse": math.sqrt(mu_sum / count),
        "sampled_target_P1_centroid_beltrami_floor": math.sqrt(floor_sum / count),
        "minimum_factor_signed_area_ratios": minimum_areas,
        "minimum_chain_jacobian_determinant_at_centroids": minimum_chain_det,
        "maximum_predicted_beltrami_modulus_at_centroids": maximum_mu,
        "fine_projection_estimates": fine_projections if family == "high32" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.batch, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
