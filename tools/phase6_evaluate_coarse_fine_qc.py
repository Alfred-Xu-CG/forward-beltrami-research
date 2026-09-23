"""Evaluate coarse-to-fine exact PL chain Jacobians on source face centroids."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import HierarchicalConvexQuadFreeCenterLayer, certify_convex_quad_output, evaluate_structured_p1_with_jacobian


def _decode(parameters, side: int, offset: int):
    decoder = HierarchicalConvexQuadFreeCenterLayer(side)
    count = len(decoder.latent_sides)
    root = parameters[offset]
    horizontal = parameters[offset + 1:offset + 1 + count]
    vertical = parameters[offset + 1 + count:offset + 1 + 2 * count]
    centers = parameters[offset + 1 + 2 * count:offset + 1 + 3 * count]
    return decoder(root, tuple(zip(horizontal, vertical, centers))), offset + 1 + 3 * count


def evaluate(checkpoint: str, batch_faces: int) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["method"] != "convex_quad_coarse_fine" or state["layers"] != 2:
        raise ValueError("expected a two-factor coarse-fine convex-cell checkpoint")
    side = state["side"]
    coarse, offset = _decode(state["parameters"], state["coarse_side"], 0)
    fine, offset = _decode(state["parameters"], side, offset)
    if offset != len(state["parameters"]):
        raise ValueError("checkpoint contains an unexpected latent count")
    if state["target_kind"] == "base":
        coefficients = torch.tensor([[0.03, 0.05, 0.003]], dtype=torch.float32)
        cycles = 8
    else:
        coefficients = torch.tensor([[0.015, 0.025, 0.0025]], dtype=torch.float32)
        cycles = 32
    mesh = structured_rectangle(side - 1, side - 1)
    vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float32)
    faces = torch.as_tensor(mesh.faces.copy(), dtype=torch.long)
    centroids = vertices[faces].mean(dim=1)
    target_vertices, _ = _target_on_faces(vertices[None], coefficients, cycles)
    target_p1 = target_vertices.reshape(1, side, side, 2)
    fine_projection = None
    if state["target_kind"] == "high32":
        with torch.no_grad():
            first_vertices, _ = evaluate_structured_p1_with_jacobian(coarse, vertices[None])
            composed_vertices, _ = evaluate_structured_p1_with_jacobian(fine, first_vertices)
            basis = torch.sin(64 * math.pi * vertices[:, 0]) * torch.sin(64 * math.pi * vertices[:, 1])
            denominator = basis.square().sum()
            predicted_amplitude = ((composed_vertices - vertices[None]) * basis[None, :, None]).sum(dim=1) / denominator
            target_amplitude = ((target_vertices - vertices[None]) * basis[None, :, None]).sum(dim=1) / denominator
            fine_projection = {
                "predicted_xy": predicted_amplitude[0].tolist(),
                "target_xy": target_amplitude[0].tolist(),
            }
    map_squared = mu_squared = floor_squared = 0.0
    maximum_mu = 0.0
    minimum_det = float("inf")
    with torch.no_grad():
        for start in range(0, len(centroids), batch_faces):
            points = centroids[start:start + batch_faces][None]
            coarse_value, coarse_jacobian = evaluate_structured_p1_with_jacobian(coarse, points)
            mapped, fine_jacobian = evaluate_structured_p1_with_jacobian(fine, coarse_value)
            jacobian = fine_jacobian @ coarse_jacobian
            target, target_jacobian = _target_on_faces(points, coefficients, cycles)
            _, target_p1_jacobian = evaluate_structured_p1_with_jacobian(target_p1, points)
            mu, true_mu, p1_mu = (_mu(value) for value in (jacobian, target_jacobian, target_p1_jacobian))
            map_squared += (mapped - target).square().sum().item()
            mu_squared += (mu - true_mu).abs().square().sum().item()
            floor_squared += (p1_mu - true_mu).abs().square().sum().item()
            maximum_mu = max(maximum_mu, mu.abs().max().item())
            determinant = jacobian[..., 0, 0] * jacobian[..., 1, 1] - jacobian[..., 0, 1] * jacobian[..., 1, 0]
            minimum_det = min(minimum_det, determinant.min().item())
    count = len(centroids)
    return {
        "checkpoint": checkpoint,
        "task": "exact_composition_source_face_centroid_sample_not_overlay_integral",
        "target_kind": state["target_kind"],
        "coarse_control_side": state["coarse_side"],
        "fine_control_side": side,
        "fine_control_vertices": side**2,
        "fine_control_faces": count,
        "sampled_query_count": count,
        "factor_minimum_signed_area_ratios": [certify_convex_quad_output(coarse), certify_convex_quad_output(fine)],
        "source_centroid_map_rmse": math.sqrt(map_squared / (2 * count)),
        "source_centroid_beltrami_rmse": math.sqrt(mu_squared / count),
        "sampled_target_P1_centroid_beltrami_floor": math.sqrt(floor_squared / count),
        "maximum_predicted_beltrami_modulus_at_centroids": maximum_mu,
        "minimum_chain_jacobian_determinant_at_centroids": minimum_det,
        "fine_projection_on_original_vertices": fine_projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-faces", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.batch_faces), sort_keys=True))


if __name__ == "__main__":
    main()
