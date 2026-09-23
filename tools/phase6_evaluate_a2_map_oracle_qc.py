"""Face-level QC geometry of a direct-latent A2 map oracle checkpoint."""

from __future__ import annotations

import argparse
import json

import torch

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    HierarchicalConvexQuadFreeCenterLayer,
    SafeColoredVertexRelaxation,
    certify_convex_quad_output,
    evaluate_structured_p1_with_jacobian,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-kind", choices=("base", "high32"), default=None, help="for older checkpoints without target_kind")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["method"] not in ("convex_quad_free", "convex_quad_local", "convex_quad_radial") or state["side"] != 257:
        raise ValueError("requires a 257-square free-center or local-relaxation direct-map checkpoint")
    target_kind = state.get("target_kind", args.target_kind or "base")
    if target_kind == "base":
        coefficients = torch.tensor([[0.03, 0.05, 0.003]], dtype=torch.float32)
        fine_cycles = 8
    elif target_kind == "high32":
        coefficients = torch.tensor([[0.015, 0.025, 0.0025]], dtype=torch.float32)
        fine_cycles = 32
    else:
        raise ValueError("unknown target kind")
    decoder = HierarchicalConvexQuadFreeCenterLayer(257)
    parameters = state["parameters"]
    levels = len(decoder.latent_sides)
    if len(parameters) != 1 + 3 * levels + (state["method"] in ("convex_quad_local", "convex_quad_radial")):
        raise ValueError("checkpoint latent count mismatch")
    root = parameters[0]
    horizontal = parameters[1:1 + levels]
    vertical = parameters[1 + levels:1 + 2 * levels]
    centers = parameters[1 + 2 * levels:1 + 3 * levels]
    with torch.no_grad():
        control = decoder(root, tuple(zip(horizontal, vertical, centers)))
        if state["method"] in ("convex_quad_local", "convex_quad_radial"):
            control = SafeColoredVertexRelaxation(
                257, safety_fraction=0.85,
                motion_mode="radial" if state["method"] == "convex_quad_radial" else "disk",
                raw_span=state.get("raw_span", 2.0) or 2.0,
            )(control, parameters[-1])
        minimum_area = certify_convex_quad_output(control)
        mesh = structured_rectangle(256, 256)
        vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32)
        source_grid = vertices.reshape(257, 257, 2)
        high_basis = torch.sin(2 * torch.pi * fine_cycles * source_grid[..., 0]) * torch.sin(2 * torch.pi * fine_cycles * source_grid[..., 1])
        projected_fine_amplitude = ((control[0] - source_grid) * high_basis[..., None]).sum(dim=(0, 1)) / high_basis.square().sum()
        faces = torch.tensor(mesh.faces.copy())
        centroids = vertices[faces].mean(dim=1)[None]
        predicted, jacobian = evaluate_structured_p1_with_jacobian(control, centroids)
        target, target_jacobian = _target_on_faces(centroids, coefficients, fine_cycles)
        sampled_target, _ = _target_on_faces(vertices[None], coefficients, fine_cycles)
        _, sampled_target_jacobian = evaluate_structured_p1_with_jacobian(sampled_target.reshape(1, 257, 257, 2), centroids)
        predicted_mu = _mu(jacobian)
        target_mu = _mu(target_jacobian)
        sampled_target_mu = _mu(sampled_target_jacobian)
        determinant = jacobian[..., 0, 0] * jacobian[..., 1, 1] - jacobian[..., 0, 1] * jacobian[..., 1, 0]
    print(json.dumps({
        "task": "direct_latent_oracle_not_image_training",
        "checkpoint": args.checkpoint,
        "method": state["method"],
        "target_kind": target_kind,
        "control_vertices": 257**2,
        "control_faces": 2 * 256**2,
        "face_centroid_map_rmse": (predicted - target).square().mean().sqrt().item(),
        "face_jacobian_component_rmse": (jacobian - target_jacobian).square().mean().sqrt().item(),
        "face_beltrami_rmse": (predicted_mu - target_mu).abs().square().mean().sqrt().item(),
        "sampled_target_P1_beltrami_discretization_rmse": (sampled_target_mu - target_mu).abs().square().mean().sqrt().item(),
        "predicted_vs_sampled_target_P1_beltrami_rmse": (predicted_mu - sampled_target_mu).abs().square().mean().sqrt().item(),
        "maximum_predicted_beltrami_modulus": predicted_mu.abs().max().item(),
        "maximum_target_beltrami_modulus": target_mu.abs().max().item(),
        "minimum_face_jacobian_determinant": determinant.min().item(),
        "minimum_source_normalized_face_area": minimum_area,
        "true_fine_amplitude": coefficients[0, 2].item(),
        "projected_fine_x_amplitude": projected_fine_amplitude[0].item(),
        "projected_fine_y_amplitude": projected_fine_amplitude[1].item(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
