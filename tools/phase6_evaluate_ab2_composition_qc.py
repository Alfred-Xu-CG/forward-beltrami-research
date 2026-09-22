"""Sample the exact two-factor PL composition's chain Jacobian on source faces."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_image_to_latent import _minimum_area_ratio
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import MultiscaleMonotoneGridLayer, evaluate_structured_p1_with_jacobian


def evaluate(checkpoint: str, batch_faces: int) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["method"] != "alternating" or state["layers"] != 2 or state["target_kind"] != "high32":
        raise ValueError("expected a two-layer high32 alternating map-oracle checkpoint")
    side = state["side"]
    parameters = state["parameters"]
    controls = (
        MultiscaleMonotoneGridLayer(side, axis="vertical")(((parameters[0], parameters[2]),)),
        MultiscaleMonotoneGridLayer(side, axis="horizontal")(((parameters[1], parameters[3]),)),
    )
    layer_areas = [_minimum_area_ratio(control) for control in controls]
    mesh = structured_rectangle(side - 1, side - 1)
    vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float32)
    faces = torch.as_tensor(mesh.faces.copy(), dtype=torch.long)
    centroids = vertices[faces].mean(dim=1)
    coefficients = torch.tensor([[0.015, 0.025, 0.0025]], dtype=torch.float32)
    target_vertices, _ = _target_on_faces(vertices[None], coefficients, 32)
    target_control = target_vertices.reshape(1, side, side, 2)
    sum_map = sum_mu = sum_sampled = 0.0
    maximum_mu = 0.0
    min_det = float("inf")
    with torch.no_grad():
        for start in range(0, centroids.shape[0], batch_faces):
            point = centroids[start:start + batch_faces][None]
            first, first_jac = evaluate_structured_p1_with_jacobian(controls[0], point)
            composed, second_jac = evaluate_structured_p1_with_jacobian(controls[1], first)
            jac = second_jac @ first_jac
            target_value, target_jac = _target_on_faces(point, coefficients, 32)
            _, sampled_jac = evaluate_structured_p1_with_jacobian(target_control, point)
            predicted_mu, analytic_mu, sampled_mu = (_mu(value) for value in (jac, target_jac, sampled_jac))
            sum_map += (composed - target_value).square().sum().item()
            sum_mu += (predicted_mu - analytic_mu).abs().square().sum().item()
            sum_sampled += (sampled_mu - analytic_mu).abs().square().sum().item()
            maximum_mu = max(maximum_mu, predicted_mu.abs().max().item())
            determinant = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
            min_det = min(min_det, determinant.min().item())
    face_count = centroids.shape[0]
    return {
        "checkpoint": checkpoint,
        "task": "exact_composition_source_face_centroid_sample_not_overlay_integral",
        "target_kind": "high32",
        "target_coefficients_ax_ay_af": [0.015, 0.025, 0.0025],
        "control_side": side,
        "control_vertices_per_factor": side**2,
        "control_faces_per_factor": face_count,
        "sampled_query_count": face_count,
        "factor_count": 2,
        "factor_minimum_signed_area_ratios": layer_areas,
        "source_centroid_map_rmse": math.sqrt(sum_map / (2 * face_count)),
        "source_centroid_beltrami_rmse": math.sqrt(sum_mu / face_count),
        "sampled_target_P1_centroid_beltrami_floor": math.sqrt(sum_sampled / face_count),
        "maximum_predicted_beltrami_modulus_at_centroids": maximum_mu,
        "minimum_chain_jacobian_determinant_at_centroids": min_det,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-faces", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.batch_faces), sort_keys=True))


if __name__ == "__main__":
    main()
