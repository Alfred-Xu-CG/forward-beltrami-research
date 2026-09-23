"""Evaluate face-wise Beltrami geometry of an image-trained A2 or A3 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import ConvexQuadImageEncoder, ConvexQuadLocalImageEncoder, make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    HierarchicalConvexQuadFreeCenterLayer,
    HierarchicalConvexQuadLocalLayer,
    certify_convex_quad_output,
    evaluate_structured_p1_with_jacobian,
    local_photometric_logits,
    spectralize_bounded_logits,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _mu(jacobian: torch.Tensor) -> torch.Tensor:
    f_z = torch.complex(
        0.5 * (jacobian[..., 0, 0] + jacobian[..., 1, 1]),
        0.5 * (jacobian[..., 1, 0] - jacobian[..., 0, 1]),
    )
    f_zbar = torch.complex(
        0.5 * (jacobian[..., 0, 0] - jacobian[..., 1, 1]),
        0.5 * (jacobian[..., 1, 0] + jacobian[..., 0, 1]),
    )
    return f_zbar / f_z


def _target_on_faces(
    points: torch.Tensor, coefficients: torch.Tensor, fine_cycles: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    x = points[..., 0]
    y = points[..., 1]
    ax, ay, af = (coefficients[:, index, None] for index in range(3))
    low = torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y)
    omega = 2 * math.pi * fine_cycles
    high = torch.sin(omega * x) * torch.sin(omega * y)
    low_x = 2 * math.pi * torch.cos(2 * math.pi * x) * torch.sin(2 * math.pi * y)
    low_y = 2 * math.pi * torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y)
    high_x = omega * torch.cos(omega * x) * torch.sin(omega * y)
    high_y = omega * torch.sin(omega * x) * torch.cos(omega * y)
    value = torch.stack((x + ax * low + af * high, y + ay * low + af * high), dim=-1)
    row_x = torch.stack((1 + ax * low_x + af * high_x, ax * low_y + af * high_y), dim=-1)
    row_y = torch.stack((ay * low_x + af * high_x, 1 + ay * low_y + af * high_y), dim=-1)
    return value, torch.stack((row_x, row_y), dim=-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    method = state["args"]["method"]
    if method not in ("A2", "A3", "A4", "A5", "A6") or state["args"]["side"] != args.side:
        raise ValueError("checkpoint does not match A2/A3/A4/A5/A6 and requested side")
    target_family = state["args"].get("target_family", "base")
    fine_cycles = 8 if target_family == "base" else 32
    encoder_type = ConvexQuadImageEncoder if method == "A2" else ConvexQuadLocalImageEncoder
    encoder = encoder_type(
        args.side,
        width=state["args"].get("a2_width", 8),
        head_mode=state["args"].get("a2_head_mode", "multilevel"),
        body_mode=state["args"].get("a2_body_mode", "local"),
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    decoder = (
        HierarchicalConvexQuadFreeCenterLayer(args.side) if method == "A2"
        else HierarchicalConvexQuadLocalLayer(args.side, motion_mode="radial" if method in ("A4", "A5", "A6") else "disk")
    ).to(device)
    fixed, moving, true_map, coefficients = (
        value.to(device)
        for value in make_dataset(
            args.test_count, args.image_side, args.test_seed,
            return_coefficients=True, target_family=target_family,
        )
    )
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    vertices = torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    image_error_sum = 0.0
    pixel_map_error_sum = 0.0
    face_map_error_sum = 0.0
    jacobian_error_sum = 0.0
    mu_error_sum = 0.0
    sampled_target_mu_error_sum = 0.0
    predicted_to_sampled_target_mu_error_sum = 0.0
    count = 0
    min_area = float("inf")
    min_determinant = float("inf")
    max_predicted_mu = 0.0
    max_target_mu = 0.0
    fine_projection_estimates = []
    began = time.perf_counter()
    with torch.no_grad():
        for start in range(0, args.test_count, args.batch):
            end = min(start + args.batch, args.test_count)
            current = end - start
            pair = torch.cat((fixed[start:end], moving[start:end]), dim=1)
            latent = encoder(pair)
            if method in ("A5", "A6"):
                base = decoder.base(latent[0], latent[1])
                hint = local_photometric_logits(
                    fixed[start:end], moving[start:end], base,
                    window=state["args"]["hint_window"],
                    ridge=state["args"]["hint_ridge"],
                    raw_span=decoder.local.raw_span,
                )
                if method == "A6":
                    hint = spectralize_bounded_logits(
                        hint, side=args.side, raw_span=decoder.local.raw_span,
                        count=state["args"]["hint_sine_modes"],
                    )
                control = decoder.local(base, latent[2] + state["args"]["hint_gain"] * hint)
            else:
                control = decoder(*latent)
            if target_family == "high32":
                source_grid = vertices.reshape(args.side, args.side, 2)
                high_basis = torch.sin(64 * math.pi * source_grid[..., 0]) * torch.sin(64 * math.pi * source_grid[..., 1])
                displacement = control - source_grid
                projected = (displacement * high_basis[None, :, :, None]).sum(dim=(1, 2)) / high_basis.square().sum()
                for index in range(current):
                    fine_projection_estimates.append({
                        "true_amplitude": coefficients[start + index, 2].item(),
                        "projected_x_amplitude": projected[index, 0].item(),
                        "projected_y_amplitude": projected[index, 1].item(),
                    })
            min_area = min(min_area, certify_convex_quad_output(control))
            dense = table.interpolate(control.reshape(current, -1, 2))
            warped = F.grid_sample(moving[start:end], 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
            image_error_sum += (warped - fixed[start:end]).square().mean().item() * current
            pixel_map_error_sum += (dense - true_map[start:end]).square().mean().item() * current
            query = centroids[None].expand(current, -1, -1)
            predicted_value, predicted_jacobian = evaluate_structured_p1_with_jacobian(control, query)
            target_value, target_jacobian = _target_on_faces(query, coefficients[start:end], fine_cycles)
            sampled_target, _ = _target_on_faces(vertices[None].expand(current, -1, -1), coefficients[start:end], fine_cycles)
            _, sampled_target_jacobian = evaluate_structured_p1_with_jacobian(
                sampled_target.reshape(current, args.side, args.side, 2), query
            )
            face_map_error_sum += (predicted_value - target_value).square().mean().item() * current
            jacobian_error_sum += (predicted_jacobian - target_jacobian).square().mean().item() * current
            predicted_mu = _mu(predicted_jacobian)
            target_mu = _mu(target_jacobian)
            sampled_target_mu = _mu(sampled_target_jacobian)
            mu_error_sum += (predicted_mu - target_mu).abs().square().mean().item() * current
            sampled_target_mu_error_sum += (sampled_target_mu - target_mu).abs().square().mean().item() * current
            predicted_to_sampled_target_mu_error_sum += (predicted_mu - sampled_target_mu).abs().square().mean().item() * current
            determinant = predicted_jacobian[..., 0, 0] * predicted_jacobian[..., 1, 1] - predicted_jacobian[..., 0, 1] * predicted_jacobian[..., 1, 0]
            min_determinant = min(min_determinant, determinant.min().item())
            max_predicted_mu = max(max_predicted_mu, predicted_mu.abs().max().item())
            max_target_mu = max(max_target_mu, target_mu.abs().max().item())
            count += current
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "method": f"{method}_image_trained",
        "target_family": target_family,
        "representation": "original_grid_P1",
        "control_side": args.side,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "test_count": count,
        "test_seed": args.test_seed,
        "batch": args.batch,
        "dtype": "float32",
        "device": str(device),
        "heldout_image_mse": image_error_sum / count,
        "heldout_pixel_query_map_rmse": math.sqrt(pixel_map_error_sum / count),
        "heldout_face_centroid_map_rmse": math.sqrt(face_map_error_sum / count),
        "heldout_face_jacobian_component_rmse": math.sqrt(jacobian_error_sum / count),
        "heldout_source_area_weighted_beltrami_rmse": math.sqrt(mu_error_sum / count),
        "heldout_sampled_target_P1_beltrami_discretization_rmse": math.sqrt(sampled_target_mu_error_sum / count),
        "heldout_predicted_vs_sampled_target_P1_beltrami_rmse": math.sqrt(predicted_to_sampled_target_mu_error_sum / count),
        "minimum_source_normalized_face_area": min_area,
        "minimum_face_jacobian_determinant": min_determinant,
        "maximum_predicted_beltrami_modulus": max_predicted_mu,
        "maximum_target_beltrami_modulus": max_target_mu,
        "fine_projection_estimates": fine_projection_estimates if target_family == "high32" else None,
        "evaluation_seconds": time.perf_counter() - began,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
