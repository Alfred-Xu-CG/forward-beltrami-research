"""Evaluate source-face Jacobians of an image-conditioned exact CF2 map."""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_multisample_image import (
    CoarseFineConvexQuadImageEncoder, make_dataset,
)
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarseFineConvexQuadComposition, certify_convex_quad_output,
    evaluate_structured_p1_with_jacobian,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-family", choices=("high32", "high64"),
                        default="high64")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--face-batch", type=int, default=65536)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.count < 1 or args.face_batch < 1:
        raise ValueError("count and face batch must be positive")
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    settings = state["args"]
    if settings["method"] != "CF2":
        raise ValueError("expected image-conditioned CF2 checkpoint")
    coarse_side, fine_side = settings["coarse_side"], settings["side"]
    image_side = settings["image_side"]
    encoder = CoarseFineConvexQuadImageEncoder(
        coarse_side, fine_side, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"],
        body_mode=settings["a2_body_mode"],
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    decoder = CoarseFineConvexQuadComposition(
        coarse_side, fine_side, image_side)
    decoder.prepare(device=device, dtype=torch.float32)
    dataset = tuple(value.to(device) for value in make_dataset(
        args.count, image_side, 99317, return_coefficients=True,
        target_family=args.target_family))
    mesh = structured_rectangle(fine_side - 1, fine_side - 1)
    vertices = torch.tensor(mesh.vertices.copy(), device=device,
                            dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.long)
    centroids = vertices[faces].mean(dim=1)
    cycles = 64 if args.target_family == "high64" else 32
    rows = []
    with torch.no_grad():
        for sample in range(args.count):
            fixed, moving, target, coefficients = (
                part[sample:sample + 1] for part in dataset)
            latent = encoder(torch.cat((fixed, moving), dim=1))
            result = decoder(*latent[0], *latent[1])
            coarse, fine = result.controls
            warped = F.grid_sample(
                moving, 2 * result.dense - 1, mode="bilinear",
                padding_mode="border", align_corners=True)
            total_mu_mse = total_map_mse = 0.0
            minimum_chain_det = float("inf")
            maximum_chain_mu = 0.0
            for start in range(0, len(centroids), args.face_batch):
                query = centroids[start:start + args.face_batch][None]
                middle, jac_coarse = evaluate_structured_p1_with_jacobian(
                    coarse, query)
                composed, jac_fine = evaluate_structured_p1_with_jacobian(
                    fine, middle)
                jacobian = jac_fine @ jac_coarse
                true_value, true_jacobian = _target_on_faces(
                    query, coefficients, cycles)
                predicted_mu, true_mu = _mu(jacobian), _mu(true_jacobian)
                total_mu_mse += float((predicted_mu - true_mu).abs().square().sum())
                total_map_mse += float((composed - true_value).square().sum())
                minimum_chain_det = min(
                    minimum_chain_det, float(torch.linalg.det(jacobian).min()))
                maximum_chain_mu = max(
                    maximum_chain_mu, float(predicted_mu.abs().max()))
            rows.append({
                "image_mse": float((warped - fixed).square().mean()),
                "query_map_mse": float((result.dense - target).square().mean()),
                "source_centroid_map_mse": total_map_mse / (2 * len(centroids)),
                "source_centroid_beltrami_mse": total_mu_mse / len(centroids),
                "minimum_sampled_chain_determinant": minimum_chain_det,
                "maximum_sampled_chain_beltrami_modulus": maximum_chain_mu,
                "minimum_factor_area_ratios": [
                    certify_convex_quad_output(coarse),
                    certify_convex_quad_output(fine),
                ],
            })
    print(json.dumps({
        "method": "image_conditioned_CF2_exact_PL_composition_geometry",
        "checkpoint": args.checkpoint,
        "target_family": args.target_family,
        "count": args.count,
        "coarse_side": coarse_side,
        "fine_side": fine_side,
        "fine_control_vertices": fine_side**2,
        "fine_control_faces": 2 * (fine_side - 1)**2,
        "source_centroid_queries_per_sample": len(centroids),
        "image_queries": image_side**2,
        "device": str(device),
        "image_mse": sum(row["image_mse"] for row in rows) / args.count,
        "query_map_rmse": math.sqrt(
            sum(row["query_map_mse"] for row in rows) / args.count),
        "source_centroid_map_rmse": math.sqrt(
            sum(row["source_centroid_map_mse"] for row in rows) / args.count),
        "source_centroid_beltrami_rmse": math.sqrt(
            sum(row["source_centroid_beltrami_mse"] for row in rows) / args.count),
        "minimum_sampled_chain_determinant": min(
            row["minimum_sampled_chain_determinant"] for row in rows),
        "maximum_sampled_chain_beltrami_modulus": max(
            row["maximum_sampled_chain_beltrami_modulus"] for row in rows),
        "minimum_factor_area_ratios": [min(row["minimum_factor_area_ratios"][k]
                                           for row in rows) for k in range(2)],
        "samples": rows,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
