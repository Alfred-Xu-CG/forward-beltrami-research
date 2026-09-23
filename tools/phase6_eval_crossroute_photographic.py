"""Frozen B/C layers on identical photographic content and known high32 maps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from phase6_eval_photographic_content import _dataset
from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_spectral_edge_encoder import SpectralEdgeImageEncoder
from phase6_train_conductance_image_to_latent import MultiscaleEdgeImageEncoder
from phase6_train_multisample_image import CoarseFineConvexQuadImageEncoder
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarseFineConvexQuadComposition, SinePreconditionedTutteLayer,
    evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _edges_and_midpoints(side: int, vertices: np.ndarray) -> np.ndarray:
    grid = np.arange(side**2, dtype=np.int64).reshape(side, side)
    edges = np.concatenate((
        np.stack((grid[:, :-1].ravel(), grid[:, 1:].ravel()), axis=1),
        np.stack((grid[:-1].ravel(), grid[1:].ravel()), axis=1),
        np.stack((grid[:-1, :-1].ravel(), grid[1:, 1:].ravel()), axis=1),
    ))
    return vertices[edges].mean(axis=1)


def _minimum_face_area(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(cross(b-a, c-a).amin(),
                               cross(c-a, d-a).amin()) * (mapped.shape[1]-1)**2)


def _models(args, side: int, mesh, device):
    state_b = torch.load(args.b_checkpoint, map_location=device, weights_only=False)
    settings_b = state_b["args"]
    if settings_b["method"] != "CF2" or settings_b["side"] != side or settings_b["coarse_side"] != side:
        raise ValueError("B checkpoint must be a same-side two-fine CF2 model")
    encoder_b = CoarseFineConvexQuadImageEncoder(
        side, side, width=settings_b["a2_width"],
        head_mode=settings_b["a2_head_mode"], body_mode=settings_b["a2_body_mode"],
    ).to(device)
    encoder_b.load_state_dict(state_b["encoder"])
    encoder_b.eval()
    decoder_b = CoarseFineConvexQuadComposition(side, side, args.image_side)
    decoder_b.prepare(device=device, dtype=torch.float32)

    midpoints = _edges_and_midpoints(side, mesh.vertices)
    c_models = {}
    for label, path in (("C_plain", args.c_plain_checkpoint),
                        ("C_spectral", args.c_spectral_checkpoint)):
        state = torch.load(path, map_location=device, weights_only=False)
        settings = state["args"]
        freqs = tuple(int(x) for x in settings.get("spectral_frequencies", "").split(",") if x)
        encoder = (
            SpectralEdgeImageEncoder(side, midpoints, body_mode=settings["encoder_body"],
                                     frequencies=freqs)
            if freqs else
            MultiscaleEdgeImageEncoder(side, midpoints, body_mode=settings["encoder_body"])
        ).to(device)
        encoder.load_state_dict(state["encoder"])
        encoder.eval()
        solver = SinePreconditionedTutteLayer(
            side, minimum_conductance=1, maximum_conductance=16,
            tolerance=1e-10, max_iterations=120,
        ).to(device)
        c_models[label] = (encoder, solver, path)
    return (encoder_b, decoder_b, args.b_checkpoint), c_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-checkpoint", required=True)
    parser.add_argument("--c-plain-checkpoint", required=True)
    parser.add_argument("--c-spectral-checkpoint", required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--variants-per-photo", type=int, default=16)
    parser.add_argument("--seed", type=int, default=973031)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    side = 257
    names, photo_ids, dataset = _dataset(args.variants_per_photo, args.image_side,
                                         args.seed, device)
    mesh = structured_rectangle(side-1, side-1)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    model_b, models_c = _models(args, side, mesh, device)
    models = {}
    for label in ("B_twofine", "C_plain", "C_spectral"):
        encoder, solver, checkpoint = model_b if label == "B_twofine" else models_c[label]
        rows, times = [], []
        with torch.no_grad():
            for begin in range(0, len(photo_ids), args.batch):
                end = min(begin+args.batch, len(photo_ids))
                fixed, moving, target, coeff = (part[begin:end] for part in dataset)
                pair = torch.cat((fixed, moving), dim=1)
                if device.type == "cuda": torch.cuda.synchronize(device)
                started = time.perf_counter()
                if label == "B_twofine":
                    latent = encoder(pair)
                    result = solver(*latent[0], *latent[1])
                    predicted = result.dense
                    controls = result.controls
                    factor_areas = [_minimum_face_area(value) for value in controls]
                    query = centroids[None].expand(end-begin, -1, -1)
                    middle, jac_first = evaluate_structured_p1_with_jacobian(
                        controls[0], query)
                    _, jac_second = evaluate_structured_p1_with_jacobian(
                        controls[1], middle)
                    jacobian = jac_second @ jac_first
                    solver_residual = None
                else:
                    flat = encoder(pair).double()
                    count_h = side*(side-1)
                    count_v = count_h
                    mapped = solver(
                        flat[:, :count_h].reshape(end-begin, side, side-1),
                        flat[:, count_h:count_h+count_v].reshape(end-begin, side-1, side),
                        flat[:, count_h+count_v:].reshape(end-begin, side-1, side-1),
                    ).float()
                    predicted = table.interpolate(mapped.reshape(end-begin, -1, 2))
                    controls = (mapped,)
                    factor_areas = [_minimum_face_area(mapped)]
                    query = centroids[None].expand(end-begin, -1, -1)
                    _, jacobian = evaluate_structured_p1_with_jacobian(mapped, query)
                    solver_residual = solver.last_forward_stats["true_relative_residual"]
                warped = F.grid_sample(moving, 2*predicted-1, mode="bilinear",
                                       padding_mode="border", align_corners=True)
                if device.type == "cuda": torch.cuda.synchronize(device)
                times.append(time.perf_counter()-started)
                _, target_jacobian = _target_on_faces(query, coeff, 32)
                predicted_mu = _mu(jacobian)
                mu_mse = (predicted_mu - _mu(target_jacobian)).abs().square().mean(dim=1)
                determinants = torch.linalg.det(jacobian)
                image_mse = (warped-fixed).square().mean(dim=(1,2,3))
                map_mse = (predicted-target).square().mean(dim=(1,2,3))
                for index in range(end-begin):
                    rows.append({
                        "photo": names[photo_ids[begin+index]],
                        "image_mse": float(image_mse[index]),
                        "query_map_mse": float(map_mse[index]),
                        "source_centroid_beltrami_mse": float(mu_mse[index]),
                        "maximum_source_centroid_beltrami_modulus": float(predicted_mu[index].abs().amax()),
                        "minimum_source_centroid_determinant": float(determinants[index].amin()),
                        "minimum_factor_face_area_ratios_in_batch": factor_areas,
                        "solver_true_relative_residual_in_batch": solver_residual,
                    })
        def aggregate(items):
            return {
                "count": len(items),
                "image_mse": statistics.mean(item["image_mse"] for item in items),
                "query_map_rmse": math.sqrt(statistics.mean(item["query_map_mse"] for item in items)),
                "source_centroid_beltrami_rmse": math.sqrt(statistics.mean(
                    item["source_centroid_beltrami_mse"] for item in items)),
                "maximum_source_centroid_beltrami_modulus": max(
                    item["maximum_source_centroid_beltrami_modulus"] for item in items),
                "minimum_source_centroid_determinant": min(
                    item["minimum_source_centroid_determinant"] for item in items),
                "minimum_factor_face_area_ratios": [min(
                    item["minimum_factor_face_area_ratios_in_batch"][factor]
                    for item in items) for factor in range(2 if label == "B_twofine" else 1)],
                "maximum_solver_true_relative_residual": (
                    max(item["solver_true_relative_residual_in_batch"] for item in items)
                    if label != "B_twofine" else None),
            }
        models[label] = {
            "checkpoint": checkpoint,
            "representation": "exact_two_factor_PL" if label == "B_twofine" else "original_grid_P1",
            "aggregate": aggregate(rows),
            "by_photo": {name: aggregate([row for row in rows if row["photo"] == name])
                         for name in names},
            "per_sample": rows,
            "mean_batch_forward_seconds": statistics.mean(times),
        }
    print(json.dumps({
        "method": "frozen_B_and_C_photographic_content_synthetic_high32_deformation",
        "photo_source": "installed_scikit_image_sample_images",
        "photos": names,
        "seed": args.seed,
        "variants_per_photo": args.variants_per_photo,
        "control_side": side,
        "control_vertices": side**2,
        "factor_or_control_faces": 2*(side-1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "device": str(device),
        "models": models,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
