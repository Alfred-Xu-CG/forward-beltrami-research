"""Evaluate an image-derived A6 map after one positive-conductance projection."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    HierarchicalConvexQuadLocalLayer, SinePreconditionedTutteLayer,
    evaluate_structured_p1_with_jacobian, local_photometric_logits,
    keep_vector_sine_modes, spectralize_bounded_logits, synthesize_bounded_conductances,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--projection-sine-modes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.projection_sine_modes < 0 or args.projection_sine_modes > (args.side - 2)**2:
        raise ValueError("projection sine count outside interior grid")
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    settings = state["args"]
    if settings["method"] != "A6" or settings["side"] != args.side or settings["target_family"] != "high32":
        raise ValueError("expected a 257² high32 A6 checkpoint")
    encoder = ConvexQuadLocalImageEncoder(
        args.side, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"], body_mode=settings["a2_body_mode"],
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    decoder = HierarchicalConvexQuadLocalLayer(args.side, motion_mode="radial").to(device)
    solver = SinePreconditionedTutteLayer(args.side, maximum_conductance=16.0,
                                           tolerance=1e-10, max_iterations=120).to(device)
    fixed, moving, target, coefficients = (
        value.to(device) for value in make_dataset(
            args.test_count, args.image_side, args.test_seed,
            target_family="high32", return_coefficients=True,
        )
    )
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    def forward(fixed_batch, moving_batch, *, stats=False):
        latent = encoder(torch.cat((fixed_batch, moving_batch), dim=1))
        base = decoder.base(latent[0], latent[1])
        hint = local_photometric_logits(
            fixed_batch, moving_batch, base, window=settings["hint_window"],
            ridge=settings["hint_ridge"], raw_span=decoder.local.raw_span,
        )
        hint = spectralize_bounded_logits(
            hint, side=args.side, raw_span=decoder.local.raw_span,
            count=settings["hint_sine_modes"],
        )
        a6 = decoder.local(base, latent[2] + settings["hint_gain"] * hint)
        candidate = a6
        if args.projection_sine_modes:
            source_grid = solver._source.to(a6.dtype)[None]
            displacement = a6 - source_grid
            filtered = keep_vector_sine_modes(displacement[:, 1:-1, 1:-1], args.projection_sine_modes)
            candidate = source_grid + F.pad(filtered.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)
        edge_logits, synthesis_stats = synthesize_bounded_conductances(
            candidate.double(), passes=1, gauge="range", return_stats=stats,
        )
        c_map = solver(*edge_logits).float()
        a6_query = table.interpolate(a6.reshape(len(fixed_batch), -1, 2))
        c_query = table.interpolate(c_map.reshape(len(fixed_batch), -1, 2))
        candidate_query = table.interpolate(candidate.reshape(len(fixed_batch), -1, 2))
        return a6, candidate, c_map, a6_query, candidate_query, c_query, synthesis_stats
    def score(control, query, fixed_batch, moving_batch, target_batch, coeff_batch):
        warped = F.grid_sample(moving_batch, 2 * query - 1,
                               mode="bilinear", padding_mode="border", align_corners=True)
        current = len(fixed_batch)
        face_query = centroids[None].expand(current, -1, -1)
        _, predicted_jacobian = evaluate_structured_p1_with_jacobian(control, face_query)
        _, target_jacobian = _target_on_faces(face_query, coeff_batch, 32)
        predicted_mu, target_mu = _mu(predicted_jacobian), _mu(target_jacobian)
        a = control[:, :-1, :-1]
        b = control[:, :-1, 1:]
        c = control[:, 1:, 1:]
        d = control[:, 1:, :-1]
        def cross(first, second):
            return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        minimum = torch.minimum(cross(b - a, c - a).amin(), cross(c - a, d - a).amin()) * (args.side - 1)**2
        source_grid = vertices.reshape(args.side, args.side, 2)
        basis = torch.sin(64 * math.pi * source_grid[..., 0]) * torch.sin(64 * math.pi * source_grid[..., 1])
        projected = ((control - source_grid) * basis[None, :, :, None]).sum(dim=(1, 2)) / basis.square().sum()
        return {
            "image_mse": (warped - fixed_batch).square().mean().item(),
            "query_map_rmse": (query - target_batch).square().mean().sqrt().item(),
            "face_beltrami_rmse": (predicted_mu - target_mu).abs().square().mean().sqrt().item(),
            "maximum_predicted_beltrami_modulus": predicted_mu.abs().max().item(),
            "minimum_signed_area_ratio": minimum.item(),
            "projected_fine_amplitudes": projected.detach().cpu().tolist(),
        }
    scores = {"a6": [], "spectral_candidate": [], "conductance_projection": []}
    clipping = []
    with torch.no_grad():
        for start in range(0, args.test_count, args.batch):
            stop = min(start + args.batch, args.test_count)
            a6, candidate, c_map, a6_query, candidate_query, c_query, stats = forward(fixed[start:stop], moving[start:stop], stats=True)
            scores["a6"].append(score(a6, a6_query, fixed[start:stop], moving[start:stop], target[start:stop], coefficients[start:stop]))
            scores["spectral_candidate"].append(score(candidate, candidate_query, fixed[start:stop], moving[start:stop], target[start:stop], coefficients[start:stop]))
            scores["conductance_projection"].append(score(c_map, c_query, fixed[start:stop], moving[start:stop], target[start:stop], coefficients[start:stop]))
            clipping.append(stats["passes"][0])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times = [], []
    first_gradient_norm = None
    for repeat in range(args.repeats + 1):
        encoder.zero_grad(set_to_none=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        began = time.perf_counter()
        _, _, _, _, _, query, _ = forward(fixed[:args.batch], moving[:args.batch])
        warped = F.grid_sample(moving[:args.batch], 2 * query - 1,
                               mode="bilinear", padding_mode="border", align_corners=True)
        loss = (warped - fixed[:args.batch]).square().mean()
        if device.type == "cuda": torch.cuda.synchronize(device)
        middle = time.perf_counter()
        loss.backward()
        if device.type == "cuda": torch.cuda.synchronize(device)
        finished = time.perf_counter()
        if repeat:
            forward_times.append(middle - began)
            backward_times.append(finished - middle)
            if first_gradient_norm is None:
                first_gradient_norm = float(torch.linalg.vector_norm(torch.cat([
                    parameter.grad.reshape(-1) for parameter in encoder.parameters() if parameter.grad is not None
                ])))
    print(json.dumps({
        "method": "frozen_A6_image_map_to_one_pass_bounded_conductance",
        "checkpoint": args.checkpoint,
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "projection_sine_modes": args.projection_sine_modes,
        "batch": args.batch,
        "device": str(device),
        "dtype": "float32_image_float64_conductance_solver",
        "a6_scores_per_batch": scores["a6"],
        "spectral_candidate_scores_per_batch": scores["spectral_candidate"],
        "conductance_projection_scores_per_batch": scores["conductance_projection"],
        "conductance_clipping_per_batch": clipping,
        "last_solver_stats": dict(solver.last_forward_stats),
        "first_encoder_gradient_norm_from_projected_image_loss": first_gradient_norm,
        "median_full_forward_seconds": statistics.median(forward_times),
        "median_full_vjp_seconds": statistics.median(backward_times),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
