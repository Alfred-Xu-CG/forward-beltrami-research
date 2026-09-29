"""Group-held-out same-modality deformation capability probe for safe Q1 network.

Synthetic fixed images are made from whole-slide thumbnails and known smooth
boundary-fixed maps. The analytic map supervises training but is never given
to the encoder. This is not evidence of cross-stain real registration utility.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers, warp_moving_at_q1_map
from tools.digital_q1_network_teacher import _save_checkpoint


def sine_basis(points: torch.Tensor) -> torch.Tensor:
    """Three boundary-zero basis functions, shape (...,3)."""
    x, y = points[..., 0], points[..., 1]
    pi = torch.pi
    return torch.stack((
        torch.sin(pi * x) * torch.sin(pi * y),
        torch.sin(2 * pi * x) * torch.sin(pi * y),
        torch.sin(pi * x) * torch.sin(2 * pi * y),
    ), dim=-1)


def point_grid(side: int, *, centers: bool, device: torch.device) -> torch.Tensor:
    if centers:
        return fixed_pixel_centers(side, side, dtype=torch.float32, device=device)
    axis = torch.arange(side, dtype=torch.float32, device=device) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def analytic_map(points: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate a six-coefficient smooth map at arbitrary normalized points."""
    basis = sine_basis(points)
    displacement = torch.einsum("bhwk,bck->bhwc", basis.expand(coefficients.shape[0], -1, -1, -1),
                                coefficients.reshape(-1, 2, 3))
    return points.expand(coefficients.shape[0], -1, -1, -1) + displacement


def make_coefficients(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    amplitudes = torch.tensor((.018, .008, .008, .018, .008, .008))
    return (2 * torch.rand((count, 6), generator=generator) - 1) * amplitudes


def synthesize(moving: torch.Tensor, coefficients: torch.Tensor,
               pixel_points: torch.Tensor) -> torch.Tensor:
    coordinates = analytic_map(pixel_points, coefficients)
    return F.grid_sample(moving, 2 * coordinates - 1, mode="bilinear",
                         padding_mode="border", align_corners=False)


def _batch(templates: torch.Tensor, coefficients: torch.Tensor,
           indices: torch.Tensor, pixel_points: torch.Tensor,
           vertex_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    moving = templates[indices % templates.shape[0]]
    selected = coefficients[indices]
    fixed = synthesize(moving, selected, pixel_points)
    target = analytic_map(vertex_points, selected)
    return fixed, moving, target


def run(*, textures: torch.Tensor, train_texture_indices: list[int],
        test_texture_indices: list[int], side: int, steps: int, batch: int,
        train_count: int, test_count: int, device: str,
        output_example: Path | None = None,
        output_weights: Path | None = None,
        loss_mode: str = "map") -> dict:
    if textures.ndim != 4 or textures.shape[1] != 1 or textures.shape[-1] != textures.shape[-2]:
        raise ValueError("textures must have shape (T,1,S,S)")
    if not train_texture_indices or not test_texture_indices or (
        set(train_texture_indices) & set(test_texture_indices)
    ):
        raise ValueError("train and test texture groups must be nonempty and disjoint")
    if any(index < 0 or index >= textures.shape[0]
           for index in (*train_texture_indices, *test_texture_indices)):
        raise ValueError("texture indices must be canonical nonnegative indices")
    if min(side, steps, batch, train_count, test_count) < 1:
        raise ValueError("positive dimensions and counts required")
    if loss_mode not in {"map", "image"}:
        raise ValueError("loss_mode must be map or image")
    torch.manual_seed(291001)
    target_device = torch.device(device)
    textures = textures.to(target_device, dtype=torch.float32)
    image_side = int(textures.shape[-1])
    pixel_points = point_grid(image_side, centers=True, device=target_device)
    vertex_points = point_grid(side, centers=False, device=target_device)
    train_textures = textures[train_texture_indices]
    test_textures = textures[test_texture_indices]
    train_coeff = make_coefficients(train_count, 291002).to(target_device)
    test_coeff = make_coefficients(test_count, 291003).to(target_device)
    model = Q1ImageRegistrationNetwork(seed_side=17, final_side=side,
                                       width=16, feature_side=min(side, 257),
                                       flow_hint=True).to(target_device)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=.002)
    rng = torch.Generator(device="cpu").manual_seed(291004)
    times = []
    losses = []
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    model.train()
    for step in range(steps):
        indices = torch.randint(train_count, (batch,), generator=rng).to(target_device)
        fixed, moving, target = _batch(train_textures, train_coeff, indices,
                                       pixel_points, vertex_points)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _ = model(fixed, moving)
        if loss_mode == "map":
            loss = (prediction - target).square().sum(-1).mean()
        else:
            warped = warp_moving_at_q1_map(moving, prediction,
                                           height=image_side, width=image_side)
            loss = (warped - fixed).square().mean()
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        if not grads or any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError(f"nonfinite encoder gradient at step {step}")
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        times.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))
    training_peak = (torch.cuda.max_memory_allocated(target_device)
                     if target_device.type == "cuda" else None)
    model.eval()
    results = []
    cases = []
    example = None
    with torch.no_grad():
        for start in range(0, test_count, batch):
            indices = torch.arange(start, min(start + batch, test_count), device=target_device)
            fixed, moving, target = _batch(test_textures, test_coeff, indices,
                                           pixel_points, vertex_points)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            timed = time.perf_counter()
            prediction, _, _ = model(fixed, moving)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            inference_time = (time.perf_counter() - timed) / indices.numel()
            validity = validate_q1_map(
                prediction, vertex_points.expand(prediction.shape[0], -1, -1, -1)
            )
            if validity["nonpositive_corners"] or validity["boundary_max_error"]:
                raise AssertionError("predicted Q1 geometry invalid")
            warped = warp_moving_at_q1_map(moving, prediction,
                                           height=image_side, width=image_side)
            for local in range(indices.numel()):
                predicted_rmse = float((prediction[local] - target[local]).square().sum(-1).mean().sqrt())
                identity_rmse = float((vertex_points[0] - target[local]).square().sum(-1).mean().sqrt())
                image_mse = float((warped[local] - fixed[local]).square().mean())
                identity_image_mse = float((moving[local] - fixed[local]).square().mean())
                results.append((predicted_rmse, identity_rmse, image_mse,
                                identity_image_mse, inference_time))
                cases.append({
                    "sample_index": int(indices[local]),
                    "texture_index": test_texture_indices[int(indices[local]) % len(test_texture_indices)],
                    "coefficients": test_coeff[indices[local]].cpu().tolist(),
                    "map_rmse": predicted_rmse,
                    "identity_rmse": identity_rmse,
                    "image_mse": image_mse,
                    "identity_image_mse": identity_image_mse,
                })
            if example is None:
                example = prediction[0:1].cpu().numpy().copy()
    saved_certificate = None
    if output_example is not None:
        assert example is not None
        output_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_example, vertices=example,
                            boundary_reference=vertex_points.cpu().numpy())
        saved_certificate = certify_q1_binary_map(output_example)
    if output_weights is not None:
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(output_weights, model)
    return {
        "question": "can a safe image-to-Q1 encoder recover unseen synthetic warps on held-out real-slide textures?",
        "scope": "same-modality synthetic deformation; not real cross-stain registration",
        "train_texture_indices": train_texture_indices,
        "test_texture_indices": test_texture_indices,
        "control_side": side,
        "image_side": image_side,
        "steps": steps,
        "batch": batch,
        "train_count": train_count,
        "test_count": test_count,
        "training_loss_mode": loss_mode,
        "first_train_objective": losses[0],
        "last_train_objective": losses[-1],
        "train_objective_batches_differ": True,
        "test_map_rmse_mean": statistics.mean(x[0] for x in results),
        "test_identity_rmse_mean": statistics.mean(x[1] for x in results),
        "test_image_mse_mean": statistics.mean(x[2] for x in results),
        "test_identity_image_mse_mean": statistics.mean(x[3] for x in results),
        "train_step_seconds_median": statistics.median(times),
        "inference_seconds_per_image_median": statistics.median(x[4] for x in results),
        "training_peak_cuda_allocated_bytes": training_peak,
        "saved_example_certificate": saved_certificate,
        "test_cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textures", type=Path, required=True,
                        help="D-drive NPZ with float32 texture tensor under key images")
    parser.add_argument("--train-texture-index", type=int, action="append", required=True)
    parser.add_argument("--test-texture-index", type=int, action="append", required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-example", type=Path)
    parser.add_argument("--output-weights", type=Path)
    parser.add_argument("--loss-mode", choices=("map", "image"), default="map")
    args = parser.parse_args()
    with np.load(args.textures) as archive:
        textures = torch.from_numpy(archive["images"].copy())
        sources = archive["sources"].tolist() if "sources" in archive else None
    report = run(textures=textures,
                 train_texture_indices=args.train_texture_index,
                 test_texture_indices=args.test_texture_index,
                 side=args.side, steps=args.steps, batch=args.batch,
                 train_count=args.train_count, test_count=args.test_count,
                 device=args.device, output_example=args.output_example,
                 output_weights=args.output_weights, loss_mode=args.loss_mode)
    report["texture_sources"] = sources
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "test_cases"}, indent=2))


if __name__ == "__main__":
    main()
