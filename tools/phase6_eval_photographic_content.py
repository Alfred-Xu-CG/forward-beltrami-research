"""Frozen dense layers on real image content with known synthetic high32 maps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from skimage import data

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_multisample_image import ConvexQuadLocalImageEncoder
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    SpectralSafeFeedbackLayer, evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _photo_bank(side: int, names: list[str] | None = None) -> tuple[list[str], torch.Tensor]:
    if names is None:
        names = ["camera", "coins", "moon", "page", "grass", "gravel"]
    if not names:
        raise ValueError("at least one photographic image is required")
    images = []
    for name in names:
        array = np.asarray(getattr(data, name)(), dtype=np.float32)
        if array.ndim == 3:
            array = (array[..., :3] * np.array(
                [0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(axis=-1)
        if array.ndim != 2:
            raise ValueError(f"photograph {name} must be grayscale or RGB")
        image = torch.tensor(array)[None, None] / 255.0
        image = F.interpolate(image, size=(side, side), mode="bilinear", align_corners=True)
        image = 0.3 * (image - image.mean()) / image.std().clamp_min(1e-6)
        images.append(image[0])
    return names, torch.stack(images)


def _dataset(variants_per_photo: int, image_side: int, seed: int, device: torch.device,
             photo_names: list[str] | None = None,
             target_family: str = "high32"):
    if target_family not in ("high32", "high64"):
        raise ValueError("photographic target family must be high32 or high64")
    names, bank = _photo_bank(image_side, photo_names)
    count = len(names) * variants_per_photo
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ax = 0.005 + 0.010 * torch.rand(count, 1, 1, generator=generator)
    ay = 0.010 + 0.015 * torch.rand(count, 1, 1, generator=generator)
    af = -0.002 + 0.0045 * torch.rand(count, 1, 1, generator=generator)
    if target_family == "high64":
        af = 0.5 * af
    line = torch.linspace(0, 1, image_side)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    low = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    cycles = 64 if target_family == "high64" else 32
    fine = torch.sin(2 * cycles * math.pi * xx) * torch.sin(
        2 * cycles * math.pi * yy)
    target = torch.stack((xx + ax * low + af * fine,
                          yy + ay * low + af * fine), dim=-1)
    source_indices = torch.arange(count) // variants_per_photo
    moving = bank[source_indices].contiguous()
    fixed = F.grid_sample(
        moving, 2 * target - 1, mode="bilinear",
        padding_mode="border", align_corners=True,
    ).detach()
    coefficients = torch.cat((ax, ay, af), dim=-1).reshape(count, 3)
    return names, source_indices.tolist(), tuple(
        value.to(device) for value in (fixed, moving, target, coefficients)
    )


def _model(path: str, device: torch.device, initial_qc_cap: float | None = None):
    state = torch.load(path, map_location=device, weights_only=False)
    settings = state["args"]
    side = settings["side"]
    encoder = ConvexQuadLocalImageEncoder(
        side, width=settings["a2_width"],
        head_mode=settings["a2_head_mode"],
        body_mode=settings["a2_body_mode"],
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    method = settings["method"]
    decoder = SpectralSafeFeedbackLayer(
        side, initial_window=settings["hint_window"],
        initial_ridge=settings["hint_ridge"],
        initial_gain=settings["hint_gain"],
        initial_modes=settings["hint_sine_modes"],
        extra_passes=0 if method == "A6" else settings.get("extra_passes", 1),
        extra_gain=settings.get("extra_gain", 0.25),
        extra_modes=16,
        extra_qc_cap=settings.get("extra_qc_cap"),
        initial_qc_cap=initial_qc_cap,
        floor_fraction=settings.get("floor_fraction", 0.2),
    ).to(device)
    decoder.eval()
    return method, encoder, decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-checkpoint", required=True)
    parser.add_argument("--a7-checkpoint", required=True)
    parser.add_argument("--a8-checkpoint", required=True)
    parser.add_argument("--variants-per-photo", type=int, default=16)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=973031)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--a8-initial-qc-cap", type=float, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    names, image_ids, dataset = _dataset(
        args.variants_per_photo, args.image_side, args.seed, device)
    side = 257
    mesh = structured_rectangle(side - 1, side - 1)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    models = {}
    for label, checkpoint in (
        ("A6", args.a6_checkpoint),
        ("A7", args.a7_checkpoint),
        ("A8", args.a8_checkpoint),
    ):
        method, encoder, decoder = _model(
            checkpoint, device,
            args.a8_initial_qc_cap if label == "A8" else None)
        if method != label:
            raise ValueError(f"checkpoint labelled {label} contains {method}")
        per_sample, forward_times = [], []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for start in range(0, len(image_ids), args.batch):
                stop = min(start + args.batch, len(image_ids))
                fixed, moving, target, coefficients = (
                    value[start:stop] for value in dataset
                )
                synchronize()
                began = time.perf_counter()
                latent = encoder(torch.cat((fixed, moving), dim=1))
                mapped = decoder(fixed, moving, latent)
                query = table.interpolate(mapped.reshape(stop - start, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query - 1, mode="bilinear",
                    padding_mode="border", align_corners=True,
                )
                synchronize()
                forward_times.append(time.perf_counter() - began)
                image_mse = (warped - fixed).square().mean(dim=(1, 2, 3))
                map_mse = (query - target).square().mean(dim=(1, 2, 3))
                face_query = centroids[None].expand(stop - start, -1, -1)
                _, jacobian = evaluate_structured_p1_with_jacobian(mapped, face_query)
                _, target_jacobian = _target_on_faces(face_query, coefficients, 32)
                predicted_mu = _mu(jacobian)
                mu_mse = (predicted_mu - _mu(target_jacobian)).abs().square().mean(dim=1)
                max_mu = predicted_mu.abs().amax(dim=1)
                determinant = torch.linalg.det(jacobian)
                min_area = determinant.amin(dim=1)
                initial_max = None
                if label == "A8":
                    initial = decoder.initial_map(fixed, moving, latent)
                    _, initial_jacobian = evaluate_structured_p1_with_jacobian(
                        initial, face_query)
                    initial_max = _mu(initial_jacobian).abs().amax(dim=1).tolist()
                for index in range(stop - start):
                    per_sample.append({
                        "photo": names[image_ids[start + index]],
                        "coefficients": coefficients[index].tolist(),
                        "image_mse": float(image_mse[index]),
                        "query_map_mse": float(map_mse[index]),
                        "face_beltrami_mse": float(mu_mse[index]),
                        "maximum_predicted_beltrami_modulus": float(max_mu[index]),
                        "minimum_face_determinant": float(min_area[index]),
                        "initial_maximum_beltrami_modulus": (
                            initial_max[index] if initial_max is not None else None),
                    })
        def aggregate(items):
            return {
                "count": len(items),
                "image_mse": statistics.mean(value["image_mse"] for value in items),
                "query_map_rmse": math.sqrt(statistics.mean(
                    value["query_map_mse"] for value in items)),
                "face_beltrami_rmse": math.sqrt(statistics.mean(
                    value["face_beltrami_mse"] for value in items)),
                "maximum_predicted_beltrami_modulus": max(
                    value["maximum_predicted_beltrami_modulus"] for value in items),
                "minimum_face_determinant": min(
                    value["minimum_face_determinant"] for value in items),
                "initial_qc_cap_eligible": (
                    sum(value["initial_maximum_beltrami_modulus"] < 0.8 for value in items)
                    if label == "A8" else None),
            }
        models[label] = {
            "checkpoint": checkpoint,
            "aggregate": aggregate(per_sample),
            "by_photo": {
                name: aggregate([value for value in per_sample if value["photo"] == name])
                for name in names
            },
            "per_sample": per_sample,
            "mean_batch_forward_seconds": statistics.mean(forward_times),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        }
    print(json.dumps({
        "method": "frozen_A6_A7_A8_photographic_content_synthetic_high32_deformation",
        "photo_source": "installed_scikit_image_sample_images",
        "photos": names,
        "preprocessing": "bilinear_resize_512_then_per_image_zero_mean_std_0p3",
        "target_family": "same_high32_coefficient_ranges_new_seed",
        "seed": args.seed,
        "variants_per_photo": args.variants_per_photo,
        "control_side": side,
        "control_vertices": side**2,
        "control_faces": 2 * (side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "device": str(device),
        "models": models,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
