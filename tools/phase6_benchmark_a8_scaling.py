"""Full image-layer forward/VJP and memory scaling for the QC-safe feedback."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_train_image_to_latent import _minimum_area_ratio
from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    SpectralSafeFeedbackLayer, structured_p1_face_beltrami_modulus,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint-refiner", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    began_setup = time.perf_counter()
    dataset = make_dataset(args.batch, args.image_side, 441007,
                           target_family="high32", return_coefficients=True)
    fixed, moving = dataset[0].to(device), dataset[1].to(device)
    encoder = ConvexQuadLocalImageEncoder(
        args.side, width=8, head_mode="multilevel", body_mode="local",
    ).to(device)
    decoder = SpectralSafeFeedbackLayer(
        args.side, initial_window=3, initial_ridge=1, initial_gain=1,
        initial_modes=16, extra_passes=2, extra_gain=1,
        extra_modes=16, extra_qc_cap=0.8, floor_fraction=0.8,
        extra_checkpoint=args.checkpoint_refiner,
    ).to(device)
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    setup_seconds = time.perf_counter() - began_setup

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def once():
        latent = encoder(torch.cat((fixed, moving), dim=1))
        control = decoder(fixed, moving, latent)
        query = table.interpolate(control.reshape(args.batch, -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), control

    # Warm-up before the peak reset includes first CUDA kernels and allocations.
    encoder.zero_grad(set_to_none=True)
    loss, _ = once()
    loss.backward()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times = [], []
    for _ in range(args.repeats):
        encoder.zero_grad(set_to_none=True)
        synchronize()
        start = time.perf_counter()
        loss, control = once()
        synchronize()
        middle = time.perf_counter()
        loss.backward()
        synchronize()
        end = time.perf_counter()
        forward_times.append(middle - start)
        backward_times.append(end - middle)
    allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    with torch.no_grad():
        maximum_mu = float(structured_p1_face_beltrami_modulus(control).amax())
        minimum_area = _minimum_area_ratio(control)
    print(json.dumps({
        "method": "A8_untrained_full_image_layer_scaling",
        "side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "repeats": args.repeats,
        "dtype": "float32",
        "device": str(device),
        "image_seed": 441007,
        "encoder": "untrained_zero_heads",
        "checkpoint_refiner": args.checkpoint_refiner,
        "setup_seconds": setup_seconds,
        "median_full_forward_seconds": statistics.median(forward_times),
        "median_full_vjp_seconds": statistics.median(backward_times),
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "image_mse_at_untrained_parameters": float(loss),
        "minimum_signed_area_ratio": minimum_area,
        "maximum_predicted_beltrami_modulus": maximum_mu,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
