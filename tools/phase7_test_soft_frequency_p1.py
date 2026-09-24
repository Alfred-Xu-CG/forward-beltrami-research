"""Test a fully differentiable FFT frequency posterior feeding the safe P1 layer.

No top-k or argmax is used in the soft path. The carrier family remains a
known finite integer-frequency dictionary.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import CoarsePatchFineVertexP1Layer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_fft_carrier_inference import infer_cycles
from phase7_test_unknown_carrier_bank import dataset
from phase7_train_spatial_packet_images import demod_amplitude


def basis(side: int, frequencies: range,
          device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack(
        [torch.sin(2 * math.pi * k * xx) *
         torch.sin(2 * math.pi * k * yy) for k in frequencies], dim=0)


def decode_posterior(
    probabilities: torch.Tensor, fixed: torch.Tensor,
    moving: torch.Tensor, decoder: CoarsePatchFineVertexP1Layer,
    basis_image: torch.Tensor, basis_fine: torch.Tensor,
) -> torch.Tensor:
    h_image = torch.einsum("bk,khw->bhw", probabilities, basis_image)
    h_fine = torch.einsum("bk,khw->bhw", probabilities, basis_fine)
    amplitude = demod_amplitude(
        fixed, moving, h_image[:, None], window=17, ridge=1.)
    amplitude = F.interpolate(
        amplitude, size=(1025, 1025), mode="bilinear",
        align_corners=True)
    displacement = amplitude * h_fine[:, None]
    latent = torch.atanh(
        (displacement / (2 / 1024)).clamp(-.95, .95)
        [:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()
    coarse = torch.zeros(len(fixed), 255, 255, 2,
                         device=fixed.device, dtype=torch.float32)
    return decoder(coarse, latent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=99023)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--low", type=int, default=80)
    parser.add_argument("--high", type=int, default=128)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[.2, .1, .05, .02, .01])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    frequencies = range(args.low, args.high + 1)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    fixed_all, moving_all, target_all, support_all, _, truth_all = dataset(
        args.count, args.seed, device, args.batch, table,
        tuple(frequencies))
    image_basis = basis(512, frequencies, device)
    fine_basis = basis(1025, frequencies, device)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True).to(device)
    names = ["hard"] + [f"soft_{t}" for t in args.temperatures]
    accum = {
        name: {"map": 0., "support": 0., "image": 0., "support_n": 0,
               "minimum": math.inf, "fallback": 0,
               "pmax": 0., "entropy": 0., "time": []}
        for name in names
    }
    grad_test = {}
    for start in range(0, args.count, args.batch):
        fixed, moving, target, support, truth = (
            tensor[start:start + args.batch] for tensor in
            (fixed_all, moving_all, target_all, support_all, truth_all))
        batch = len(fixed)
        _, scores = infer_cycles(
            fixed, moving, args.low, args.high, radius=2)
        log_scores = torch.log(scores.clamp_min(1e-30))
        for name, row in accum.items():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            if name == "hard":
                probabilities = F.one_hot(
                    scores.argmax(dim=1), num_classes=len(frequencies)
                ).float()
            else:
                temperature = float(name.split("_")[1])
                probabilities = torch.softmax(
                    log_scores / temperature, dim=1)
            with torch.no_grad():
                output = decode_posterior(
                    probabilities, fixed, moving, decoder,
                    image_basis, fine_basis)
                query = table.interpolate(output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1,
                    mode="bilinear", padding_mode="border",
                    align_corners=True)
                square = (output - target).square().sum(dim=-1)
                row["map"] += float(square.mean(dim=(1, 2)).sum())
                mask = support > 0
                row["support"] += float(square[mask].sum())
                row["support_n"] += int(mask.sum())
                row["image"] += float((warped - fixed).square().mean(
                    dim=(1, 2, 3)).sum())
                row["minimum"] = min(
                    row["minimum"], minimum_jacobian(output))
                row["fallback"] += int(torch.all(
                    output == decoder.fine_identity,
                    dim=(1, 2, 3)).sum())
                row["pmax"] += float(probabilities.amax(dim=1).sum())
                entropy = -(probabilities * torch.log(
                    probabilities.clamp_min(1e-30))).sum(dim=1)
                row["entropy"] += float(entropy.sum())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row["time"].append(time.perf_counter() - tick)
        if start == 0:
            for temperature in args.temperatures:
                prior_logits = torch.zeros(
                    len(frequencies), device=device,
                    dtype=torch.float32, requires_grad=True)
                p = torch.softmax(
                    (log_scores.detach() + prior_logits[None]) /
                    temperature, dim=1)
                output = decode_posterior(
                    p, fixed, moving, decoder,
                    image_basis, fine_basis)
                loss = 1e8 * (output - target).square().sum(dim=-1).mean()
                grad, = torch.autograd.grad(loss, prior_logits)
                grad_test[str(temperature)] = {
                    "finite": bool(torch.isfinite(grad).all()),
                    "maxabs": float(grad.abs().max()),
                    "nonzero_entries": int((grad != 0).sum()),
                }
    rows = []
    for name, raw in accum.items():
        rows.append({
            "method": name,
            "map_vector_rmse": math.sqrt(raw["map"] / args.count),
            "support_vector_rmse": math.sqrt(
                raw["support"] / raw["support_n"]),
            "image_mse": raw["image"] / args.count,
            "minimum_jacobian": raw["minimum"],
            "identity_outputs": raw["fallback"],
            "mean_posterior_max": raw["pmax"] / args.count,
            "mean_posterior_entropy": raw["entropy"] / args.count,
            "median_eval_batch_seconds": sorted(raw["time"])[
                len(raw["time"]) // 2],
        })
    print(json.dumps({
        "experiment": "phase7_soft_frequency_p1",
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": 512,
        "count": args.count,
        "seed": args.seed,
        "batch": args.batch,
        "frequency_range": [args.low, args.high],
        "temperatures": args.temperatures,
        "gradient_to_frequency_prior": grad_test,
        "rows": rows,
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
