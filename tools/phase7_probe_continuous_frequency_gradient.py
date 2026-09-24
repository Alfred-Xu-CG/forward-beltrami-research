"""Full VJP and finite-difference probe for soft continuous carrier refinement."""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_refine_continuous_frequency import local_amplitude_and_score
from phase7_test_continuous_frequency import decode
from phase7_test_fft_carrier_inference import carrier_field, infer_cycles
from phase7_test_unknown_carrier_bank import dataset


def infer_map(fixed: torch.Tensor, moving: torch.Tensor,
              log_gain: torch.Tensor,
              decoder: CoarsePatchFineVertexP1Layer,
              offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, fft_scores = infer_cycles(fixed, moving, 80, 128, radius=2)
    p = torch.softmax(
        torch.log(fft_scores.clamp_min(1e-30)) / .1, dim=1)
    center = (
        p * torch.arange(80, 129, device=fixed.device)[None]).sum(dim=1)
    gradient = physical_image_gradient(moving)
    scores = []
    for offset in offsets:
        carrier = carrier_field(center + offset, 512, fixed.device)
        _, score = local_amplitude_and_score(
            fixed, moving, gradient, carrier)
        scores.append(score)
    scores = torch.stack(scores, dim=1)
    normalized = (
        scores - scores.amax(dim=1, keepdim=True)) / (
            scores.std(dim=1, keepdim=True) + 1e-30)
    weights = torch.softmax(
        normalized * log_gain.exp() / .05, dim=1)
    estimated = center + (weights * offsets[None]).sum(dim=1)
    h_image = carrier_field(estimated, 512, fixed.device)
    h_fine = carrier_field(estimated, 1025, fixed.device)
    return decode(h_image, h_fine, fixed, moving, decoder), estimated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=59473)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    fixed, moving, target, _, _, truth = dataset(
        args.batch, args.seed, device, args.batch, table,
        (80.5, 96.5, 112.5, 127.5))
    fixed = fixed.detach().requires_grad_()
    gain = torch.zeros((), device=device, requires_grad=True)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16,
        compute_dtype=torch.float64, minimum_jacobian=.05,
        certify_output=True).to(device)
    offsets = torch.arange(
        -.5, .525, .05, device=device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    tick = time.perf_counter()
    output, estimated = infer_map(
        fixed, moving, gain, decoder, offsets)
    loss = 1e8 * (output - target).square().sum(dim=-1).mean()
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    grad_image, grad_gain = torch.autograd.grad(
        loss, (fixed, gain))
    torch.cuda.synchronize(device)
    vjp_seconds = time.perf_counter() - tick
    if not bool(torch.isfinite(grad_image).all()):
        raise AssertionError("nonfinite gradient to fixed image")
    if not bool(torch.isfinite(grad_gain)):
        raise AssertionError("nonfinite gradient to gain")
    finite_differences = {}
    with torch.no_grad():
        for epsilon in (1e-3, 1e-2, 5e-2, 1e-1):
            plus, _ = infer_map(
                fixed, moving, gain + epsilon, decoder, offsets)
            minus, _ = infer_map(
                fixed, moving, gain - epsilon, decoder, offsets)
            lp = 1e8 * (plus - target).square().sum(dim=-1).mean()
            lm = 1e8 * (minus - target).square().sum(dim=-1).mean()
            finite_differences[str(epsilon)] = float(
                (lp - lm) / (2 * epsilon))
    print(json.dumps(dict(
        experiment="phase7_soft_continuous_frequency_full_vjp",
        control_vertices=1025 ** 2,
        batch=args.batch,
        seed=args.seed,
        true_frequency=truth.cpu().tolist(),
        estimated_frequency=estimated.detach().cpu().tolist(),
        map_vector_rmse=math.sqrt(
            float((output - target).square().sum(dim=-1).mean())),
        image_gradient_nonzero=int((grad_image != 0).sum()),
        image_gradient_finite=bool(torch.isfinite(grad_image).all()),
        gain_gradient=float(grad_gain),
        gain_gradient_finite_differences=finite_differences,
        forward_seconds=forward_seconds,
        vjp_seconds=vjp_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
