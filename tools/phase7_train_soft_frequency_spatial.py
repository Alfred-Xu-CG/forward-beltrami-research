"""End-to-end trainable soft-frequency image-to-1025²-certified-P1 encoder."""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, certify_p1_or_identity,
)
from qcopt.neural_bijection.dense.patch_field import ResidualStaggeredPatchP1Layer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_fft_carrier_inference import infer_cycles
from phase7_test_soft_frequency_p1 import basis
from phase7_test_unknown_carrier_bank import dataset
from phase7_train_spatial_packet_images import (
    SpatialResidualEncoder, demod_amplitude,
)


class PatchDecoderAdapter(nn.Module):
    """Use one staggered four-pass patch cycle as the fine P1 decoder."""

    def __init__(self, side: int = 1025, *, checkpoint_patch: bool = False) -> None:
        super().__init__()
        if side != 1025:
            raise ValueError("current image encoder assumes a 1025-side fine grid")
        axis = torch.arange(side, dtype=torch.float64) / (side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "fine_identity", torch.stack((xx, yy), dim=-1)[None],
            persistent=False)
        self.patch = ResidualStaggeredPatchP1Layer(
            side, patch_cells=4, cycles=1, minimum_jacobian=.05,
            raw_span=.5)
        self.checkpoint_patch = checkpoint_patch

    def forward(self, coarse_latent: torch.Tensor,
                fine_latent: torch.Tensor) -> torch.Tensor:
        if coarse_latent.shape != (len(fine_latent), 255, 255, 2):
            raise ValueError("coarse latent has the wrong shape")
        base = self.fine_identity.expand(len(fine_latent), -1, -1, -1)
        latent = fine_latent.to(torch.float64)
        candidate = (
            checkpoint(self.patch, base, latent, use_reentrant=False)
            if self.checkpoint_patch else self.patch(base, latent))
        return certify_p1_or_identity(candidate, self.fine_identity)[0]


class SoftFrequencySpatialEncoder(nn.Module):
    def __init__(self, frequencies: range, device: torch.device,
                 temperature: float, width: int) -> None:
        super().__init__()
        self.low = frequencies.start
        self.high = frequencies.stop - 1
        self.temperature = temperature
        self.register_buffer(
            "image_basis", basis(512, frequencies, device), persistent=False)
        self.register_buffer(
            "fine_basis", basis(1025, frequencies, device), persistent=False)
        self.log_sharpness = nn.Parameter(torch.zeros((), device=device))
        self.spatial = SpatialResidualEncoder(width).to(device)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor,
                decoder: nn.Module):
        _, scores = infer_cycles(
            fixed, moving, self.low, self.high, radius=2)
        log_scores = torch.log(scores.clamp_min(1e-30))
        centered = log_scores - log_scores.amax(dim=1, keepdim=True)
        posterior = torch.softmax(
            centered * self.log_sharpness.exp() / self.temperature,
            dim=1)
        image_h = torch.einsum(
            "bk,khw->bhw", posterior, self.image_basis)
        fine_h = torch.einsum(
            "bk,khw->bhw", posterior, self.fine_basis)
        amplitude = demod_amplitude(
            fixed, moving, image_h[:, None], window=17, ridge=1.)
        features = F.interpolate(
            amplitude / 1e-4, size=(128, 128),
            mode="bilinear", align_corners=True)
        amplitude = self.spatial(features)
        amplitude = F.interpolate(
            amplitude, size=(1025, 1025),
            mode="bilinear", align_corners=True)
        displacement = 1e-4 * amplitude * fine_h[:, None]
        fine_latent = torch.atanh(
            (displacement / (2 / 1024)).clamp(-.95, .95)
            [:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()
        coarse_latent = torch.zeros(
            len(fixed), 255, 255, 2, device=fixed.device,
            dtype=torch.float32)
        return decoder(coarse_latent, fine_latent), posterior


def evaluate(model: SoftFrequencySpatialEncoder,
             data: tuple[torch.Tensor, ...],
             decoder: nn.Module,
             table: StructuredDenseQueryTable, batch_size: int):
    fixed, moving, target, support, _, true_frequency = data
    total_map = total_support = total_image = 0.
    support_count = frequency_correct = fallback = 0
    pmax = entropy = 0.
    minimum = math.inf
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fixed), batch_size):
            stop = min(start + batch_size, len(fixed))
            batch = stop - start
            output, posterior = model(
                fixed[start:stop], moving[start:stop], decoder)
            estimated = posterior.argmax(dim=1) + model.low
            frequency_correct += int(
                (estimated == true_frequency[start:stop]).sum())
            pmax += float(posterior.amax(dim=1).sum())
            entropy += float(-(posterior * torch.log(
                posterior.clamp_min(1e-30))).sum())
            query = table.interpolate(output.reshape(batch, -1, 2))
            warped = F.grid_sample(
                moving[start:stop], 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            square = (output - target[start:stop]).square().sum(dim=-1)
            total_map += float(square.mean(dim=(1, 2)).sum())
            mask = support[start:stop] > 0
            total_support += float(square[mask].sum())
            support_count += int(mask.sum())
            total_image += float((warped - fixed[start:stop]).square().mean(
                dim=(1, 2, 3)).sum())
            minimum = min(minimum, minimum_jacobian(output))
            fallback += int(torch.all(
                output == decoder.fine_identity,
                dim=(1, 2, 3)).sum())
    return {
        "map_vector_rmse": math.sqrt(total_map / len(fixed)),
        "support_vector_rmse": math.sqrt(total_support / support_count),
        "image_mse": total_image / len(fixed),
        "frequency_argmax_correct": frequency_correct,
        "frequency_total": len(fixed),
        "posterior_max_mean": pmax / len(fixed),
        "posterior_entropy_mean": entropy / len(fixed),
        "minimum_jacobian": minimum,
        "identity_outputs": fallback,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objective", choices=["map", "image"], default="map")
    parser.add_argument("--decoder", choices=["vertex", "patch"], default="vertex")
    parser.add_argument("--patch-checkpoint", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=31719)
    parser.add_argument("--test-seed", type=int, default=99113)
    parser.add_argument("--low", type=int, default=80)
    parser.add_argument("--high", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--ood-cycles", type=float, nargs="+", default=None)
    parser.add_argument("--ood-seed", type=int, default=59473)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(21071)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    frequencies = range(args.low, args.high + 1)
    train = dataset(
        args.train_count, args.train_seed, device, args.batch,
        table, tuple(frequencies))
    test = dataset(
        args.test_count, args.test_seed, device, args.batch,
        table, tuple(frequencies))
    if args.decoder == "vertex":
        decoder = CoarsePatchFineVertexP1Layer(
            257, 1025, coarse_patch_cells=16,
            compute_dtype=torch.float64, minimum_jacobian=.05,
            certify_output=True).to(device)
    else:
        decoder = PatchDecoderAdapter(
            checkpoint_patch=args.patch_checkpoint).to(device)
    model = SoftFrequencySpatialEncoder(
        frequencies, device, args.temperature, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline = evaluate(model, test, decoder, table, args.batch)
    gradient_nonzero_steps = 0
    gradient_nonfinite_steps = 0
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    for step in range(args.steps):
        model.train()
        ids = torch.randint(
            args.train_count, (args.batch,), device=device)
        fixed, moving, target = (
            train[i][ids] for i in (0, 1, 2))
        output, _ = model(fixed, moving, decoder)
        if args.objective == "map":
            loss = 1e8 * (output - target).square().sum(dim=-1).mean()
        else:
            query = table.interpolate(output.reshape(args.batch, -1, 2))
            warped = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            loss = 1e6 * (warped - fixed).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gain_grad = model.log_sharpness.grad
        gradient_nonzero_steps += int(gain_grad != 0)
        gradient_nonfinite_steps += int(not torch.isfinite(gain_grad))
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({
                "step": step + 1,
                "loss_last_100_mean": sum(losses[-100:]) / 100,
                "effective_temperature": args.temperature /
                                         float(model.log_sharpness.exp()),
            }), flush=True)
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated(device)
    final = evaluate(model, test, decoder, table, args.batch)
    if args.ood_cycles is not None:
        ood = dataset(
            args.test_count, args.ood_seed, device, args.batch,
            table, tuple(args.ood_cycles))
        ood_metrics = evaluate(model, ood, decoder, table, args.batch)
    else:
        ood_metrics = None
    print(json.dumps({
        "experiment": "phase7_differentiable_soft_frequency_training",
        "objective": args.objective,
        "decoder": args.decoder,
        "patch_checkpoint": args.patch_checkpoint,
        "control_vertices": 1025 ** 2,
        "control_faces": 2 * 1024 ** 2,
        "image_side": 512,
        "frequency_range": [args.low, args.high],
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "batch": args.batch,
        "steps": args.steps,
        "temperature_initial": args.temperature,
        "temperature_final": args.temperature /
                             float(model.log_sharpness.exp()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "lr": args.lr,
        "baseline": baseline,
        "final": final,
        "ood_cycles": args.ood_cycles,
        "ood_seed": args.ood_seed if args.ood_cycles is not None else None,
        "ood_metrics": ood_metrics,
        "gain_gradient_nonzero_steps": gradient_nonzero_steps,
        "gain_gradient_nonfinite_steps": gradient_nonfinite_steps,
        "mean_training_step_seconds": duration / args.steps,
        "peak_cuda_allocated_bytes": peak,
        "torch_version": torch.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
