"""Train a soft continuous-frequency image encoder through a 1025² safe P1 layer."""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_refine_continuous_frequency import local_amplitude_and_score
from phase7_test_fft_carrier_inference import (
    carrier_field, infer_cycles,
)
from phase7_test_unknown_carrier_bank import dataset
from phase7_train_spatial_packet_images import (
    SpatialResidualEncoder, demod_amplitude,
)


class ContinuousFrequencySpatialEncoder(nn.Module):
    """FFT coarse mean, differentiable local photometric soft refinement, CNN."""

    def __init__(self, width: int, device: torch.device,
                 photo_temperature: float = .05) -> None:
        super().__init__()
        self.photo_temperature = photo_temperature
        self.log_photo_sharpness = nn.Parameter(
            torch.zeros((), device=device))
        self.spatial = SpatialResidualEncoder(width).to(device)
        self.register_buffer(
            "offsets", torch.arange(
                -.5, .525, .05, device=device, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "integer_axis", torch.arange(
                80, 129, device=device, dtype=torch.float32),
            persistent=False)

    def forward(
        self, fixed: torch.Tensor, moving: torch.Tensor,
        decoder: CoarsePatchFineVertexP1Layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, fft_scores = infer_cycles(
            fixed, moving, 80, 128, radius=2)
        coarse_p = torch.softmax(
            torch.log(fft_scores.clamp_min(1e-30)) / .1, dim=1)
        coarse_k = (
            coarse_p * self.integer_axis[None]).sum(dim=1)
        image_gradient = physical_image_gradient(moving)
        photo_scores = []
        for offset in self.offsets:
            carrier = carrier_field(
                coarse_k + offset, 512, fixed.device)
            _, score = local_amplitude_and_score(
                fixed, moving, image_gradient, carrier)
            photo_scores.append(score)
        photo_scores = torch.stack(photo_scores, dim=1)
        normalized = (
            photo_scores - photo_scores.amax(
                dim=1, keepdim=True)) / (
                    photo_scores.std(
                        dim=1, keepdim=True) + 1e-30)
        local_p = torch.softmax(
            normalized * self.log_photo_sharpness.exp() /
            self.photo_temperature, dim=1)
        estimated_k = coarse_k + (
            local_p * self.offsets[None]).sum(dim=1)
        image_h = carrier_field(
            estimated_k, 512, fixed.device)
        fine_h = carrier_field(
            estimated_k, 1025, fixed.device)
        amplitude = demod_amplitude(
            fixed, moving, image_h[:, None],
            window=17, ridge=1.)
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
            [:, :, 1:-1, 1:-1]).permute(
                0, 2, 3, 1).contiguous()
        coarse_latent = torch.zeros(
            len(fixed), 255, 255, 2,
            device=fixed.device, dtype=torch.float32)
        return decoder(coarse_latent, fine_latent), estimated_k


def evaluate(
    model: ContinuousFrequencySpatialEncoder,
    data: tuple[torch.Tensor, ...],
    decoder: CoarsePatchFineVertexP1Layer,
    table: StructuredDenseQueryTable,
    batch_size: int,
) -> dict[str, float | int]:
    fixed, moving, target, support, _, truth = data
    total_map = total_support = total_image = 0.
    support_count = fallback = 0
    frequency_abs = frequency_sq = 0.
    frequency_within_005 = 0
    minimum = math.inf
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fixed), batch_size):
            stop = min(start + batch_size, len(fixed))
            batch = stop - start
            output, estimated = model(
                fixed[start:stop],
                moving[start:stop], decoder)
            error = (estimated - truth[start:stop]).abs()
            frequency_abs += float(error.sum())
            frequency_sq += float(error.square().sum())
            frequency_within_005 += int((error <= .05).sum())
            square = (output - target[start:stop]).square().sum(dim=-1)
            total_map += float(square.mean(dim=(1, 2)).sum())
            mask = support[start:stop] > 0
            total_support += float(square[mask].sum())
            support_count += int(mask.sum())
            query = table.interpolate(output.reshape(batch, -1, 2))
            warped = F.grid_sample(
                moving[start:stop],
                2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            total_image += float(
                (warped - fixed[start:stop]).square().mean(
                    dim=(1, 2, 3)).sum())
            minimum = min(
                minimum, minimum_jacobian(output))
            fallback += int(torch.all(
                output == decoder.fine_identity,
                dim=(1, 2, 3)).sum())
    return dict(
        map_vector_rmse=math.sqrt(total_map / len(fixed)),
        support_vector_rmse=math.sqrt(
            total_support / support_count),
        image_mse=total_image / len(fixed),
        frequency_mae=frequency_abs / len(fixed),
        frequency_rmse=math.sqrt(frequency_sq / len(fixed)),
        frequency_within_005=frequency_within_005,
        frequency_total=len(fixed),
        minimum_jacobian=minimum,
        identity_outputs=fallback,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objective", choices=["map", "image"], default="map")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=31719)
    parser.add_argument("--test-seed", type=int, default=99113)
    parser.add_argument("--ood-seed", type=int, default=59473)
    parser.add_argument(
        "--ood-cycles", type=float, nargs="+",
        default=[80.5, 96.5, 112.5, 127.5])
    parser.add_argument("--photo-temperature", type=float, default=.05)
    parser.add_argument("--extra-ood-count", type=int, default=0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(21071)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    frequencies = tuple(range(80, 129))
    train = dataset(
        args.train_count, args.train_seed, device,
        args.batch, table, frequencies)
    test = dataset(
        args.test_count, args.test_seed, device,
        args.batch, table, frequencies)
    ood = dataset(
        args.test_count, args.ood_seed, device,
        args.batch, table, tuple(args.ood_cycles))
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16,
        compute_dtype=torch.float64,
        minimum_jacobian=.05,
        certify_output=True).to(device)
    model = ContinuousFrequencySpatialEncoder(
        args.width, device, args.photo_temperature).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    baseline = evaluate(
        model, test, decoder, table, args.batch)
    baseline_ood = evaluate(
        model, ood, decoder, table, args.batch)
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
            loss = 1e8 * (
                output - target).square().sum(dim=-1).mean()
        else:
            query = table.interpolate(
                output.reshape(args.batch, -1, 2))
            warped = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            loss = 1e6 * (warped - fixed).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gain_grad = model.log_photo_sharpness.grad
        gradient_nonzero_steps += int(gain_grad != 0)
        gradient_nonfinite_steps += int(
            not torch.isfinite(gain_grad))
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps(dict(
                step=step + 1,
                loss_last_100_mean=sum(
                    losses[-100:]) / 100,
                effective_photo_temperature=(
                    args.photo_temperature /
                    float(model.log_photo_sharpness.exp())),
            )), flush=True)
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated(device)
    final = evaluate(
        model, test, decoder, table, args.batch)
    final_ood = evaluate(
        model, ood, decoder, table, args.batch)
    noisy_ood = {}
    noise_generator = torch.Generator(device=device).manual_seed(77999)
    for sigma in (.001, .003, .01):
        noisy_fixed = ood[0] + sigma * torch.randn(
            ood[0].shape, device=device, dtype=ood[0].dtype,
            generator=noise_generator)
        noisy_ood[str(sigma)] = evaluate(
            model, (noisy_fixed, *ood[1:]), decoder,
            table, args.batch)
    extra_ood = {}
    if args.extra_ood_count > 0:
        families = (
            ("all_halves", 77531,
             tuple(k + .5 for k in range(80, 128))),
            ("quarter_grid", 77532,
             tuple(k + fraction for k in range(80, 128)
                   for fraction in (.25, .75))),
        )
        for label, seed, cycles in families:
            extra_data = dataset(
                args.extra_ood_count, seed, device,
                args.batch, table, cycles)
            extra_ood[label] = evaluate(
                model, extra_data, decoder, table,
                args.batch)
            del extra_data
    print(json.dumps(dict(
        experiment="phase7_continuous_frequency_spatial_training",
        objective=args.objective,
        control_vertices=1025 ** 2,
        control_faces=2 * 1024 ** 2,
        image_side=512,
        train_count=args.train_count,
        test_count=args.test_count,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        ood_seed=args.ood_seed,
        ood_cycles=args.ood_cycles,
        batch=args.batch,
        steps=args.steps,
        photo_temperature_initial=args.photo_temperature,
        photo_temperature_final=(
            args.photo_temperature /
            float(model.log_photo_sharpness.exp())),
        parameters=sum(
            p.numel() for p in model.parameters()),
        lr=args.lr,
        baseline=baseline,
        baseline_ood=baseline_ood,
        final=final,
        final_ood=final_ood,
        noisy_ood=noisy_ood,
        extra_ood_count=args.extra_ood_count,
        extra_ood=extra_ood,
        gain_gradient_nonzero_steps=gradient_nonzero_steps,
        gain_gradient_nonfinite_steps=gradient_nonfinite_steps,
        mean_training_step_seconds=duration / args.steps,
        peak_cuda_allocated_bytes=peak,
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
