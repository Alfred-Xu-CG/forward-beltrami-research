"""Train coarse and fine image-conditioned residual latents on mixed-scale P1 maps."""
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
    CoarsePatchFineVertexP1Layer, certify_p1_or_identity,
    exact_dyadic_p1_refine,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_matched_low_patch import (
    low_map_from_params, matched_low_patch,
    multistart_refine_low_params,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_fft_carrier_inference import carrier_field
from phase7_test_mixed_scale_image_pipeline import (
    mixed_dataset, photo_high,
)


def preprocess_low(
    data: tuple[torch.Tensor, ...], batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed, moving = data[:2]
    origins, vectors = [], []
    with torch.no_grad():
        for start in range(0, len(fixed), batch_size):
            stop = min(start + batch_size, len(fixed))
            origin, vector, _ = matched_low_patch(
                fixed[start:stop], moving[start:stop])
            origin, vector, _ = multistart_refine_low_params(
                fixed[start:stop], moving[start:stop],
                origin, vector, steps=4, directions=4)
            origins.append(origin)
            vectors.append(vector)
    return torch.cat(origins), torch.cat(vectors)


class MixedScaleResidualEncoder(nn.Module):
    def __init__(
        self, device: torch.device,
        table: StructuredDenseQueryTable,
        width: int = 16,
    ) -> None:
        super().__init__()
        self.table = table
        self.decoder = CoarsePatchFineVertexP1Layer(
            257, 1025, coarse_patch_cells=16,
            coarse_cycles=2, compute_dtype=torch.float64,
            minimum_jacobian=.05,
            certify_output=True).to(device)
        self.coarse_net = nn.Sequential(
            nn.Conv2d(5, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1),
        ).to(device)
        self.fine_net = nn.Sequential(
            nn.Conv2d(2, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1),
        ).to(device)
        for network in (self.coarse_net, self.fine_net):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
        self.register_buffer(
            "offsets", torch.arange(
                -.5, .525, .05, device=device),
            persistent=False)

    def forward(
        self, fixed: torch.Tensor,
        moving: torch.Tensor,
        origin: torch.Tensor,
        vector: torch.Tensor,
        low_width: torch.Tensor | None = None,
        proposed_low: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(fixed)
        proposed = (low_map_from_params(
            origin, vector, 257, torch.float64,
            width=1 / 16 if low_width is None else low_width)
            if proposed_low is None else proposed_low)
        if proposed.shape != (batch, 257, 257, 2):
            raise ValueError("proposed_low must have shape (batch,257,257,2)")
        coarse_base_latent = torch.atanh(((
            proposed - self.decoder.identity)[
                :, 1:-1, 1:-1] /
            (8 / 256)).clamp(-.95, .95)).float()
        fixed_coarse = F.interpolate(
            fixed, size=(257, 257),
            mode="bilinear", align_corners=True)
        moving_coarse = F.interpolate(
            moving, size=(257, 257),
            mode="bilinear", align_corners=True)
        low_displacement = (
            proposed - self.decoder.identity
        ).permute(0, 3, 1, 2).float()
        coarse_features = torch.cat((
            fixed_coarse, moving_coarse,
            fixed_coarse - moving_coarse,
            low_displacement / .006), dim=1)
        correction = self.coarse_net(coarse_features)
        coarse_latent = coarse_base_latent + .1 * correction[
            :, :, 1:-1, 1:-1].permute(
                0, 2, 3, 1).contiguous()
        coarse_map = self.decoder.coarse(
            self.decoder.identity.expand(batch, -1, -1, -1),
            coarse_latent.to(torch.float64))
        coarse_map, _ = certify_p1_or_identity(
            coarse_map, self.decoder.identity)
        fine_base = exact_dyadic_p1_refine(
            exact_dyadic_p1_refine(coarse_map))
        _, estimated_k, amplitude_image = photo_high(
            fixed, moving, fine_base,
            self.table, self.offsets,
            return_amplitude=True)
        fine_features = F.interpolate(
            amplitude_image / 1e-4,
            size=(128, 128), mode="bilinear",
            align_corners=True)
        residual_amplitude = self.fine_net(fine_features)
        amplitude = F.interpolate(
            amplitude_image, size=(1025, 1025),
            mode="bilinear", align_corners=True)
        amplitude = amplitude + 1e-4 * F.interpolate(
            residual_amplitude,
            size=(1025, 1025),
            mode="bilinear", align_corners=True)
        fine_displacement = amplitude * carrier_field(
            estimated_k, 1025, fixed.device)[:, None]
        fine_latent = torch.atanh((
            fine_displacement / (2 / 1024)
        ).clamp(-.95, .95)[:, :, 1:-1, 1:-1]).permute(
            0, 2, 3, 1).contiguous()
        floor = fine_base.new_full(
            (batch,), .05 / 1024 ** 2)
        candidate = self.decoder.fine(
            fine_base, fine_latent.to(torch.float64),
            area_floor=floor)
        output, _ = certify_p1_or_identity(
            candidate, self.decoder.fine_identity)
        return output, estimated_k


def evaluate(
    model: MixedScaleResidualEncoder,
    data: tuple[torch.Tensor, ...],
    low_params: tuple[torch.Tensor, ...],
    batch_size: int,
) -> dict[str, float | int]:
    fixed, moving, target, _, _, low_support, high_support, truth = data
    origin, vector = low_params[:2]
    low_width = low_params[2] if len(low_params) > 2 else None
    sums = dict(map=0., low=0., high=0.,
                low_n=0, high_n=0,
                image=0., freq=0.,
                minimum=math.inf, fallback=0)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fixed), batch_size):
            stop = min(start + batch_size, len(fixed))
            batch = stop - start
            output, estimated_k = model(
                fixed[start:stop], moving[start:stop],
                origin[start:stop], vector[start:stop],
                None if low_width is None else low_width[start:stop])
            square = (output - target[start:stop]).square().sum(dim=-1)
            sums["map"] += float(square.mean(dim=(1, 2)).sum())
            low_mask = low_support[start:stop] > 0
            high_mask = high_support[start:stop] > 0
            sums["low"] += float(square[low_mask].sum())
            sums["high"] += float(square[high_mask].sum())
            sums["low_n"] += int(low_mask.sum())
            sums["high_n"] += int(high_mask.sum())
            sums["freq"] += float(
                (estimated_k - truth[start:stop]).abs().sum())
            query = model.table.interpolate(
                output.reshape(batch, -1, 2))
            warped = F.grid_sample(
                moving[start:stop],
                2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            sums["image"] += float(
                (warped - fixed[start:stop]).square().mean(
                    dim=(1, 2, 3)).sum())
            sums["minimum"] = min(
                sums["minimum"], minimum_jacobian(output))
            sums["fallback"] += int(torch.all(
                output == model.decoder.fine_identity,
                dim=(1, 2, 3)).sum())
    return dict(
        map_vector_rmse=math.sqrt(sums["map"] / len(fixed)),
        low_support_vector_rmse=math.sqrt(
            sums["low"] / sums["low_n"]),
        high_support_vector_rmse=math.sqrt(
            sums["high"] / sums["high_n"]),
        image_mse=sums["image"] / len(fixed),
        frequency_mae=sums["freq"] / len(fixed),
        minimum_jacobian=sums["minimum"],
        identity_outputs=sums["fallback"],
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
    parser.add_argument("--ood-seed", type=int, default=21517)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(21071)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    train = mixed_dataset(
        args.train_count, args.train_seed, device,
        args.batch, table, tuple(range(80, 129)))
    test = mixed_dataset(
        args.test_count, args.test_seed, device,
        args.batch, table, tuple(range(80, 129)))
    ood = mixed_dataset(
        args.test_count, args.ood_seed, device,
        args.batch, table,
        (80.5, 96.5, 112.5, 127.5))
    # Training only needs the images and map. Evaluation needs the two masks,
    # not the dense oracle low maps. Release those large arrays before VJP.
    train = train[:3]
    test = (test[0], test[1], test[2], None, None,
            test[5], test[6], test[7])
    ood = (ood[0], ood[1], ood[2], None, None,
           ood[5], ood[6], ood[7])
    tick = time.perf_counter()
    train_params = preprocess_low(train, args.batch)
    test_params = preprocess_low(test, args.batch)
    ood_params = preprocess_low(ood, args.batch)
    preprocessing_seconds = time.perf_counter() - tick
    model = MixedScaleResidualEncoder(
        device, table, args.width).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr)
    baseline = evaluate(
        model, test, test_params, args.batch)
    baseline_ood = evaluate(
        model, ood, ood_params, args.batch)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    coarse_gradient_nonzero_steps = 0
    fine_gradient_nonzero_steps = 0
    gradient_nonfinite_steps = 0
    losses = []
    for step in range(args.steps):
        model.train()
        ids = torch.randint(
            args.train_count, (args.batch,), device=device)
        fixed, moving, target = (
            train[i][ids] for i in (0, 1, 2))
        origin, vector = (
            parameter[ids] for parameter in train_params)
        output, _ = model(
            fixed, moving, origin, vector)
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
        coarse_grad = model.coarse_net[-1].weight.grad
        fine_grad = model.fine_net[-1].weight.grad
        coarse_gradient_nonzero_steps += int(
            (coarse_grad != 0).any())
        fine_gradient_nonzero_steps += int(
            (fine_grad != 0).any())
        gradient_nonfinite_steps += int(
            not torch.isfinite(coarse_grad).all() or
            not torch.isfinite(fine_grad).all())
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps(dict(
                step=step + 1,
                loss_last_100_mean=sum(
                    losses[-100:]) / 100,
            )), flush=True)
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated(device)
    final = evaluate(model, test, test_params, args.batch)
    final_ood = evaluate(model, ood, ood_params, args.batch)
    if args.save:
        torch.save(dict(
            model=model.state_dict(),
            config=vars(args)), args.save)
    print(json.dumps(dict(
        experiment="phase7_mixed_scale_residual_training",
        objective=args.objective,
        train_count=args.train_count,
        test_count=args.test_count,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        ood_seed=args.ood_seed,
        batch=args.batch,
        steps=args.steps,
        control_vertices=1025 ** 2,
        control_faces=2 * 1024 ** 2,
        image_side=512,
        parameters=sum(
            p.numel() for p in model.parameters()),
        baseline=baseline,
        baseline_ood=baseline_ood,
        final=final,
        final_ood=final_ood,
        coarse_gradient_nonzero_steps=coarse_gradient_nonzero_steps,
        fine_gradient_nonzero_steps=fine_gradient_nonzero_steps,
        gradient_nonfinite_steps=gradient_nonfinite_steps,
        low_preprocessing_seconds=preprocessing_seconds,
        mean_training_step_seconds=duration / args.steps,
        peak_cuda_allocated_bytes=peak,
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
