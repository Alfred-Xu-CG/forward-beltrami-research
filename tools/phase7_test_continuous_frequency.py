"""Compare differentiable continuous-frequency readouts on off-dictionary packets."""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import CoarsePatchFineVertexP1Layer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_fft_carrier_inference import carrier_field, infer_cycles
from phase7_test_soft_frequency_p1 import basis
from phase7_test_unknown_carrier_bank import dataset
from phase7_train_spatial_packet_images import demod_amplitude


def decode(h_image: torch.Tensor, h_fine: torch.Tensor,
           fixed: torch.Tensor, moving: torch.Tensor,
           decoder: CoarsePatchFineVertexP1Layer) -> torch.Tensor:
    amplitude = demod_amplitude(
        fixed, moving, h_image[:, None], window=17, ridge=1.)
    amplitude = F.interpolate(
        amplitude, size=(1025, 1025), mode="bilinear", align_corners=True)
    displacement = amplitude * h_fine[:, None]
    latent = torch.atanh(
        (displacement / (2 / 1024)).clamp(-.95, .95)
        [:, :, 1:-1, 1:-1]).permute(0, 2, 3, 1).contiguous()
    coarse = torch.zeros(len(fixed), 255, 255, 2, device=fixed.device)
    return decoder(coarse, latent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=59473)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-cycles", type=float, nargs="+",
                        default=[80.5, 96.5, 112.5, 127.5])
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[.2, .1, .05, .02])
    args = parser.parse_args()
    device = torch.device(args.device)
    low, high = 80, 128
    frequencies = range(low, high + 1)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    fixed_all, moving_all, target_all, support_all, _, truth_all = dataset(
        args.count, args.seed, device, args.batch, table,
        tuple(args.target_cycles))
    image_basis = basis(512, frequencies, device)
    fine_basis = basis(1025, frequencies, device)
    decoder = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True).to(device)
    names = (["identity", "oracle", "hard", "peak_parabola"] +
             [f"mean_{t}" for t in args.temperatures] +
             [f"mix_{t}" for t in args.temperatures])
    sums = {
        name: dict(map=0., support=0., support_n=0, image=0.,
                   freq_abs=0., freq_sq=0., minimum=math.inf, fallback=0)
        for name in names
    }
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            stop = min(start + args.batch, args.count)
            fixed, moving, target, support, truth = (
                item[start:stop] for item in
                (fixed_all, moving_all, target_all, support_all, truth_all))
            batch = len(fixed)
            _, scores = infer_cycles(fixed, moving, low, high, radius=2)
            logs = torch.log(scores.clamp_min(1e-30))
            peak_id = scores.argmax(dim=1)
            left_id = (peak_id - 1).clamp_min(0)
            right_id = (peak_id + 1).clamp_max(high - low)
            left = logs.gather(1, left_id[:, None]).squeeze(1)
            center = logs.gather(1, peak_id[:, None]).squeeze(1)
            right = logs.gather(1, right_id[:, None]).squeeze(1)
            denominator = left - 2 * center + right
            offset = torch.where(
                denominator.abs() > 1e-10,
                .5 * (left - right) / denominator,
                torch.zeros_like(center)).clamp(-.5, .5)
            parabolic = (peak_id + low).float() + offset
            p_by_t = {
                t: torch.softmax(logs / t, dim=1)
                for t in args.temperatures
            }
            for name in names:
                row = sums[name]
                if name == "identity":
                    output = decoder.fine_identity.expand(batch, -1, -1, -1)
                    estimated = None
                else:
                    if name == "oracle":
                        estimated = truth.float()
                    elif name == "hard":
                        estimated = (peak_id + low).float()
                    elif name == "peak_parabola":
                        estimated = parabolic
                    elif name.startswith("mean_"):
                        t = float(name.split("_")[1])
                        weights = p_by_t[t]
                        k_axis = torch.arange(
                            low, high + 1, device=device).float()
                        estimated = (weights * k_axis[None]).sum(dim=1)
                    else:
                        t = float(name.split("_")[1])
                        weights = p_by_t[t]
                        estimated = (weights * torch.arange(
                            low, high + 1, device=device)[None]).sum(dim=1)
                    if name.startswith("mix_"):
                        h_image = torch.einsum(
                            "bk,khw->bhw", weights, image_basis)
                        h_fine = torch.einsum(
                            "bk,khw->bhw", weights, fine_basis)
                    else:
                        h_image = carrier_field(estimated, 512, device)
                        h_fine = carrier_field(estimated, 1025, device)
                    output = decode(
                        h_image, h_fine, fixed, moving, decoder)
                    frequency_error = (estimated - truth.float()).abs()
                    row["freq_abs"] += float(frequency_error.sum())
                    row["freq_sq"] += float(frequency_error.square().sum())
                square = (output - target).square().sum(dim=-1)
                row["map"] += float(square.mean(dim=(1, 2)).sum())
                mask = support > 0
                row["support"] += float(square[mask].sum())
                row["support_n"] += int(mask.sum())
                query = table.interpolate(output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1, mode="bilinear",
                    padding_mode="border", align_corners=True)
                row["image"] += float(
                    (warped - fixed).square().mean(dim=(1, 2, 3)).sum())
                row["minimum"] = min(
                    row["minimum"], minimum_jacobian(output))
                row["fallback"] += int(torch.all(
                    output == decoder.fine_identity,
                    dim=(1, 2, 3)).sum())
    rows = []
    for name, row in sums.items():
        rows.append(dict(
            method=name,
            map_vector_rmse=math.sqrt(row["map"] / args.count),
            support_vector_rmse=math.sqrt(
                row["support"] / row["support_n"]),
            image_mse=row["image"] / args.count,
            frequency_mae=(
                row["freq_abs"] / args.count if name != "identity" else None),
            frequency_rmse=(
                math.sqrt(row["freq_sq"] / args.count)
                if name != "identity" else None),
            minimum_jacobian=row["minimum"],
            identity_outputs=row["fallback"]))
    print(json.dumps(dict(
        experiment="phase7_continuous_frequency_readout",
        control_vertices=1025 ** 2,
        control_faces=2 * 1024 ** 2,
        image_side=512,
        count=args.count,
        batch=args.batch,
        seed=args.seed,
        target_cycles=args.target_cycles,
        temperatures=args.temperatures,
        rows=rows,
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
