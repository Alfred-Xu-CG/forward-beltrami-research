"""Where does the high32 component sit in a basis-free optical-flow hint?"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from qcopt.neural_bijection.dense import HierarchicalConvexQuadLocalLayer, dst2, local_photometric_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=55101)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prior = state["args"]
    if prior["method"] != "A4" or prior["side"] != 257 or prior["target_family"] != "high32":
        raise ValueError("requires the A4 image-only high32 257 checkpoint")
    encoder = ConvexQuadLocalImageEncoder(
        257, width=prior["a2_width"], head_mode=prior["a2_head_mode"], body_mode=prior["a2_body_mode"]
    )
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    decoder = HierarchicalConvexQuadLocalLayer(257, motion_mode="radial")
    fixed_all, moving_all, _, true_coefficients = make_dataset(
        args.count, 512, args.seed, return_coefficients=True, target_family="high32"
    )
    cases = []
    with torch.no_grad():
        for window in (1, 3, 5):
            for ridge in (0.1, 1.0, 10.0):
                mode_ranks = []
                mode_amplitudes = []
                captured_energy_16 = []
                captured_energy_64 = []
                for start in range(0, args.count, args.batch):
                    stop = min(start + args.batch, args.count)
                    fixed = fixed_all[start:stop]
                    moving = moving_all[start:stop]
                    root, levels, _ = encoder(torch.cat((fixed, moving), dim=1))
                    base = decoder.base(root, levels)
                    hint = local_photometric_logits(
                        fixed, moving, base, window=window, ridge=ridge, raw_span=decoder.local.raw_span
                    )
                    raw_displacement = (
                        decoder.local.raw_span / 256 * hint.tanh()
                    ).permute(0, 3, 1, 2)
                    coefficients = dst2(raw_displacement)
                    joint_energy = coefficients.square().sum(dim=1).reshape(stop - start, -1)
                    target_index = (64 - 1) * 255 + (64 - 1)
                    target_energy = joint_energy[:, target_index]
                    rank = (joint_energy > target_energy[:, None]).sum(dim=1) + 1
                    mode_ranks.extend(rank.tolist())
                    mode_amplitudes.extend(
                        (4 / 256**2 * coefficients[:, :, 63, 63]).tolist()
                    )
                    sorted_energy = joint_energy.sort(dim=1, descending=True).values
                    total = sorted_energy.sum(dim=1)
                    captured_energy_16.extend((sorted_energy[:, :16].sum(dim=1) / total).tolist())
                    captured_energy_64.extend((sorted_energy[:, :64].sum(dim=1) / total).tolist())
                cases.append({
                    "window": window,
                    "ridge": ridge,
                    "rank_high32_mode": mode_ranks,
                    "median_rank_high32_mode": statistics.median(mode_ranks),
                    "raw_high32_mode_amplitudes_xy": mode_amplitudes,
                    "mean_energy_fraction_top16": statistics.mean(captured_energy_16),
                    "mean_energy_fraction_top64": statistics.mean(captured_energy_64),
                })
    print(json.dumps({
        "question": "basis_free_photometric_hint_sine_mode_signal_to_noise",
        "checkpoint": args.checkpoint,
        "count": args.count,
        "seed": args.seed,
        "control_side": 257,
        "interior_side": 255,
        "high32_sine_index_one_based": [64, 64],
        "true_high32_amplitudes_evaluation_only": true_coefficients[:, 2].tolist(),
        "cases": cases,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
