"""Evaluate saved mixed-scale image encoders on fresh seeds and low widths."""
from __future__ import annotations

import argparse
import json

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_mixed_scale_image_pipeline import mixed_dataset
from phase7_train_mixed_scale_residual import (
    MixedScaleResidualEncoder, evaluate, preprocess_low,
)
from phase7_matched_low_patch import (
    matched_low_patch, multistart_refine_low_params,
)


def preprocess_width_bank(
    data: tuple[torch.Tensor, ...], batch_size: int,
    widths: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed, moving = data[:2]
    origins, vectors, chosen_widths = [], [], []
    with torch.no_grad():
        for start in range(0, len(fixed), batch_size):
            stop = min(start + batch_size, len(fixed))
            proposals = []
            for width in widths:
                origin, vector, _ = matched_low_patch(
                    fixed[start:stop], moving[start:stop],
                    width=width)
                origin, vector, loss = multistart_refine_low_params(
                    fixed[start:stop], moving[start:stop],
                    origin, vector, steps=4,
                    directions=4, width=width)
                proposals.append((origin, vector, loss))
            losses = torch.stack([p[2] for p in proposals], dim=1)
            best = losses.argmin(dim=1)
            origins.append(torch.stack(
                [p[0] for p in proposals], dim=1)[
                    torch.arange(stop - start, device=fixed.device), best])
            vectors.append(torch.stack(
                [p[1] for p in proposals], dim=1)[
                    torch.arange(stop - start, device=fixed.device), best])
            chosen_widths.append(torch.tensor(
                widths, device=fixed.device)[best])
    return (torch.cat(origins), torch.cat(vectors),
            torch.cat(chosen_widths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--map-checkpoint", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    model = MixedScaleResidualEncoder(device, table).to(device)
    state_zero = {k: v.clone() for k, v in model.state_dict().items()}

    settings = (
        ("integer_new_seed", 38729, tuple(range(80, 129)), 1 / 16),
        ("half_new_seed", 38731, (80.5, 96.5, 112.5, 127.5), 1 / 16),
        ("double_low_width", 47871, (80.5, 96.5, 112.5, 127.5), 1 / 8),
    )
    for name, seed, frequencies, low_width in settings:
        data = mixed_dataset(
            args.count, seed, device, args.batch,
            table, frequencies, low_width=low_width)
        truth_min_j = minimum_jacobian(data[2])
        data = (data[0], data[1], data[2], None, None,
                data[5], data[6], data[7])
        preprocessing = (("fixed_train_width", preprocess_low(
            data, args.batch)),)
        if low_width != 1 / 16:
            preprocessing += (
                ("known_true_width", preprocess_width_bank(
                    data, args.batch, (low_width,))),
                ("two_width_bank", preprocess_width_bank(
                    data, args.batch, (1 / 16, low_width))),
            )
        states = (
            ("zero_init", state_zero),
            ("map_train", torch.load(
                args.map_checkpoint, map_location=device,
                weights_only=False)["model"]),
            ("image_train", torch.load(
                args.image_checkpoint, map_location=device,
                weights_only=False)["model"]),
        )
        for proposal_name, low_params in preprocessing:
            width_accuracy = (float((
                low_params[2] == low_width).float().mean())
                if len(low_params) == 3 else None)
            for model_name, state in states:
                model.load_state_dict(state)
                result = evaluate(model, data, low_params, args.batch)
                print(json.dumps(dict(
                    condition=name, seed=seed, count=args.count,
                    low_width=low_width, model=model_name,
                    preprocessing=proposal_name,
                    width_accuracy=width_accuracy,
                    target_minimum_jacobian=truth_min_j,
                    **result), sort_keys=True), flush=True)
        del data, preprocessing


if __name__ == "__main__":
    main()
