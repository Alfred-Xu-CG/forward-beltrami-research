"""Compare 32-cycle edge-weight spectra of direct oracle and image CNNs."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from phase6_train_conductance_image_to_latent import MultiscaleEdgeImageEncoder
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle


def spectral_summary(values: torch.Tensor) -> dict[str, float]:
    cropped = values[:256, :256].float()
    centered = cropped - cropped.mean()
    transform = torch.fft.fft2(centered)
    power = transform.abs().square()
    bins = (32, 224)
    high_power = sum(float(power[i, j]) for i in bins for j in bins)
    total_power = float(power.sum())
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
        "cycle32_diagonal_bin_power_fraction": high_power / total_power if total_power else 0.0,
        "cycle32_diagonal_bin_root_mean_amplitude": high_power**0.5 / (256**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=int, default=29)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--pixel", required=True)
    parser.add_argument("--gradient", required=True)
    parser.add_argument("--teacher", required=True)
    args = parser.parse_args()
    side = 257
    mesh = structured_rectangle(side - 1, side - 1)
    grid = np.arange(side**2, dtype=np.int64).reshape(side, side)
    edges = np.concatenate((
        np.stack((grid[:, :-1].ravel(), grid[:, 1:].ravel()), axis=1),
        np.stack((grid[:-1].ravel(), grid[1:].ravel()), axis=1),
        np.stack((grid[:-1, :-1].ravel(), grid[1:, 1:].ravel()), axis=1),
    ))
    dataset = make_dataset(32, 512, 55101, target_family="high32", return_coefficients=True)
    fixed, moving = dataset[0][args.item:args.item + 1], dataset[1][args.item:args.item + 1]
    pair = torch.cat((fixed, moving), dim=1)
    count_h, count_v, count_d = side * (side - 1), side * (side - 1), (side - 1)**2
    summaries = {}
    for name, path in (
        ("pixel", args.pixel), ("gradient", args.gradient), ("teacher", args.teacher)
    ):
        state = torch.load(path, map_location="cpu", weights_only=False)
        encoder = MultiscaleEdgeImageEncoder(
            side, mesh.vertices[edges].mean(axis=1),
            body_mode=state["args"]["encoder_body"],
        )
        encoder.load_state_dict(state["encoder"])
        with torch.no_grad():
            logits = encoder(pair)[0]
        fields = torch.split(logits, (count_h, count_v, count_d))
        fields = (fields[0].reshape(side, side - 1),
                  fields[1].reshape(side - 1, side),
                  fields[2].reshape(side - 1, side - 1))
        summaries[name] = {
            kind: {
                "logit": spectral_summary(field),
                "conductance": spectral_summary(1 + 15 * field.sigmoid()),
            }
            for kind, field in zip(("horizontal", "vertical", "diagonal"), fields)
        }
    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    summaries["direct_target_coordinate_oracle"] = {
        kind: {
            "logit": spectral_summary(field[0]),
            "conductance": spectral_summary(1 + 15 * field[0].sigmoid()),
        }
        for kind, field in zip(("horizontal", "vertical", "diagonal"), oracle["logits"])
    }
    print(json.dumps({
        "method": "edge_logit_and_conductance_cycle32_spectral_diagnostic",
        "control_side": side,
        "train_seed": 55101,
        "train_count": 32,
        "train_item": args.item,
        "frequency_bins": [[32, 32], [32, 224], [224, 32], [224, 224]],
        "crop": "first_256_by_256_values_of_each_edge_field",
        "summaries": summaries,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
