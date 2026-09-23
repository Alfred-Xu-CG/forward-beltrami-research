"""Separate old-frequency gain drift from new frequency-64 gain amplification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base18", type=Path, required=True)
    parser.add_argument("--trained21", type=Path, required=True)
    parser.add_argument("--base-old-trained64", type=Path, required=True)
    parser.add_argument("--trained-old-neutral64", type=Path, required=True)
    args = parser.parse_args()

    base = torch.load(args.base18, map_location="cpu", weights_only=False)
    trained = torch.load(args.trained21, map_location="cpu", weights_only=False)
    base_gains = base["raw_gains"].reshape(3, 6)
    trained_gains = trained["raw_gains"].reshape(3, 7)
    if trained.get("args", {}).get("frequencies") != "1,2,4,8,16,32,64":
        raise ValueError("trained checkpoint must use ordered frequencies 1..64")

    base_old_trained64 = trained_gains.clone()
    base_old_trained64[:, :6] = base_gains
    trained_old_neutral64 = trained_gains.clone()
    trained_old_neutral64[:, 6] = 0.0  # exp(3*tanh(0)) = 1

    for path, gains, label in (
        (args.base_old_trained64, base_old_trained64, "base_old_trained64"),
        (args.trained_old_neutral64, trained_old_neutral64,
         "trained_old_neutral64"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"raw_gains": gains.reshape(-1),
                    "args": dict(trained["args"]),
                    "gain_ablation": label,
                    "source_base18": str(args.base18),
                    "source_trained21": str(args.trained21)}, path)


if __name__ == "__main__":
    main()
