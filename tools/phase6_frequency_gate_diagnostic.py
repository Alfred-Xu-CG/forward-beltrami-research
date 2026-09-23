"""Can image-derived conductance coefficients indicate need for frequency 64?"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from phase6_eval_photographic_content import _dataset as photo_dataset
from phase6_train_multisample_image import make_dataset
from qcopt.neural_bijection.dense import PhotometricSpectralTutteLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--fit-side", type=int, default=256)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1170031)
    parser.add_argument("--photo-variants", type=int, default=8)
    parser.add_argument("--photo-seed", type=int, default=703231)
    parser.add_argument("--photo-names", default="astronaut,coffee,chelsea,rocket")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    frequencies = (1, 2, 4, 8, 16, 32, 64)
    layer = PhotometricSpectralTutteLayer(
        args.side, fit_side=args.fit_side,
        frequencies=frequencies).to(device)
    with torch.no_grad():
        preparation = layer.prepare(device=device)

    def evaluate(fixed: torch.Tensor, moving: torch.Tensor) -> dict:
        coefficients = []
        with torch.no_grad():
            for first in range(0, len(fixed), args.batch):
                last = first + args.batch
                current = layer._latent(
                    fixed[first:last].to(device),
                    moving[first:last].to(device))
                coefficients.append(current.cpu().numpy().reshape(-1, 3, 7))
        values = np.concatenate(coefficients)
        energy64 = np.linalg.norm(values[:, :, 6], axis=1)
        energy32 = np.linalg.norm(values[:, :, 5], axis=1)
        total = np.linalg.norm(values.reshape(len(values), -1), axis=1)
        return {
            "energy64": energy64.tolist(),
            "energy32": energy32.tolist(),
            "fraction64": (energy64 / np.maximum(total, 1e-12)).tolist(),
            "ratio64_to32": (energy64 / np.maximum(energy32, 1e-12)).tolist(),
        }

    synthetic = {}
    for family in ("high32", "high64"):
        fixed, moving, *_ = make_dataset(
            args.count, 512, args.seed, target_family=family,
            return_coefficients=True)
        synthetic[family] = evaluate(fixed, moving)

    names = [name.strip() for name in args.photo_names.split(",") if name.strip()]
    photo_names, _, (fixed, moving, *_ ) = photo_dataset(
        args.photo_variants, 512, args.photo_seed, device, names)
    photo = evaluate(fixed, moving)
    _, _, (fixed64, moving64, *_) = photo_dataset(
        args.photo_variants, 512, args.photo_seed, device, names,
        target_family="high64")
    photo64 = evaluate(fixed64, moving64)

    score32 = np.asarray(synthetic["high32"]["fraction64"])
    score64 = np.asarray(synthetic["high64"]["fraction64"])
    pairwise = score64[:, None] - score32[None, :]
    auc = float(np.mean(pairwise > 0) + 0.5*np.mean(pairwise == 0))
    photo_score32 = np.asarray(photo["fraction64"])
    photo_score64 = np.asarray(photo64["fraction64"])
    photo_pairwise = photo_score64[:, None] - photo_score32[None, :]
    photo_auc = float(np.mean(photo_pairwise > 0)
                      + 0.5*np.mean(photo_pairwise == 0))
    output = {
        "question": "does the neutral-gain image-derived latent separate high32 and high64 synthetic targets?",
        "side": args.side, "control_vertices": args.side**2,
        "fit_side": args.fit_side, "image_side": 512,
        "frequencies": frequencies, "device": str(device),
        "mode_gains": "all neutral 1, no trained checkpoint",
        "count_per_synthetic_family": args.count,
        "synthetic_seed": args.seed,
        "photo_names": photo_names,
        "photo_variants_per_name": args.photo_variants,
        "photo_seed": args.photo_seed,
        "response_validation": preparation,
        "synthetic": synthetic,
        "photographic_high32": photo,
        "photographic_high64": photo64,
        "fraction64_auc_high64_vs_high32": auc,
        "photo_fraction64_auc_high64_vs_high32": photo_auc,
        "photo_paired_high64_fraction_greater_count":
            int(np.count_nonzero(photo_score64 > photo_score32)),
        "paired_high64_fraction_greater_count":
            int(np.count_nonzero(score64 > score32)),
        "median_fraction64": {
            "high32": float(np.median(score32)),
            "high64": float(np.median(score64)),
            "photo_high32": float(np.median(photo["fraction64"])),
            "photo_high64": float(np.median(photo64["fraction64"])),
        },
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
