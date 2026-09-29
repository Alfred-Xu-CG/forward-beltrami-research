"""Read-only MIND-like objective check on image-only candidate maps.

No landmarks or full DHR fields are loaded. This first determines whether
the proposed cross-stain descriptor even ranks existing maps plausibly before
using it to optimize latent variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.digital_q1_p1_layer import FactorizedP1Map, evaluate_factorized_p1
from tools.digital_q1_real_optimize import _read_gray_thumbnail


OFFSETS = ((2, 0), (-2, 0), (0, 2), (0, -2),
           (2, 2), (2, -2), (-2, 2), (-2, -2))


def self_similarity(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight normalized patch self-similarities and a local-texture scale."""
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("B1HW image required")
    height, width = image.shape[-2:]
    padded = F.pad(image, (2, 2, 2, 2), mode="replicate")
    squared = []
    for dx, dy in OFFSETS:
        shifted = padded[:, :, 2 + dy:2 + dy + height, 2 + dx:2 + dx + width]
        squared.append(F.avg_pool2d((image - shifted).square(),
                                    3, stride=1, padding=1, count_include_pad=False))
    distances = torch.cat(squared, dim=1)
    scale = distances.mean(dim=1, keepdim=True)
    normalized = (distances - distances.amin(dim=1, keepdim=True)) / (scale + 1e-4)
    return torch.exp(-normalized), scale


def sampled_p1_descriptor(descriptor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    side = descriptor.shape[-1]
    if descriptor.shape[-2] != side:
        raise ValueError("square descriptor required")
    batch = vertices.shape[0]
    if descriptor.shape[0] != batch:
        raise ValueError("descriptor and map batch must match")
    query = fixed_pixel_centers(side, side, dtype=vertices.dtype,
                                device=vertices.device).reshape(1, side * side, 2).expand(batch, -1, -1)
    mapped = FactorizedP1Map(
        vertices, torch.eye(2, device=vertices.device, dtype=vertices.dtype)[None].expand(batch, -1, -1),
        torch.zeros((batch, 2), device=vertices.device, dtype=vertices.dtype),
    )
    target = evaluate_factorized_p1(mapped, query).reshape(batch, side, side, 2)
    return F.grid_sample(descriptor, 2 * target - 1, mode="bilinear",
                         padding_mode="border", align_corners=False)


def objective(fixed: torch.Tensor, moving: torch.Tensor,
              vertices: torch.Tensor) -> tuple[torch.Tensor, float]:
    fixed_feature, fixed_scale = self_similarity(fixed)
    moving_feature, _ = self_similarity(moving)
    warped = sampled_p1_descriptor(moving_feature, vertices)
    mask = ((fixed > .04) & (fixed_scale > 1e-4)).to(fixed.dtype)
    loss = ((fixed_feature - warped).abs() * mask).sum() / (
        mask.sum() * fixed_feature.shape[1] + 1e-8)
    return loss, float(mask.mean())


def affine_vertices(matrix_path: Path, *, side: int, device: torch.device) -> torch.Tensor:
    with np.load(matrix_path) as factor:
        matrix = torch.tensor(factor["post_affine_matrix"].copy(), device=device, dtype=torch.float32)
        offset = torch.tensor(factor["post_affine_offset"].copy(), device=device, dtype=torch.float32)
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    reference = torch.stack((x, y), dim=-1)[None]
    return reference @ matrix.T + offset


def run(fixed_path: Path, moving_path: Path, affine: Path,
        maps: dict[str, Path], output: Path, *, side: int, device_name: str) -> dict:
    from tools.digital_q1_real_eval import load_effective_vertices
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(device_name)
    fixed, _ = _read_gray_thumbnail(fixed_path, side)
    moving, _ = _read_gray_thumbnail(moving_path, side)
    fixed, moving = fixed.to(device), moving.to(device)
    candidates = {"initial_affine": affine_vertices(affine, side=17, device=device)}
    certificates = {}
    for name, path in maps.items():
        data, certificate = load_effective_vertices(path)
        candidates[name] = torch.tensor(data[None], device=device, dtype=torch.float32)
        certificates[name] = certificate["composite_representation_valid"]
    scores = {}
    with torch.no_grad():
        for name, vertices in candidates.items():
            loss, mask_fraction = objective(fixed, moving, vertices)
            scores[name] = float(loss)
    result = {"question": "Does a frozen MIND-like structural objective rank existing maps?",
              "descriptor": "eight r=2 self-similarity channels, local 3x3 SSD, fixed tissue mask",
              "image_side": side, "fixed": str(fixed_path), "moving": str(moving_path),
              "affine": str(affine), "maps": {k: str(v) for k, v in maps.items()},
              "map_certificates_valid": certificates,
              "mask_fraction": mask_fraction, "scores_lower_is_better": scores,
              "landmarks_loaded": False, "full_DHR_displacement_loaded": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--neural-map", type=Path, required=True)
    parser.add_argument("--fixed-gaussian-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args.fixed_image, args.moving_image, args.affine,
                 {"neural": args.neural_map, "fixed_gaussian": args.fixed_gaussian_map},
                 args.output, side=args.side, device_name=args.device)
    print(json.dumps(result["scores_lower_is_better"]))


if __name__ == "__main__":
    main()
