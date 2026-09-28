"""Train one safe Q1 image predictor on several independent image pairs.

Input pairs have precomputed image-only DHR teachers; no landmark paths are
accepted. A leave-one-group-out pilot with only three available development
sets is a pipeline stress test, not a statistically credible G2 result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.digital_q1_network_teacher import _save_checkpoint, train_to_teacher
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def read_pair_batch(
    pairs: list[tuple[Path, Path, Path]], *, image_side: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    if len(pairs) < 2 or image_side < 16:
        raise ValueError("at least two pairs and image_side>=16 required")
    fixed_images = []
    moving_images = []
    teacher_maps = []
    metadata = []
    for fixed_path, moving_path, teacher_path in pairs:
        fixed, fixed_size = _read_gray_thumbnail(fixed_path, image_side)
        moving, moving_size = _read_gray_thumbnail(moving_path, image_side)
        with np.load(teacher_path) as archive:
            key = "raw_teacher_vertices" if "raw_teacher_vertices" in archive else "teacher_vertices"
            teacher = torch.from_numpy(archive[key].copy())
        if teacher.shape[0] != 1 or teacher.shape[-1] != 2 or teacher.shape[1] != teacher.shape[2]:
            raise ValueError(f"invalid teacher Q1 table: {teacher_path}")
        fixed_images.append(fixed)
        moving_images.append(moving)
        teacher_maps.append(teacher)
        metadata.append({
            "fixed": str(fixed_path), "moving": str(moving_path),
            "teacher": str(teacher_path),
            "fixed_size_xy": list(fixed_size), "moving_size_xy": list(moving_size),
        })
    if len({tuple(target.shape) for target in teacher_maps}) != 1:
        raise ValueError("all teachers need the same control grid")
    return (torch.cat(fixed_images), torch.cat(moving_images),
            torch.cat(teacher_maps), metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", metavar=("FIXED", "MOVING", "TEACHER"),
                        required=True, help="repeat for each TRAIN group; never pass evaluation landmarks")
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, help="save the exact fold membership and training diagnostics")
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--seed", type=int, default=290929)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    pairs = [(Path(a), Path(b), Path(c)) for a, b, c in args.pair]
    fixed, moving, teacher, metadata = read_pair_batch(pairs, image_side=args.image_side)
    model, _, report = train_to_teacher(
        fixed, moving, teacher, steps=args.steps,
        learning_rate=args.learning_rate, device=args.device,
    )
    _save_checkpoint(args.output_weights, model)
    report.update({
        "train_pairs": metadata,
        "output_weights": str(args.output_weights),
        "seed": args.seed,
        "warning": "teacher-conditioned 2-group pilot; no landmarks in training; not formal G2",
    })
    if args.output_report is not None:
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
