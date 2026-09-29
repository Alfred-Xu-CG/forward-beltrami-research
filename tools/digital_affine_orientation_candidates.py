"""Image-only four-quarter-turn affine-initialization diagnostic.

Each candidate remains a positive-determinant affine. Selection reads
grayscale images and an image-only initial affine, never landmarks or DHR
full fields. The MIND-like descriptor criterion may still select badly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_q1_real_optimize import _read_gray_thumbnail


ROTATIONS = (
    (np.eye(2), np.zeros(2)),
    (np.array([[0., -1.], [1., 0.]]), np.array([1., 0.])),
    (-np.eye(2), np.ones(2)),
    (np.array([[0., 1.], [-1., 0.]]), np.array([0., 1.])),
)


def score_pair(name: str, fixed_path: Path, moving_path: Path,
               affine_path: Path, device: torch.device) -> dict:
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with np.load(affine_path) as data:
        matrix = np.asarray(data["post_affine_matrix"], dtype=np.float64)
        offset = np.asarray(data["post_affine_offset"], dtype=np.float64)
    if matrix.shape != (2, 2) or offset.shape != (2,) or np.linalg.det(matrix) <= 0:
        raise ValueError("positive image-only affine required")
    fixed128 = F.interpolate(fixed, size=(128, 128), mode="area")
    fixed_desc, fixed_scale = self_similarity(fixed128)
    mask = ((fixed128 > .04) & (fixed_scale > 1e-4)).to(fixed.dtype)
    losses = []
    candidate_affines = []
    with torch.no_grad():
        for rotation, translation in ROTATIONS:
            m = matrix @ rotation
            b = matrix @ translation + offset
            aligned = warp_moving_to_fixed(
                moving, torch.tensor(m, device=device, dtype=torch.float32),
                torch.tensor(b, device=device, dtype=torch.float32),
                height=512, width=512)
            moving128 = F.interpolate(aligned, size=(128, 128), mode="area")
            moving_desc, _ = self_similarity(moving128)
            loss = ((fixed_desc - moving_desc).abs() * mask).sum() / (
                mask.sum() * fixed_desc.shape[1] + 1e-8)
            losses.append(float(loss))
            candidate_affines.append((m.tolist(), b.tolist()))
    chosen = int(np.argmin(losses))
    return {
        "name": name, "losses_0_90_180_270": losses,
        "chosen_quarter_turn": chosen,
        "chosen_vs_original_loss": losses[chosen] - losses[0],
        "chosen_matrix": candidate_affines[chosen][0],
        "chosen_offset": candidate_affines[chosen][1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--pair", action="append", nargs=4,
                        metavar=("NAME", "FIXED", "MOVING", "AFFINE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = []
    if args.selection is not None:
        if args.root is None:
            raise ValueError("--root required with --selection")
        ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
        for case in ids:
            affine = args.root / f"{case}_directSG_affine.npz"
            if affine.exists():
                fixed, moving = _case_images(args.root, case)
                pairs.append((str(case), fixed, moving, affine))
    if args.pair:
        pairs += [(name, Path(fixed), Path(moving), Path(affine))
                  for name, fixed, moving, affine in args.pair]
    if not pairs:
        raise ValueError("at least one available pair required")
    device = torch.device(args.device)
    started = time.perf_counter()
    rows = [score_pair(name, fixed, moving, affine, device)
            for name, fixed, moving, affine in pairs]
    report = {
        "method": "image-only four-quarter-turn masked MIND-like descriptor selector",
        "case_count": len(rows), "device": args.device,
        "seconds": time.perf_counter() - started,
        "selected_counts": {str(k): sum(r["chosen_quarter_turn"] == k for r in rows)
                            for k in range(4)},
        "rows": rows, "anatomical_landmarks_or_DHR_full_field_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "case_count", "selected_counts", "seconds",
    )}))


if __name__ == "__main__":
    main()
