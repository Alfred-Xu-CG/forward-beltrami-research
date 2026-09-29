"""Image-only frozen Q1 prediction on a BIRL/ANHIR development canvas.

No landmark or full-DHR path is accepted. Saves actual/blank outputs separately
and certifies their *stored* binary32 factorized Q1/P1 topology.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_acrobat_blank_ablation import blank_output
from tools.digital_acrobat_teacher_probe import _gray, affine_prewarp
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_network_teacher import load_checkpoint


def predict(fixed: Path, moving: Path, initial_affine: Path, checkpoint: Path,
            output: Path, *, device: str = "cpu") -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh prediction output required")
    target_device = torch.device(device)
    with np.load(initial_affine) as data:
        matrix = torch.from_numpy(np.asarray(data["post_affine_matrix"],
                                             dtype=np.float32).copy())[None].to(target_device)
        offset = torch.from_numpy(np.asarray(data["post_affine_offset"],
                                             dtype=np.float32).copy())[None].to(target_device)
    if matrix.shape != (1, 2, 2) or offset.shape != (1, 2) or (
        not bool(torch.isfinite(matrix).all()) or
        not bool(torch.isfinite(offset).all()) or
        float(torch.linalg.det(matrix)) <= 0
    ):
        raise ValueError("finite orientation-preserving initial affine required")
    model = load_checkpoint(checkpoint, device=device).eval()
    fixed_image = _gray(fixed).to(target_device)
    moving_image = _gray(moving).to(target_device)
    prewarped = affine_prewarp(moving_image, matrix, offset)
    reference = identity_vertices(257, device=target_device).cpu().numpy()
    with torch.no_grad():
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        start = time.perf_counter()
        actual, _, _ = model(fixed_image, prewarped)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        forward_seconds = time.perf_counter() - start
        blank = blank_output(model, fixed_image, prewarped)
    output.mkdir(parents=True, exist_ok=True)
    certificates = {}
    for arm, map_vertices in (("actual", actual), ("blank", blank)):
        archive = output / f"{arm}_safe_q1.npz"
        np.savez_compressed(
            archive, vertices=map_vertices.cpu().numpy().astype(np.float32),
            boundary_reference=reference.astype(np.float32),
            post_affine_matrix=matrix[0].cpu().numpy().astype(np.float32),
            post_affine_offset=offset[0].cpu().numpy().astype(np.float32),
        )
        certificates[arm] = certify_q1_binary_map(archive)
        if not certificates[arm]["valid"]:
            raise ArithmeticError(f"saved {arm} map failed topology certificate")
    report = {
        "mode": "frozen_image_only_BIRL_development_prediction",
        "fixed": str(fixed), "moving": str(moving),
        "initial_affine": str(initial_affine), "checkpoint": str(checkpoint),
        "landmarks_used": False, "full_DHR_field_used": False,
        "network_forward_seconds_excluding_affine_prewarp_io": forward_seconds,
        "certificates": certificates,
    }
    (output / "prediction.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "initial_affine", "checkpoint", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(predict(args.fixed, args.moving, args.initial_affine,
                             args.checkpoint, args.output, device=args.device)))


if __name__ == "__main__":
    main()
