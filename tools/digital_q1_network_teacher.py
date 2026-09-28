"""One-pair image-to-latent Q1 integration diagnostic using an image-only teacher.

This is supervised distillation of a precomputed registration field, not a
held-out registration experiment. Landmarks are never read here. The saved
factorization is U followed by A: raw Q1 residual vertices, positive-affine
matrix, and offset. A later evaluator may read labels independently.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def _identity(side: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def train_to_teacher(
    fixed: torch.Tensor, moving: torch.Tensor, teacher: torch.Tensor, *,
    seed_side: int = 17, steps: int = 200, learning_rate: float = .002,
    device: str = "cuda:0", width: int = 16, feature_side: int = 257,
    flow_hint: bool = True,
) -> tuple[Q1ImageRegistrationNetwork,
           tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
    """Fit an image encoder and safe geometry to one or more teacher maps."""
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
        raise ValueError("fixed/moving must be matching BCHW grayscale images")
    if fixed.shape[0] < 1 or teacher.ndim != 4 or teacher.shape[0] != fixed.shape[0] or (
        teacher.shape[1] != teacher.shape[2] or teacher.shape[-1] != 2
    ):
        raise ValueError("matching nonempty batch of square Q1 teachers required")
    if not all(bool(torch.isfinite(x).all()) for x in (fixed, moving, teacher)):
        raise ValueError("nonfinite input")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("positive steps and learning rate required")
    target_device = torch.device(device)
    fixed = fixed.to(device=target_device, dtype=torch.float32)
    moving = moving.to(device=target_device, dtype=torch.float32)
    teacher = teacher.to(device=target_device, dtype=torch.float32)
    side = teacher.shape[1]
    model = Q1ImageRegistrationNetwork(
        seed_side=seed_side, final_side=side, width=width,
        feature_side=feature_side, flow_hint=flow_hint,
    ).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reference = _identity(side, target_device).expand(fixed.shape[0], -1, -1, -1)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    best_loss = float("inf")
    best_state = None
    first_loss = None
    last_loss = None
    durations: list[float] = []
    finite_steps = 0
    for _ in range(steps):
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        residual, matrix, offset = model(fixed, moving)
        mapped = model.apply_affine(residual, matrix, offset)
        loss = (mapped - teacher).square().sum(-1).mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        finite_steps += int(finite)
        if not finite:
            raise FloatingPointError("nonfinite or missing network gradient")
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - started)
    assert best_state is not None and first_loss is not None and last_loss is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(fixed, moving)
    residual, matrix, offset = output
    effective = model.apply_affine(residual, matrix, offset)
    per_pair_rmse = (effective - teacher).square().sum(-1).mean(dim=(-2, -1)).sqrt()
    validity = validate_q1_map(residual, reference)
    report = {
        "mode": "image_encoder_DHR_teacher_distillation_not_G2",
        "batch": int(fixed.shape[0]),
        "control_side": side,
        "image_size_hw": list(fixed.shape[-2:]),
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_vector_rmse": first_loss ** .5,
        "last_vector_rmse": last_loss ** .5,
        "best_vector_rmse": best_loss ** .5,
        "best_per_pair_vector_rmse": [float(value) for value in per_pair_rmse],
        "median_complete_step_seconds": statistics.median(durations),
        "finite_gradient_steps": finite_steps,
        "nonpositive_residual_corners": validity["nonpositive_corners"],
        "residual_boundary_max_error": validity["boundary_max_error"],
        "affine_determinants": [float(value) for value in torch.linalg.det(matrix)],
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
    }
    return model, output, report


def _save_checkpoint(path: Path, model: Q1ImageRegistrationNetwork) -> None:
    payload = {
        "meta_seed_side": np.asarray(model.decoder.seed_side),
        "meta_final_side": np.asarray(model.decoder.final_side),
        "meta_width": np.asarray(model.encoder.stem[0].out_channels),
        "meta_feature_side": np.asarray(model.encoder.feature_side),
        "meta_flow_hint": np.asarray(int(model.encoder.flow_hint)),
    }
    payload.update({
        "state__" + key: value.detach().cpu().numpy()
        for key, value in model.state_dict().items()
    })
    np.savez_compressed(path, **payload)


def load_checkpoint(path: Path, *, device: str) -> Q1ImageRegistrationNetwork:
    with np.load(path) as archive:
        model = Q1ImageRegistrationNetwork(
            seed_side=int(archive["meta_seed_side"]),
            final_side=int(archive["meta_final_side"]),
            width=int(archive["meta_width"]),
            feature_side=int(archive["meta_feature_side"]),
            flow_hint=bool(int(archive["meta_flow_hint"])),
        )
        state = {
            key[len("state__"):]: torch.from_numpy(archive[key].copy())
            for key in archive.files if key.startswith("state__")
        }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def input_sensitivity(
    model: Q1ImageRegistrationNetwork, fixed: torch.Tensor, moving: torch.Tensor,
) -> dict[str, float | int]:
    """Diagnostic only: map response to image swap and all-zero images.

    A nonzero value establishes input dependence, not useful correspondence.
    """
    if fixed.shape != moving.shape:
        raise ValueError("image pair shapes must match")
    device = next(model.parameters()).device
    fixed = fixed.to(device=device, dtype=torch.float32)
    moving = moving.to(device=device, dtype=torch.float32)
    baseline = model.apply_affine(*model(fixed, moving))
    swapped = model.apply_affine(*model(moving, fixed))
    blank = torch.zeros_like(fixed)
    blank_output = model.apply_affine(*model(blank, blank))
    rmse = lambda other: float((baseline - other).square().sum(-1).mean().sqrt())
    identity = _identity(baseline.shape[1], device)
    return {
        "original_map_vector_rmse_from_identity": rmse(identity),
        "swapped_input_map_vector_rmse": rmse(swapped),
        "blank_input_map_vector_rmse": rmse(blank_output),
        "original_outside_source_control_vertices": int(
            ((baseline < 0) | (baseline > 1)).any(-1).sum()),
    }


def _save_map(path: Path, output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict:
    residual, matrix, offset = output
    if residual.shape[0] != 1:
        raise ValueError("saved map archive currently supports one pair")
    side = residual.shape[1]
    reference = _identity(side, torch.device("cpu"))
    np.savez_compressed(
        path,
        vertices=residual.detach().cpu().numpy(),
        boundary_reference=reference.numpy(),
        post_affine_matrix=matrix[0].detach().cpu().numpy(),
        post_affine_offset=offset[0].detach().cpu().numpy(),
    )
    certificate = certify_q1_binary_map(path)
    return {
        "saved_residual_valid": bool(certificate["valid"]),
        "saved_residual_nonpositive_corners": int(certificate["nonpositive_corners"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--teacher", type=Path, required=True)
    training.add_argument("--output-map", type=Path, required=True)
    training.add_argument("--output-weights", type=Path, required=True)
    training.add_argument("--steps", type=int, default=200)
    training.add_argument("--learning-rate", type=float, default=.002)
    inference = commands.add_parser("infer")
    inference.add_argument("--weights", type=Path, required=True)
    inference.add_argument("--output-map", type=Path, required=True)
    for command in (training, inference):
        command.add_argument("--fixed-image", type=Path, required=True)
        command.add_argument("--moving-image", type=Path, required=True)
        command.add_argument("--image-side", type=int, default=512)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--ablation", action="store_true",
                             help="report image-swap/blank-map sensitivity; not an accuracy score")
        command.add_argument("--output-report", type=Path,
                             help="save inference/training metadata beside the map")
    args = parser.parse_args()
    fixed, fixed_size = _read_gray_thumbnail(args.fixed_image, args.image_side)
    moving, moving_size = _read_gray_thumbnail(args.moving_image, args.image_side)
    if args.command == "train":
        with np.load(args.teacher) as archive:
            teacher = torch.from_numpy(archive["raw_teacher_vertices"] if
                                       "raw_teacher_vertices" in archive else
                                       archive["teacher_vertices"])
        model, output, report = train_to_teacher(
            fixed, moving, teacher, steps=args.steps,
            learning_rate=args.learning_rate, device=args.device,
        )
        _save_checkpoint(args.output_weights, model)
        report["weights_path"] = str(args.output_weights)
    else:
        model = load_checkpoint(args.weights, device=args.device)
        with torch.no_grad():
            output = model(fixed.to(args.device), moving.to(args.device))
        report = {"mode": "frozen_network_inference", "weights_path": str(args.weights)}
    report.update(_save_map(args.output_map, output))
    if args.ablation:
        report.update(input_sensitivity(model, fixed, moving))
    report.update({
        "fixed_image": str(args.fixed_image), "moving_image": str(args.moving_image),
        "fixed_original_size_xy": list(fixed_size),
        "moving_original_size_xy": list(moving_size),
        "output_map": str(args.output_map),
        "input_convention": "independent whole-JPEG 512-style unit-square pixel-center thumbnails",
    })
    if args.output_report is not None:
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
