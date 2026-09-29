"""Case-held-out ACROBAT pseudo-teacher probe for the existing safe Q1 CNN.

An image-derived DHR initial affine is supplied to *both* the identity and
learned-residual arms. Full DHR fields are training labels only and held-out
evaluation pseudo-targets, never inference inputs. This isolates learned safe
residual usefulness; it is not an end-to-end fast registration method or an
anatomical accuracy experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_network_teacher import _save_checkpoint


def affine_prewarp(moving: torch.Tensor, matrix: torch.Tensor,
                   offset: torch.Tensor) -> torch.Tensor:
    """Evaluate I_m(A_0(q)) at fixed pixel centers; no topology claim here."""
    if moving.ndim != 4 or moving.shape[1] != 1 or matrix.shape != (
        moving.shape[0], 2, 2
    ) or offset.shape != (moving.shape[0], 2):
        raise ValueError("expected moving B1HW, affine B22 and offset B2")
    q = fixed_pixel_centers(moving.shape[-2], moving.shape[-1],
                            dtype=moving.dtype, device=moving.device)
    mapped = torch.einsum("bhwi,bji->bhwj", q.expand(moving.shape[0], -1, -1, -1),
                          matrix) + offset[:, None, None, :]
    return F.grid_sample(moving, 2 * mapped - 1, mode="bilinear",
                         padding_mode="border", align_corners=False)


def factored_residual_target(full_teacher: torch.Tensor, matrix: torch.Tensor,
                             offset: torch.Tensor) -> torch.Tensor:
    """For T=A_0(U), return U=A_0^{-1}(T); no target-topology assumption."""
    if full_teacher.ndim != 4 or full_teacher.shape[-1] != 2 or (
        matrix.shape != (full_teacher.shape[0], 2, 2)
        or offset.shape != (full_teacher.shape[0], 2)
    ):
        raise ValueError("teacher BHWC and affine batch dimensions must agree")
    if not bool(torch.isfinite(full_teacher).all()) or not bool(torch.isfinite(matrix).all()):
        raise ValueError("finite teacher and matrix required")
    if bool((torch.linalg.det(matrix) <= 0).any()):
        raise ValueError("positive affine determinant required")
    return torch.linalg.solve(
        matrix[:, None, None], (full_teacher - offset[:, None, None])[:, :, :, :, None]
    ).squeeze(-1)


def split_case_ids(all_ids: list[int], test_ids: list[int]) -> tuple[list[int], list[int]]:
    if len(set(all_ids)) != len(all_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("duplicate physical case id")
    if not test_ids or not set(test_ids) < set(all_ids):
        raise ValueError("test cases must be a nonempty strict subset")
    return [case for case in all_ids if case not in test_ids], list(test_ids)


def _gray(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        array = np.asarray(im.convert("L"), dtype=np.float32)
    if array.shape != (512, 512):
        raise ValueError(f"physical canvas must be 512 square: {path}")
    return torch.from_numpy(1. - array / 255.)[None, None]


def _case_images(root: Path, case: int) -> tuple[Path, Path]:
    fixed = root / f"{case}_HE_physical512.png"
    candidates = sorted(root.glob(f"{case}_*_physical512.png"))
    moving = [p for p in candidates if p != fixed]
    if not fixed.is_file() or len(moving) != 1:
        raise FileNotFoundError(f"expected exactly HE and one stain for case {case}")
    return fixed, moving[0]


def load_case_inputs(root: Path, case: int, device: torch.device, *,
                     affine_source: str = "DHR") -> dict:
    """Read only inference inputs; no full-DHR teacher is accessed."""
    fixed_path, moving_path = _case_images(root, case)
    if affine_source not in ("DHR", "image_only"):
        raise ValueError("affine source must be DHR or image_only")
    suffix = ("DHR_physical_initial_teacher_affine" if affine_source == "DHR"
              else "directSG_affine")
    with np.load(root / f"{case}_{suffix}.npz") as initial:
        matrix = torch.from_numpy(initial["post_affine_matrix"].astype(np.float32))[None].to(device)
        offset = torch.from_numpy(initial["post_affine_offset"].astype(np.float32))[None].to(device)
    fixed = _gray(fixed_path).to(device)
    moving = _gray(moving_path).to(device)
    prewarped = affine_prewarp(moving, matrix, offset)
    return {"id": case, "fixed": fixed, "moving": moving,
            "prewarped": prewarped, "matrix": matrix, "offset": offset,
            "affine_source": affine_source}


def load_case_teacher(root: Path, case: int, device: torch.device,
                      matrix: torch.Tensor, offset: torch.Tensor) -> dict:
    """Read the full-DHR pseudo-target only after the inference stage."""
    with np.load(root / f"{case}_DHR_physical_full_teacher_affine.npz") as full:
        raw = torch.from_numpy(full["raw_teacher_vertices"].astype(np.float32)).to(device)
    target = factored_residual_target(raw, matrix, offset)
    return {"raw_teacher": raw, "target": target}


def load_case(root: Path, case: int, device: torch.device) -> dict:
    """Read images, initial affine and full teacher for development training."""
    example = load_case_inputs(root, case, device)
    example.update(load_case_teacher(root, case, device, example["matrix"],
                                     example["offset"]))
    return example


def _map_rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float((predicted - target).square().sum(-1).mean().sqrt())


def run(*, root: Path, output: Path, all_ids: list[int], test_ids: list[int],
        steps: int, batch: int, device: str, seed: int = 20260929,
        learning_rate: float = .002, flow_hint: bool = True) -> dict:
    train_ids, heldout_ids = split_case_ids(all_ids, test_ids)
    if steps < 1 or batch < 1 or learning_rate <= 0:
        raise ValueError("positive steps, batch and learning rate required")
    target_device = torch.device(device)
    torch.manual_seed(seed)
    cases = {case: load_case(root, case, target_device) for case in all_ids}
    model = Q1ImageRegistrationNetwork(seed_side=17, final_side=257,
                                       width=16, feature_side=257,
                                       flow_hint=flow_hint).to(target_device)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    step_times: list[float] = []
    losses: list[float] = []
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    model.train()
    for step in range(steps):
        selected = torch.randint(len(train_ids), (batch,), generator=rng).tolist()
        examples = [cases[train_ids[index]] for index in selected]
        fixed = torch.cat([example["fixed"] for example in examples])
        prewarped = torch.cat([example["prewarped"] for example in examples])
        target = torch.cat([example["target"] for example in examples])
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        tic = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        residual, _, _ = model(fixed, prewarped)
        loss = (residual[:, 1:-1, 1:-1] - target[:, 1:-1, 1:-1]).square().sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at {step}")
        loss.backward()
        gradients = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        if not gradients or any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError(f"nonfinite encoder gradient at {step}")
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        step_times.append(time.perf_counter() - tic)
        losses.append(float(loss.detach()))
    training_peak = (torch.cuda.max_memory_allocated(target_device)
                     if target_device.type == "cuda" else None)
    output.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(output / "acrobat_teacher_probe_weights.npz", model)
    identity = identity_vertices(257, device=target_device)
    model.eval()
    results = []
    with torch.no_grad():
        for case in all_ids:
            example = cases[case]
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            tic = time.perf_counter()
            predicted, _, _ = model(example["fixed"], example["prewarped"])
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            inference_s = time.perf_counter() - tic
            matrix, offset = example["matrix"], example["offset"]
            predicted_full = Q1ImageRegistrationNetwork.apply_affine(predicted, matrix, offset)
            identity_full = Q1ImageRegistrationNetwork.apply_affine(identity, matrix, offset)
            map_path = output / f"{case}_predicted_safe_q1.npz"
            np.savez_compressed(map_path, vertices=predicted.cpu().numpy().astype(np.float32),
                                boundary_reference=identity.cpu().numpy().astype(np.float32),
                                post_affine_matrix=matrix[0].cpu().numpy().astype(np.float32),
                                post_affine_offset=offset[0].cpu().numpy().astype(np.float32))
            certificate = certify_q1_binary_map(map_path)
            if not certificate["valid"]:
                raise ArithmeticError(f"saved predicted map invalid for case {case}: {certificate}")
            results.append({
                "case": case, "split": "test" if case in heldout_ids else "train",
                "predicted_to_DHR_full_vertex_rmse": _map_rmse(predicted_full, example["raw_teacher"]),
                "initial_affine_to_DHR_full_vertex_rmse": _map_rmse(identity_full, example["raw_teacher"]),
                "inference_seconds_excluding_initial_affine": inference_s,
                "saved_residual_certificate": certificate,
            })
    report = {
        "mode": "ACROBAT_case_heldout_DHR_pseudoteacher_existing_CNN",
        "train_case_ids": train_ids, "test_case_ids": heldout_ids,
        "full_DHR_used_at_inference": False,
        "initial_DHR_affine_used_at_inference": True,
        "initial_affine_cost_in_timing": False,
        "fixed_and_moving_images": "common-physical-scale 512x512 white-padded canvases",
        "control_vertices": 257 * 257, "steps": steps, "batch": batch,
        "learning_rate": learning_rate, "seed": seed,
        "flow_hint": flow_hint,
        "first_train_loss": losses[0], "last_train_loss": losses[-1],
        "median_training_step_seconds": statistics.median(step_times),
        "peak_torch_cuda_allocated_bytes": training_peak,
        "cases": results,
    }
    (output / "acrobat_teacher_probe_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-ids", type=int, nargs="+", default=[495, 733])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--no-flow-hint", action="store_true",
                        help="ablate the local brightness-constancy proposal channels")
    args = parser.parse_args()
    result = run(root=args.root, output=args.output,
                 all_ids=[100, 156, 315, 330, 399, 495, 585, 586, 638, 733],
                 test_ids=args.test_ids, steps=args.steps, batch=args.batch,
                 device=args.device, seed=args.seed,
                 flow_hint=not args.no_flow_hint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
