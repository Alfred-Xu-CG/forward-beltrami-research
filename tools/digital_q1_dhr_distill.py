"""Fit the safe Q1 decoder to an image-only DHR field, without landmarks.

This hybrid teacher experiment tests representation capacity. It is not an
independent image-to-latent method: a DHR run supplies the target coordinates.
The two-stage CLI permits target extraction with SimpleITK on D: and fitting
with PyTorch on another machine without copying the original JPEGs there.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qcopt.neural_bijection.dense import HybridPatchSeedVertexQ1Pyramid
from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_compare_appearance import dhr_map_at_unit_queries


def identity_vertices(side: int, *, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1)[None]


def factor_affine_teacher(
    teacher: torch.Tensor, reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares positive affine fit and fixed-boundary residual target.

    The teacher is fixed external data, so this preprocessing is intentionally
    non-differentiable. The safe decoder and its trained latent path remain
    differentiable; a later image encoder would need its own affine head.
    """
    if teacher.shape != reference.shape or teacher.ndim != 4 or (
        teacher.shape[0] != 1 or teacher.shape[-1] != 2
    ) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("finite matching B=1 vertex tables required")
    source = reference.detach().cpu().numpy().reshape(-1, 2).astype(np.float64)
    target = teacher.detach().cpu().numpy().reshape(-1, 2).astype(np.float64)
    design = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    matrix = coefficients[:2].T
    offset = coefficients[2]
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or determinant <= 1e-8:
        raise ValueError("affine teacher factor needs well-conditioned positive orientation")
    residual = np.linalg.solve(matrix, (target - offset).T).T.reshape(teacher.shape)
    return (
        torch.as_tensor(residual, device=teacher.device, dtype=teacher.dtype),
        torch.as_tensor(matrix.copy(), device=teacher.device, dtype=teacher.dtype),
        torch.as_tensor(offset.copy(), device=teacher.device, dtype=teacher.dtype),
    )


def prepare_target(
    field_path: Path, params_path: Path, moving_image: Path, fixed_image: Path,
    output_path: Path, *, side: int, affine_factor: bool = False,
) -> dict:
    import SimpleITK as sitk

    if side < 17:
        raise ValueError("control side must be at least 17")
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(field_path)))
    params = json.loads(params_path.read_text())
    if float(params.get("initial_resample_ratio", 1)) != 1:
        raise ValueError("unaccounted DHR initial resample ratio")
    with Image.open(moving_image) as image:
        moving_size = image.size
    with Image.open(fixed_image) as image:
        fixed_size = image.size
    reference = identity_vertices(side, device=torch.device("cpu"))
    target = dhr_map_at_unit_queries(
        field, params, fixed_size=fixed_size, moving_size=moving_size,
        query=reference,
    )
    validity = validate_q1_map(target, reference)
    residual = target
    affine_matrix = None
    affine_offset = None
    if affine_factor:
        residual, affine_matrix, affine_offset = factor_affine_teacher(target, reference)
    payload = {
        "teacher_vertices": residual.numpy(),
        "boundary_reference": reference.numpy(),
    }
    if affine_matrix is not None and affine_offset is not None:
        payload.update({
            "post_affine_matrix": affine_matrix.numpy(),
            "post_affine_offset": affine_offset.numpy(),
            "raw_teacher_vertices": target.numpy(),
        })
    np.savez_compressed(output_path, **payload)
    result = {
        "mode": "prepare_DHR_image_only_affine_factored_teacher" if affine_factor
                else "prepare_DHR_image_only_teacher",
        "control_side": side,
        "field_path": str(field_path),
        "moving_size_xy": list(moving_size),
        "fixed_size_xy": list(fixed_size),
        "target_nonpositive_corners": validity["nonpositive_corners"],
        "target_boundary_max_error": validity["boundary_max_error"],
        "target_corner_min": validity["corner_min"],
        "target_out_of_unit_vertices": int(((target < 0) | (target > 1)).any(-1).sum()),
        "target_interior_vector_rmse_from_identity": float(
            (target[:, 1:-1, 1:-1] - reference[:, 1:-1, 1:-1])
            .square().sum(-1).mean().sqrt()
        ),
        "residual_boundary_max_error": validate_q1_map(residual, reference)["boundary_max_error"],
        "residual_interior_vector_rmse_from_identity": float(
            (residual[:, 1:-1, 1:-1] - reference[:, 1:-1, 1:-1])
            .square().sum(-1).mean().sqrt()
        ),
        "post_affine_matrix": affine_matrix.tolist() if affine_matrix is not None else None,
        "post_affine_offset": affine_offset.tolist() if affine_offset is not None else None,
        "post_affine_det": float(torch.linalg.det(affine_matrix)) if affine_matrix is not None else None,
        "output_path": str(output_path),
    }
    return result


def fit_target_vertices(
    teacher: torch.Tensor, *, steps: int, learning_rate: float, device: str,
) -> tuple[torch.Tensor, dict]:
    """Optimize all latent fields against target vertices; no image/landmark I/O."""
    if teacher.ndim != 4 or teacher.shape[0] != 1 or teacher.shape[-1] != 2 or (
        teacher.shape[1] != teacher.shape[2]
    ):
        raise ValueError("expected teacher shape (1,N,N,2)")
    if not bool(torch.isfinite(teacher).all()) or steps < 1 or learning_rate <= 0:
        raise ValueError("finite teacher, positive steps and learning rate required")
    target_device = torch.device(device)
    teacher = teacher.to(device=target_device, dtype=torch.float32)
    side = teacher.shape[1]
    decoder = HybridPatchSeedVertexQ1Pyramid(17, side, patch_cells=4).to(target_device)
    seed = torch.nn.ParameterList(
        torch.nn.Parameter(torch.zeros((1, 15, 15, 2), device=target_device))
        for _ in range(decoder.seed_passes)
    )
    levels = torch.nn.ParameterList(
        torch.nn.Parameter(torch.zeros((1, level - 2, level - 2, 2), device=target_device))
        for level in decoder.level_sides
    )
    optimizer = torch.optim.Adam((*seed, *levels), lr=learning_rate)
    reference = identity_vertices(side, device=target_device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    best_loss = float("inf")
    best_map = None
    first_loss = None
    last_loss = None
    durations = []
    finite_gradient_steps = 0
    interior_teacher = teacher[:, 1:-1, 1:-1]
    for _ in range(steps):
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = decoder(tuple(seed), tuple(levels))
        loss = (mapped[:, 1:-1, 1:-1] - interior_teacher).square().sum(-1).mean()
        loss.backward()
        gradients = [parameter.grad for parameter in (*seed, *levels)]
        finite = all(gradient is not None and bool(torch.isfinite(gradient).all())
                     for gradient in gradients)
        finite_gradient_steps += int(finite)
        if not finite:
            raise FloatingPointError("nonfinite or missing latent gradient")
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        if value < best_loss:
            best_loss = value
            best_map = mapped.detach().clone()
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - started)
    assert best_map is not None and first_loss is not None and last_loss is not None
    validity = validate_q1_map(best_map, reference)
    return best_map, {
        "mode": "H_DHR_teacher_to_safe_Q1_latents",
        "control_side": side,
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_interior_vertex_rmse": first_loss ** .5,
        "last_interior_vertex_rmse": last_loss ** .5,
        "best_interior_vertex_rmse": best_loss ** .5,
        "median_complete_step_seconds": statistics.median(durations),
        "finite_gradient_steps": finite_gradient_steps,
        "nonpositive_corners": validity["nonpositive_corners"],
        "corner_min": validity["corner_min"],
        "boundary_max_error": validity["boundary_max_error"],
        "boundary_ordered_rectangle": validity["boundary_ordered_rectangle"],
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
    }


def _saved_output_summary(certificate: dict, affine_matrix: np.ndarray | None) -> dict:
    """State exactly what the basic stored-binary certificate has checked."""
    determinant = None
    if affine_matrix is not None:
        determinant = float(affine_matrix[0, 0]) * float(affine_matrix[1, 1]) - (
            float(affine_matrix[0, 1]) * float(affine_matrix[1, 0])
        )
    return {
        "saved_binary_residual_valid": bool(certificate["valid"]),
        "saved_binary_nonpositive_corners": int(certificate["nonpositive_corners"]),
        "saved_object_is_affine_postcomposition": affine_matrix is not None,
        "post_affine_det": determinant,
    }


def _load_target_archive(path: Path) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]:
    with np.load(path) as archive:
        teacher = torch.from_numpy(archive["teacher_vertices"])
        has_matrix = "post_affine_matrix" in archive
        has_offset = "post_affine_offset" in archive
        if has_matrix != has_offset:
            raise ValueError("post-affine matrix and offset must be stored together")
        affine_matrix = archive["post_affine_matrix"] if has_matrix else None
        affine_offset = archive["post_affine_offset"] if has_offset else None
    if affine_matrix is not None:
        if affine_matrix.shape != (2, 2) or affine_offset.shape != (2,) or (
            not np.all(np.isfinite(affine_matrix)) or not np.all(np.isfinite(affine_offset))
        ):
            raise ValueError("post-affine data must be finite 2x2 and 2-vector")
        determinant = float(affine_matrix[0, 0]) * float(affine_matrix[1, 1]) - (
            float(affine_matrix[0, 1]) * float(affine_matrix[1, 0])
        )
        if determinant <= 0:
            raise ValueError("post-affine matrix requires positive orientation")
    return teacher, affine_matrix, affine_offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("field", "params", "moving_image", "fixed_image", "output_target"):
        preparation.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    preparation.add_argument("--side", type=int, default=257)
    preparation.add_argument("--affine-factor", action="store_true")
    fitting = commands.add_parser("fit")
    fitting.add_argument("--target", type=Path, required=True)
    fitting.add_argument("--output-map", type=Path, required=True)
    fitting.add_argument("--steps", type=int, default=200)
    fitting.add_argument("--learning-rate", type=float, default=.04)
    fitting.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_target(
            args.field, args.params, args.moving_image, args.fixed_image,
            args.output_target, side=args.side, affine_factor=args.affine_factor,
        )
    else:
        teacher, affine_matrix, affine_offset = _load_target_archive(args.target)
        mapped, result = fit_target_vertices(
            teacher, steps=args.steps, learning_rate=args.learning_rate,
            device=args.device,
        )
        reference = identity_vertices(mapped.shape[1], device=torch.device("cpu"))
        payload = {"vertices": mapped.cpu().numpy(),
                   "boundary_reference": reference.numpy()}
        if affine_matrix is not None and affine_offset is not None:
            payload.update({"post_affine_matrix": affine_matrix,
                            "post_affine_offset": affine_offset})
        np.savez_compressed(args.output_map, **payload)
        certificate = certify_q1_binary_map(args.output_map)
        result.update({
            "target": str(args.target), "saved_map": str(args.output_map),
        })
        result.update(_saved_output_summary(certificate, affine_matrix))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
