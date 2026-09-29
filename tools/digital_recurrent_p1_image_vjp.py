"""Benchmark 257² recurrent map -> 512² P1 image loss -> head VJP."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from tools.digital_acrobat_recurrent_probe import _load_model
from tools.digital_acrobat_teacher_probe import load_case_inputs


def benchmark(root: Path, model_output: Path, output: Path, *,
              case: int, device: str, repeats: int = 3) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if repeats < 1:
        raise ValueError("positive repeats required")
    target_device = torch.device(device)
    model = _load_model(model_output, device).train()
    training = json.loads((model_output / "train_manifest.json").read_text())
    frozen_head_sides = training.get("initialized_head_sides", [])
    for side in frozen_head_sides:
        for parameter in model.heads[str(side)].parameters():
            parameter.requires_grad_(False)
    data = load_case_inputs(root, case, target_device)
    fixed, moving = data["fixed"], data["prewarped"]
    if fixed.shape != (1, 1, 512, 512) or moving.shape != fixed.shape:
        raise ValueError("one 512² image pair required")
    axis = (torch.arange(512, dtype=fixed.dtype, device=target_device) + .5) / 512
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    query = torch.stack((xx, yy), -1).reshape(-1, 2)
    table = StructuredDenseQueryTable.from_shape(256, 256, height=512, width=512)

    def one() -> tuple[torch.Tensor, torch.Tensor]:
        vertices, matrix, offset = model(fixed, moving)
        p1 = table.interpolate_points(vertices.reshape(1, -1, 2), query)
        positions = Q1ImageRegistrationNetwork.apply_affine(
            p1.reshape(1, 512, 512, 2), matrix, offset,
        )
        warped = F.grid_sample(moving, 2 * positions - 1,
                               padding_mode="border", align_corners=False)
        loss = (warped - fixed).square().mean()
        return loss, vertices

    def sync() -> None:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    rows = []
    gradients = {}
    maps = {}
    for checkpointed in (False, True):
        model.checkpoint_rounds = checkpointed
        samples = []
        for repeat in range(repeats + 1):
            model.zero_grad(set_to_none=True)
            if target_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target_device)
            sync()
            start = time.perf_counter()
            loss, vertices = one()
            sync()
            forward = time.perf_counter() - start
            loss.backward()
            sync()
            total = time.perf_counter() - start
            head = model.heads[str(vertices.shape[1])]
            head_gradients = [parameter.grad for parameter in head.parameters()]
            if any(gradient is None or not bool(torch.isfinite(gradient).all())
                   for gradient in head_gradients):
                raise FloatingPointError("missing or nonfinite 257-head gradient")
            family_grad_l1 = ({
                family: float(sum(parameter.grad.abs().sum()
                                  for parameter in subhead.parameters()))
                for family, subhead in head.items()
            } if isinstance(head, torch.nn.ModuleDict) else None)
            if repeat == repeats:
                key = "checkpoint" if checkpointed else "ordinary"
                gradients[key] = torch.cat([
                    gradient.detach().flatten().cpu().clone()
                    for gradient in head_gradients
                ])
                maps[key] = vertices.detach().cpu().clone()
            if repeat:
                samples.append({
                    "forward_seconds": forward,
                    "forward_plus_vjp_seconds": total,
                    "peak_torch_cuda_allocated_bytes": (
                        torch.cuda.max_memory_allocated(target_device)
                        if target_device.type == "cuda" else None
                    ),
                    "photometric_mse": float(loss.detach()),
                    "head_parameter_grad_l1": float(sum(
                        gradient.abs().sum() for gradient in head_gradients)),
                    "family_parameter_grad_l1": family_grad_l1,
                })
        rows.append({
            "checkpoint_rounds": checkpointed,
            "median_forward_seconds": statistics.median(s["forward_seconds"] for s in samples),
            "median_forward_plus_vjp_seconds": statistics.median(
                s["forward_plus_vjp_seconds"] for s in samples),
            "max_measured_peak_allocated_bytes": max(
                s["peak_torch_cuda_allocated_bytes"] for s in samples)
                if target_device.type == "cuda" else None,
            "samples": samples,
        })
    report = {
        "question": "P1 query plus image-sampling first-order VJP and checkpoint cost",
        "model_output": str(model_output), "case": case,
        "control_vertices_per_axis": 257,
        "image_and_query_pixels_per_axis": 512,
        "query": "fixed canvas pixel centers ((j+0.5)/512,(i+0.5)/512)",
        "loss": "mean (fixed brightness minus prewarped-moving brightness at internal-affine P1 map)^2",
        "external_initial_affine_already_prewarped_moving": True,
        "frozen_inherited_head_sides": frozen_head_sides,
        "all_257_head_parameter_gradients_checked": True,
        "warmups": 1, "timed_repeats": repeats,
        "device": device,
        "rows": rows,
        "ordinary_checkpoint_map_bitwise_equal": torch.equal(maps["ordinary"], maps["checkpoint"]),
        "maximum_absolute_257_head_parameter_gradient_difference": float(
            (gradients["ordinary"] - gradients["checkpoint"]).abs().amax()),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, default=168)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args.root, args.model_output, args.output,
                       case=args.case, device=args.device, repeats=args.repeats)
    print(json.dumps({key: result[key] for key in
                      ("ordinary_checkpoint_map_bitwise_equal",
                       "maximum_absolute_257_head_parameter_gradient_difference",
                       "rows")}, indent=2))


if __name__ == "__main__":
    main()
