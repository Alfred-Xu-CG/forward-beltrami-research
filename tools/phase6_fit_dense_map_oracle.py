"""Direct-latent expressivity oracle on the same fine-grid target.

This is deliberately NOT image-to-latent training: the analytic map supervises
the latent variables, measuring representational/optimization limits apart
from texture ambiguity and encoder capacity.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import torch

from phase6_train_image_to_latent import _minimum_area_ratio
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    DenseMonotoneGridLayer,
    CoarseFineConvexQuadComposition,
    ExactAlternatingMonotoneComposition,
    HierarchicalConvexQuadFreeCenterLayer,
    SafeColoredVertexRelaxation,
    HierarchicalConvexQuadLayer,
    LocalPatchComposition,
    LocalPatchMonotoneLayer,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def target_map(side: int, device: torch.device, dtype: torch.dtype, target_kind: str = "base") -> torch.Tensor:
    """Smooth global motion plus a safe localized high-frequency component.

    Both variants have a globally subunit displacement Lipschitz bound and
    fix the square boundary. In the base variant, the loose two-term bound is
    below 0.821; in high32 it is below 0.895. Identity plus either perturbation
    is therefore injective and maps the square onto itself.
    """
    line = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    low = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    if target_kind == "base":
        high = torch.sin(16 * math.pi * xx) * torch.sin(16 * math.pi * yy)
        ax, ay, af = 0.03, 0.05, 0.003
    elif target_kind == "high32":
        high = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
        ax, ay, af = 0.015, 0.025, 0.0025
    else:
        raise ValueError("target_kind must be base or high32")
    return torch.stack((xx + ax * low + af * high, yy + ay * low + af * high), dim=-1)[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("single_vertical", "single_horizontal", "alternating", "patches", "convex_quad", "convex_quad_free", "convex_quad_local", "convex_quad_radial", "convex_quad_coarse_fine"), required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--coarse-side", type=int, default=17)
    parser.add_argument("--train-factor", choices=("both", "coarse", "fine"), default="both")
    parser.add_argument("--layers", type=int, choices=(2, 4), default=2)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-kind", choices=("base", "high32"), default="base")
    parser.add_argument("--raw-span", type=float, default=2.0, help="only for convex_quad_radial")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-state", default=None, help="warm-start latent tensors; Adam state restarts")
    args = parser.parse_args()
    if args.train_factor != "both" and args.method != "convex_quad_coarse_fine":
        raise ValueError("train-factor applies only to convex_quad_coarse_fine")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    dtype = torch.float32
    side = args.side
    mesh = structured_rectangle(side - 1, side - 1)
    target = target_map(side, device, dtype, args.target_kind)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=side, width=side)
    table.prepare(device=device, dtype=dtype)
    if args.method.startswith("single_"):
        axis = args.method.removeprefix("single_")
        decoder = DenseMonotoneGridLayer(side, axis=axis)
        global_logits = torch.nn.Parameter(torch.zeros(1, side - 1, dtype=dtype, device=device))
        line_logits = torch.nn.Parameter(torch.zeros(1, side, side - 1, dtype=dtype, device=device))
        parameters = [global_logits, line_logits]

        def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            control = decoder(global_logits, line_logits)
            return control, (control,)

    elif args.method == "alternating":
        axes = tuple("vertical" if i % 2 == 0 else "horizontal" for i in range(args.layers))
        decoder = ExactAlternatingMonotoneComposition(side, table, axes)
        global_logits_list = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(1, side - 1, dtype=dtype, device=device)) for _ in axes)
        line_logits_list = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(1, side, side - 1, dtype=dtype, device=device)) for _ in axes)
        parameters = list(global_logits_list) + list(line_logits_list)

        def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            levels = tuple(((global_logits_list[i], line_logits_list[i]),) for i in range(args.layers))
            result = decoder(levels)
            return result.dense, result.controls

    elif args.method == "convex_quad":
        decoder = HierarchicalConvexQuadLayer(side)
        horizontal_logits_list = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, current, current - 1, dtype=dtype, device=device))
            for current in decoder.latent_sides
        )
        vertical_logits_list = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, current - 1, current, dtype=dtype, device=device))
            for current in decoder.latent_sides
        )
        parameters = list(horizontal_logits_list) + list(vertical_logits_list)

        def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            control = decoder(tuple(zip(horizontal_logits_list, vertical_logits_list)))
            return control, (control,)

    elif args.method == "convex_quad_coarse_fine":
        decoder = CoarseFineConvexQuadComposition(args.coarse_side, side, side)
        decoder.prepare(device=device, dtype=dtype)

        def make_factor_latents(factor: HierarchicalConvexQuadFreeCenterLayer):
            root = torch.nn.Parameter(torch.zeros(1, 1, 1, 2, dtype=dtype, device=device))
            horizontal = torch.nn.ParameterList(
                torch.nn.Parameter(torch.zeros(1, current, current - 1, dtype=dtype, device=device))
                for current in factor.latent_sides
            )
            vertical = torch.nn.ParameterList(
                torch.nn.Parameter(torch.zeros(1, current - 1, current, dtype=dtype, device=device))
                for current in factor.latent_sides
            )
            centers = torch.nn.ParameterList(
                torch.nn.Parameter(torch.zeros(1, current - 1, current - 1, 2, dtype=dtype, device=device))
                for current in factor.latent_sides
            )
            return root, horizontal, vertical, centers

        coarse_root, coarse_h, coarse_v, coarse_c = make_factor_latents(decoder.coarse)
        fine_root, fine_h, fine_v, fine_c = make_factor_latents(decoder.fine)
        parameters = [coarse_root, *coarse_h, *coarse_v, *coarse_c, fine_root, *fine_h, *fine_v, *fine_c]

        def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            result = decoder(
                coarse_root, tuple(zip(coarse_h, coarse_v, coarse_c)),
                fine_root, tuple(zip(fine_h, fine_v, fine_c)),
            )
            return result.dense, result.controls

    elif args.method in ("convex_quad_free", "convex_quad_local", "convex_quad_radial"):
        decoder = HierarchicalConvexQuadFreeCenterLayer(side)
        root_center = torch.nn.Parameter(torch.zeros(1, 1, 1, 2, dtype=dtype, device=device))
        horizontal_logits_list = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, current, current - 1, dtype=dtype, device=device))
            for current in decoder.latent_sides
        )
        vertical_logits_list = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, current - 1, current, dtype=dtype, device=device))
            for current in decoder.latent_sides
        )
        center_logits_list = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, current - 1, current - 1, 2, dtype=dtype, device=device))
            for current in decoder.latent_sides
        )
        parameters = [root_center] + list(horizontal_logits_list) + list(vertical_logits_list) + list(center_logits_list)
        relaxation = (
            SafeColoredVertexRelaxation(
                side, safety_fraction=0.85,
                motion_mode="radial" if args.method == "convex_quad_radial" else "disk",
                raw_span=args.raw_span,
            )
            if args.method in ("convex_quad_local", "convex_quad_radial") else None
        )
        local_logits = torch.nn.Parameter(torch.zeros(1, side - 2, side - 2, 2, dtype=dtype, device=device)) if relaxation else None
        if local_logits is not None:
            parameters.append(local_logits)

        def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            control = decoder(root_center, tuple(zip(horizontal_logits_list, vertical_logits_list, center_logits_list)))
            if relaxation is not None:
                control = relaxation(control, local_logits)
            return control, (control,)

    else:
        patch_layers = []
        for index in range(args.layers):
            offset = args.patch_cells // 2 if index % 2 else 0
            patch_layers.append(LocalPatchMonotoneLayer(side, args.patch_cells, offset_x=offset, offset_y=offset, axis="horizontal" if index % 2 else "vertical").to(device))
        decoder = LocalPatchComposition(patch_layers, table)
        latents = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(1, layer.patch_count, args.patch_cells - 1, args.patch_cells, dtype=dtype, device=device)) for layer in patch_layers)
        parameters = list(latents)

        def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            result = decoder(tuple(latents))
            return result.dense, result.controls

    if args.load_state is not None:
        previous = torch.load(args.load_state, map_location=device, weights_only=True)
        if previous["method"] != args.method or previous["side"] != side:
            raise ValueError("loaded latent checkpoint method/side mismatch")
        if args.method == "convex_quad_radial" and previous.get("raw_span", 2.0) != args.raw_span:
            raise ValueError("loaded latent checkpoint radial raw-span mismatch")
        if args.method == "convex_quad_coarse_fine" and previous.get("coarse_side") != args.coarse_side:
            raise ValueError("loaded latent checkpoint coarse-side mismatch")
        if len(previous["parameters"]) != len(parameters):
            raise ValueError("loaded latent checkpoint count mismatch")
        with torch.no_grad():
            for parameter, prior in zip(parameters, previous["parameters"]):
                if parameter.shape != prior.shape:
                    raise ValueError("loaded latent tensor shape mismatch")
                parameter.copy_(prior)
    if args.method == "convex_quad_coarse_fine":
        split = 1 + 3 * len(decoder.coarse.latent_sides)
        train_parameters = parameters[:split] if args.train_factor == "coarse" else parameters[split:] if args.train_factor == "fine" else parameters
    else:
        train_parameters = parameters
    active_ids = {id(parameter) for parameter in train_parameters}
    for parameter in parameters:
        parameter.requires_grad_(id(parameter) in active_ids)
    optimizer = torch.optim.Adam(train_parameters, lr=args.learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    samples = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        mapped, _ = decode()
        loss = (mapped - target).square().mean()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            samples.append({"step": step + 1, "map_rmse": math.sqrt(loss.item())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - begin
    with torch.no_grad():
        mapped, controls = decode()
        difference = mapped - target
        rmse = difference.square().mean().sqrt().item()
        maximum_error = torch.linalg.vector_norm(difference, dim=-1).max().item()
        areas = [_minimum_area_ratio(control) for control in controls]
    if args.save_state is not None:
        torch.save({
            "method": args.method,
            "raw_span": args.raw_span if args.method == "convex_quad_radial" else None,
            "target_kind": args.target_kind,
            "side": side,
            "layers": 1 if args.method.startswith("single_") or args.method in ("convex_quad", "convex_quad_free", "convex_quad_local", "convex_quad_radial") else 2 if args.method == "convex_quad_coarse_fine" else args.layers,
            "coarse_side": args.coarse_side if args.method == "convex_quad_coarse_fine" else None,
            "train_factor": args.train_factor if args.method == "convex_quad_coarse_fine" else None,
            "patch_cells": args.patch_cells if args.method == "patches" else None,
            "parameters": [parameter.detach().cpu().clone() for parameter in parameters],
        }, args.save_state)
    print(json.dumps({
        "task": "direct_latent_map_oracle_not_image_training",
        "target_kind": args.target_kind,
        "method": args.method,
        "raw_span": args.raw_span if args.method == "convex_quad_radial" else None,
        "representation": "original_grid_P1" if args.method.startswith("single_") or args.method in ("convex_quad", "convex_quad_free", "convex_quad_local", "convex_quad_radial") else "exact_PL_composition",
        "control_side": side,
        "control_vertices": mesh.n_vertices,
        "control_faces_per_layer": mesh.n_faces,
        "query_side": side,
        "query_count": side**2,
        "layers": 1 if args.method.startswith("single_") or args.method in ("convex_quad", "convex_quad_free", "convex_quad_local", "convex_quad_radial") else 2 if args.method == "convex_quad_coarse_fine" else args.layers,
        "coarse_side": args.coarse_side if args.method == "convex_quad_coarse_fine" else None,
        "train_factor": args.train_factor if args.method == "convex_quad_coarse_fine" else None,
        "patch_cells": args.patch_cells if args.method == "patches" else None,
        "latent_values": sum(parameter.numel() for parameter in parameters),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "dtype": "float32",
        "device": str(device),
        "loaded_state": args.load_state,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "training_seconds": training_seconds,
        "final_map_rmse": rmse,
        "maximum_pointwise_map_error": maximum_error,
        "minimum_layer_signed_area_ratios": areas,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "trajectory_samples": samples,
        "saved_state": args.save_state,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
