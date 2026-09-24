"""Inspect one overlapping-patch outlier without changing the decoder."""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import physical_image_gradient
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_matched_low_patch import (
    low_map_from_params, matched_low_patch,
    multistart_refine_low_params,
)
from phase7_joint_low_patch import multistart_joint_refine_two_patches
from phase7_test_two_low_patch_stack import low_parameters
from phase7_test_unknown_carrier_bank import dataset as high_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=57691)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--index", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--damping", type=float, default=.05)
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    _, moving_all, high_all, _, _, _ = high_dataset(
        args.count, args.seed, device, 4, table,
        (80.5, 96.5, 112.5, 127.5))
    o1_all, o2_all, v1_all, v2_all = low_parameters(
        args.count, args.seed, device)
    index = args.index
    moving = moving_all[index:index + 1]
    high_target = high_all[index:index + 1]
    o1, o2, v1, v2 = (
        item[index:index + 1] for item in
        (o1_all, o2_all, v1_all, v2_all))
    identity1025 = low_map_from_params(
        o1, torch.zeros_like(v1), 1025, torch.float64)
    true_low = (
        low_map_from_params(o1, v1, 1025, torch.float64) +
        low_map_from_params(o2, v2, 1025, torch.float64) -
        identity1025)
    true_full = true_low + high_target - identity1025
    query = table.interpolate(true_full.reshape(1, -1, 2))
    fixed = F.grid_sample(
        moving, 2 * query.float() - 1,
        mode="bilinear", padding_mode="border",
        align_corners=True)
    first_origin, first_vector, _ = matched_low_patch(fixed, moving)
    first_origin, first_vector, _ = multistart_refine_low_params(
        fixed, moving, first_origin, first_vector,
        steps=4, directions=4)
    first_image_map = low_map_from_params(
        first_origin, first_vector, 512, torch.float32)
    first_image = F.grid_sample(
        moving, 2 * first_image_map - 1,
        mode="bilinear", padding_mode="border",
        align_corners=True)
    pseudo_fixed = moving + (fixed - first_image)
    second_origin, second_vector, _ = matched_low_patch(
        pseudo_fixed, moving)
    second_origin, second_vector, _ = multistart_refine_low_params(
        pseudo_fixed, moving, second_origin, second_vector,
        steps=4, directions=4)
    estimated_origin, estimated_vector, _ = (
        multistart_joint_refine_two_patches(
            fixed, moving,
            torch.stack((first_origin, second_origin), dim=1),
            torch.stack((first_vector, second_vector), dim=1),
            steps=8, damping=args.damping))
    estimated_low = (
        low_map_from_params(
            estimated_origin[:, 0], estimated_vector[:, 0],
            1025, torch.float64) +
        low_map_from_params(
            estimated_origin[:, 1], estimated_vector[:, 1],
            1025, torch.float64) - identity1025)

    def image_loss(mapping: torch.Tensor) -> float:
        q = table.interpolate(mapping.reshape(1, -1, 2))
        image = F.grid_sample(
            moving, 2 * q.float() - 1,
            mode="bilinear", padding_mode="border",
            align_corners=True)
        return float((image - fixed).square().mean())

    axis = torch.arange(
        512, device=device, dtype=torch.float32) / 511
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    gradient_image = physical_image_gradient(moving)
    low_query = table.interpolate(true_low.reshape(1, -1, 2))
    gradient = F.grid_sample(
        gradient_image, 2 * low_query.float() - 1,
        mode="bilinear", padding_mode="border",
        align_corners=True)
    gx, gy = gradient[0, 0], gradient[0, 1]
    columns = []
    for origin in (o1, o2):
        tx = (xx - origin[0, 0]) * 16
        ty = (yy - origin[0, 1]) * 16
        window = (
            torch.where((tx >= 0) & (tx <= 1),
                        torch.sin(math.pi * tx).square(), 0) *
            torch.where((ty >= 0) & (ty <= 1),
                        torch.sin(math.pi * ty).square(), 0))
        columns.extend((window * gx, window * gy))
    design = torch.stack(columns, dim=0).flatten(1)
    gram = design @ design.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    output = dict(
        experiment="phase7_two_patch_ambiguity",
        index=index, seed=args.seed,
        damping=args.damping,
        true_origins=torch.stack((o1, o2), dim=1)[0].tolist(),
        estimated_origins=estimated_origin[0].tolist(),
        true_vectors=torch.stack((v1, v2), dim=1)[0].tolist(),
        estimated_vectors=estimated_vector[0].tolist(),
        low_map_vector_rmse=float((
            estimated_low - true_low).square().sum(
                dim=-1).mean().sqrt()),
        true_low_only_image_mse=image_loss(true_low),
        estimated_low_only_image_mse=image_loss(estimated_low),
        true_full_image_mse=image_loss(true_full),
        eigenvalues=eigenvalues.tolist(),
        smallest_eigenvector=eigenvectors[:, 0].tolist(),
        gram_condition_number=float(
            eigenvalues[-1] / eigenvalues[0]),
    )
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
