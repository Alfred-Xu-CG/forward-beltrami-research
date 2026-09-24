"""Test two independent/overlapping local low patches plus one fine packet."""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_matched_low_patch import (
    low_map_from_params, matched_low_patch,
    multistart_refine_low_params,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_joint_low_patch import multistart_joint_refine_two_patches
from phase7_test_unknown_carrier_bank import dataset as high_dataset
from phase7_train_mixed_scale_residual import MixedScaleResidualEncoder


def low_parameters(
    count: int, seed: int, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    rng = torch.Generator(device="cpu").manual_seed(seed + 100000)
    first_origin = .08 + .78 * torch.rand(count, 2, generator=rng)
    unrelated_origin = .08 + .78 * torch.rand(count, 2, generator=rng)
    close_offset = .03 * (2 * torch.rand(
        count, 2, generator=rng) - 1)
    second_origin = unrelated_origin.clone()
    second_origin[count // 2:] = (
        first_origin[count // 2:] + close_offset[count // 2:]
    ).clamp(.08, .86)
    magnitude = .0015 + .0015 * torch.rand(
        count, 2, generator=rng)
    angle = 2 * math.pi * torch.rand(
        count, 2, generator=rng)
    vector = torch.stack((
        magnitude * torch.cos(angle),
        magnitude * torch.sin(angle)), dim=-1)
    return (first_origin.to(device), second_origin.to(device),
            vector[:, 0].to(device), vector[:, 1].to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=57691)
    parser.add_argument("--joint-steps", type=int, default=4)
    parser.add_argument("--joint-damping", type=float, default=.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--image-checkpoint", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    _, moving_all, high_all, high_support_all, _, frequency_all = (
        high_dataset(args.count, args.seed, device, args.batch,
                     table, (80.5, 96.5, 112.5, 127.5)))
    parameters = low_parameters(args.count, args.seed, device)
    model = MixedScaleResidualEncoder(device, table).to(device).eval()
    zero_state = {key: value.clone() for key, value in
                  model.state_dict().items()}
    trained_states = {}
    for name, path in (
        ("joint_two_low_map_train", args.map_checkpoint),
        ("joint_two_low_image_train", args.image_checkpoint),
    ):
        if path is not None:
            trained_states[name] = torch.load(
                path, map_location=device,
                weights_only=False)["model"]
    identity257 = model.decoder.identity
    identity1025 = model.decoder.fine_identity
    names = ("identity", "oracle_two_low", "estimated_one_low",
             "greedy_two_low", "joint_two_low") + tuple(trained_states)
    accumulators = {
        group: {
            name: dict(map_sq=0., image=0., count=0,
                       min_j=math.inf, fallback=0,
                       frequency=0.) for name in names
        } for group in ("independent_origins", "close_origins")
    }
    target_minimum = math.inf
    estimates = dict(first_origin_error=0.,
                     second_origin_error=0.)
    close_joint_cases = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            model.load_state_dict(zero_state)
            stop = min(start + args.batch, args.count)
            batch = stop - start
            moving = moving_all[start:stop]
            high_target = high_all[start:stop]
            high_support = high_support_all[start:stop]
            frequency = frequency_all[start:stop]
            o1, o2, v1, v2 = (
                item[start:stop] for item in parameters)
            low1 = low_map_from_params(
                o1, v1, 1025, torch.float64)
            low2 = low_map_from_params(
                o2, v2, 1025, torch.float64)
            target = low1 + low2 + high_target - 2 * identity1025
            target_minimum = min(
                target_minimum, minimum_jacobian(target))
            query = table.interpolate(
                target.reshape(batch, -1, 2))
            fixed = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            oracle_low = (
                low_map_from_params(o1, v1, 257, torch.float64)
                + low_map_from_params(o2, v2, 257, torch.float64)
                - identity257)

            first_origin, first_vector, _ = matched_low_patch(
                fixed, moving)
            first_origin, first_vector, _ = (
                multistart_refine_low_params(
                    fixed, moving, first_origin,
                    first_vector, steps=4, directions=4))
            first_low = low_map_from_params(
                first_origin, first_vector, 257,
                torch.float64)
            first_image_map = low_map_from_params(
                first_origin, first_vector, 512,
                torch.float32)
            first_image = F.grid_sample(
                moving, 2 * first_image_map - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            pseudo_fixed = moving + (fixed - first_image)
            second_origin, second_vector, _ = matched_low_patch(
                pseudo_fixed, moving)
            second_origin, second_vector, _ = (
                multistart_refine_low_params(
                    pseudo_fixed, moving, second_origin,
                    second_vector, steps=4, directions=4))
            greedy_low = first_low + low_map_from_params(
                second_origin, second_vector, 257,
                torch.float64) - identity257
            joint_origin, joint_vector, _ = (
                multistart_joint_refine_two_patches(
                    fixed, moving,
                    torch.stack((first_origin, second_origin), dim=1),
                    torch.stack((first_vector, second_vector), dim=1),
                    steps=args.joint_steps,
                    damping=args.joint_damping))
            joint_low = (
                low_map_from_params(
                    joint_origin[:, 0], joint_vector[:, 0],
                    257, torch.float64) +
                low_map_from_params(
                    joint_origin[:, 1], joint_vector[:, 1],
                    257, torch.float64) - identity257)
            estimates["first_origin_error"] += float(
                torch.linalg.vector_norm(
                    first_origin - o1, dim=1).sum())
            estimates["second_origin_error"] += float(
                torch.linalg.vector_norm(
                    second_origin - o2, dim=1).sum())

            predictions = {"identity": (identity1025.expand(
                batch, -1, -1, -1), None)}
            for name, proposed in (
                ("oracle_two_low", oracle_low),
                ("estimated_one_low", first_low),
                ("greedy_two_low", greedy_low),
                ("joint_two_low", joint_low),
            ):
                predictions[name] = model(
                    fixed, moving, first_origin,
                    first_vector, proposed_low=proposed)
            for name, state in trained_states.items():
                model.load_state_dict(state)
                predictions[name] = model(
                    fixed, moving, first_origin,
                    first_vector, proposed_low=joint_low)
            for name, (output, predicted_frequency) in predictions.items():
                squared = (output - target).square().sum(dim=-1)
                mapped_query = table.interpolate(
                    output.reshape(batch, -1, 2))
                image = F.grid_sample(
                    moving, 2 * mapped_query.float() - 1,
                    mode="bilinear", padding_mode="border",
                    align_corners=True)
                image_mse = (image - fixed).square().mean(
                    dim=(1, 2, 3))
                jacobian = minimum_jacobian(output)
                fallback = torch.all(
                    output == identity1025,
                    dim=(1, 2, 3))
                if name == "joint_two_low":
                    for local_index in range(batch):
                        global_index = start + local_index
                        if global_index < args.count // 2:
                            continue
                        close_joint_cases.append(dict(
                            index=global_index,
                            map_rmse=float(squared[
                                local_index].mean().sqrt()),
                            image_mse=float(image_mse[local_index]),
                            true_origins=[
                                o1[local_index].tolist(),
                                o2[local_index].tolist()],
                            true_vectors=[
                                v1[local_index].tolist(),
                                v2[local_index].tolist()],
                            estimated_origins=joint_origin[
                                local_index].tolist(),
                            estimated_vectors=joint_vector[
                                local_index].tolist(),
                        ))
                for group, mask in (
                    ("independent_origins", torch.arange(
                        start, stop, device=device) < args.count // 2),
                    ("close_origins", torch.arange(
                        start, stop, device=device) >= args.count // 2),
                ):
                    if not mask.any():
                        continue
                    stats = accumulators[group][name]
                    stats["map_sq"] += float(squared[mask].mean(
                        dim=(1, 2)).sum())
                    stats["image"] += float(image_mse[mask].sum())
                    stats["count"] += int(mask.sum())
                    stats["min_j"] = min(stats["min_j"], jacobian)
                    stats["fallback"] += int(fallback[mask].sum())
                    if predicted_frequency is not None:
                        stats["frequency"] += float((
                            predicted_frequency[mask]
                            - frequency[mask]).abs().sum())
    for group, items in accumulators.items():
        for name, stats in items.items():
            count = stats["count"]
            print(json.dumps(dict(
                experiment="phase7_two_low_patch_stack",
                group=group, method=name,
                seed=args.seed, count=count,
                joint_steps=args.joint_steps,
                joint_damping=args.joint_damping,
                control_vertices=1025 ** 2,
                image_side=512,
                target_minimum_jacobian=target_minimum,
                map_vector_rmse=math.sqrt(stats["map_sq"] / count),
                image_mse=stats["image"] / count,
                minimum_jacobian=stats["min_j"],
                identity_outputs=stats["fallback"],
                frequency_mae=(stats["frequency"] / count
                               if name != "identity" else None)),
                sort_keys=True), flush=True)
    print(json.dumps(dict(
        diagnostic="raw_ordered_origin_distance",
        first_mean=estimates["first_origin_error"] / args.count,
        second_mean=estimates["second_origin_error"] / args.count,
        note="unordered patches can swap; distances are not matching error")),
        flush=True)
    print(json.dumps(dict(
        diagnostic="worst_close_joint_cases",
        cases=sorted(close_joint_cases,
                     key=lambda row: row["map_rmse"],
                     reverse=True)[:8])), flush=True)


if __name__ == "__main__":
    main()
