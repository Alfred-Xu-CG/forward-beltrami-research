"""Independent local affine (rotation/shear) target with safe P1 decoder."""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_local_affine_inference import (
    affine_window_map, energy_window_origin,
    fit_affine_window,
)
from phase7_matched_low_patch import (
    low_map_from_params, matched_low_patch,
    multistart_refine_low_params,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_test_unknown_carrier_bank import dataset as high_dataset
from phase7_train_mixed_scale_residual import MixedScaleResidualEncoder


def target_parameters(
    count: int, seed: int, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    rng = torch.Generator(device="cpu").manual_seed(seed + 200000)
    origin = .08 + .78 * torch.rand(count, 2, generator=rng)
    magnitude = .001 + .001 * torch.rand(count, generator=rng)
    angle = 2 * math.pi * torch.rand(count, generator=rng)
    vector = torch.stack((
        magnitude * torch.cos(angle),
        magnitude * torch.sin(angle)), dim=-1)
    raw_matrix = torch.randn(count, 2, 2, generator=rng)
    sigma = torch.linalg.matrix_norm(raw_matrix, ord=2)
    target_norm = .03 + .05 * torch.rand(count, generator=rng)
    matrix = raw_matrix * (
        target_norm / sigma.clamp_min(1e-12))[:, None, None]
    return (origin.to(device), vector.to(device),
            matrix.to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42691)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--affine-steps", type=int, default=8)
    parser.add_argument("--affine-damping", type=float, default=.001)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--image-checkpoint", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    _, moving_all, high_all, _, _, frequency_all = high_dataset(
        args.count, args.seed, device, args.batch, table,
        (80.5, 96.5, 112.5, 127.5))
    origin_all, vector_all, matrix_all = target_parameters(
        args.count, args.seed, device)
    model = MixedScaleResidualEncoder(device, table).to(device).eval()
    zero_state = {key: value.clone() for key, value in
                  model.state_dict().items()}
    trained_states = {}
    for name, path in (
        ("affine8_map_train", args.map_checkpoint),
        ("affine8_image_train", args.image_checkpoint),
    ):
        if path is not None:
            trained_states[name] = torch.load(
                path, map_location=device,
                weights_only=False)["model"]
    identity257 = model.decoder.identity
    identity1025 = model.decoder.fine_identity
    names = ("identity", "oracle_affine", "translation_fit",
             "true_origin_affine6", "estimated_origin_affine6",
             "estimated_origin_affine8", "energy_origin_affine8",
             "best_of_two_affine8") + tuple(trained_states)
    sums = {name: dict(map=0., image=0., min_j=math.inf,
                       fallback=0, frequency=0.) for name in names}
    target_minimum = math.inf
    origin_error = dict(initial=0., fitted=0.,
                        energy=0., best=0.)
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            model.load_state_dict(zero_state)
            stop = min(start + args.batch, args.count)
            batch = stop - start
            moving = moving_all[start:stop]
            high_target = high_all[start:stop]
            frequency = frequency_all[start:stop]
            origin, vector, matrix = (
                item[start:stop] for item in
                (origin_all, vector_all, matrix_all))
            low_target = affine_window_map(
                origin, vector, matrix, 1025,
                torch.float64)
            target = low_target + high_target - identity1025
            target_minimum = min(
                target_minimum, minimum_jacobian(target))
            query = table.interpolate(
                target.reshape(batch, -1, 2))
            fixed = F.grid_sample(
                moving, 2 * query.float() - 1,
                mode="bilinear", padding_mode="border",
                align_corners=True)
            estimated_origin, estimated_vector, _ = (
                matched_low_patch(fixed, moving))
            estimated_origin, estimated_vector, _ = (
                multistart_refine_low_params(
                    fixed, moving, estimated_origin,
                    estimated_vector, steps=4, directions=4))
            p_known, v_known, m_known, _ = fit_affine_window(
                fixed, moving, origin, estimated_vector,
                steps=args.affine_steps, moving_origin=False,
                damping=args.affine_damping)
            p_fixed, v_fixed, m_fixed, _ = fit_affine_window(
                fixed, moving, estimated_origin,
                estimated_vector, steps=args.affine_steps,
                moving_origin=False, damping=args.affine_damping)
            p_joint, v_joint, m_joint, joint_loss = fit_affine_window(
                fixed, moving, p_fixed,
                v_fixed, m_fixed, steps=args.affine_steps,
                moving_origin=True, damping=args.affine_damping)
            energy_origin = energy_window_origin(fixed, moving)
            p_energy0, v_energy0, m_energy0, _ = fit_affine_window(
                fixed, moving, energy_origin,
                estimated_vector, steps=args.affine_steps,
                moving_origin=False, damping=args.affine_damping)
            p_energy, v_energy, m_energy, energy_loss = (
                fit_affine_window(
                    fixed, moving, p_energy0, v_energy0, m_energy0,
                    steps=args.affine_steps, moving_origin=True,
                    damping=args.affine_damping))
            use_energy = energy_loss < joint_loss
            p_best = torch.where(use_energy[:, None], p_energy, p_joint)
            v_best = torch.where(use_energy[:, None], v_energy, v_joint)
            m_best = torch.where(
                use_energy[:, None, None], m_energy, m_joint)
            origin_error["initial"] += float(
                torch.linalg.vector_norm(
                    estimated_origin - origin, dim=1).sum())
            origin_error["fitted"] += float(
                torch.linalg.vector_norm(
                    p_joint - origin, dim=1).sum())
            origin_error["energy"] += float(
                torch.linalg.vector_norm(
                    energy_origin - origin, dim=1).sum())
            origin_error["best"] += float(
                torch.linalg.vector_norm(
                    p_best - origin, dim=1).sum())
            proposals = (
                ("oracle_affine", affine_window_map(
                    origin, vector, matrix, 257, torch.float64)),
                ("translation_fit", low_map_from_params(
                    estimated_origin, estimated_vector,
                    257, torch.float64)),
                ("true_origin_affine6", affine_window_map(
                    p_known, v_known, m_known, 257, torch.float64)),
                ("estimated_origin_affine6", affine_window_map(
                    p_fixed, v_fixed, m_fixed, 257, torch.float64)),
                ("estimated_origin_affine8", affine_window_map(
                    p_joint, v_joint, m_joint, 257, torch.float64)),
                ("energy_origin_affine8", affine_window_map(
                    p_energy, v_energy, m_energy, 257, torch.float64)),
                ("best_of_two_affine8", affine_window_map(
                    p_best, v_best, m_best, 257, torch.float64)),
            )
            predictions = {"identity": (
                identity1025.expand(batch, -1, -1, -1), None)}
            for name, proposed in proposals:
                predictions[name] = model(
                    fixed, moving, estimated_origin,
                    estimated_vector, proposed_low=proposed)
            for name, state in trained_states.items():
                model.load_state_dict(state)
                predictions[name] = model(
                    fixed, moving, estimated_origin,
                    estimated_vector, proposed_low=proposals[-1][1])
            for name, (output, predicted_frequency) in predictions.items():
                stats = sums[name]
                square = (output - target).square().sum(dim=-1)
                stats["map"] += float(square.mean(
                    dim=(1, 2)).sum())
                output_query = table.interpolate(
                    output.reshape(batch, -1, 2))
                image = F.grid_sample(
                    moving, 2 * output_query.float() - 1,
                    mode="bilinear", padding_mode="border",
                    align_corners=True)
                stats["image"] += float((
                    image - fixed).square().mean(
                        dim=(1, 2, 3)).sum())
                stats["min_j"] = min(
                    stats["min_j"], minimum_jacobian(output))
                stats["fallback"] += int(torch.all(
                    output == identity1025,
                    dim=(1, 2, 3)).sum())
                if predicted_frequency is not None:
                    stats["frequency"] += float((
                        predicted_frequency - frequency).abs().sum())
    for name, stats in sums.items():
        print(json.dumps(dict(
            experiment="phase7_local_affine_packet",
            method=name, seed=args.seed,
            count=args.count,
            affine_steps=args.affine_steps,
            affine_damping=args.affine_damping,
            image_side=512,
            control_vertices=1025 ** 2,
            control_faces=2 * 1024 ** 2,
            target_minimum_jacobian=target_minimum,
            map_vector_rmse=math.sqrt(
                stats["map"] / args.count),
            image_mse=stats["image"] / args.count,
            frequency_mae=(stats["frequency"] / args.count
                           if name != "identity" else None),
            minimum_jacobian=stats["min_j"],
            identity_outputs=stats["fallback"]),
            sort_keys=True), flush=True)
    print(json.dumps(dict(
        diagnostic="local_affine_origin_error",
        initial_mean=origin_error["initial"] / args.count,
        refined_mean=origin_error["fitted"] / args.count,
        energy_initial_mean=origin_error["energy"] / args.count,
        best_refined_mean=origin_error["best"] / args.count,
    )), flush=True)


if __name__ == "__main__":
    main()
