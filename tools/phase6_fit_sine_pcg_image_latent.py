"""Per-pair fine-edge latent fitting from images only; not amortized inference."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer, evaluate_structured_p1_with_jacobian
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--item", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--image-gradient-weight", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    args = parser.parse_args()
    if not 0 <= args.item < 32 or args.steps < 1 or args.image_gradient_weight < 0:
        raise ValueError("item, steps, or image-gradient weight outside declared range")
    device = torch.device(args.device)
    fixed, moving, true_map, coefficients = (
        value[args.item:args.item + 1].to(device)
        for value in make_dataset(32, args.image_side, 55101, target_family="high32", return_coefficients=True)
    )
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    layer = SinePreconditionedTutteLayer(args.side, maximum_conductance=16.0,
                                          tolerance=1e-10, max_iterations=120).to(device)
    logits = [
        torch.nn.Parameter(torch.zeros(1, args.side, args.side - 1, device=device, dtype=torch.float64)),
        torch.nn.Parameter(torch.zeros(1, args.side - 1, args.side, device=device, dtype=torch.float64)),
        torch.nn.Parameter(torch.zeros(1, args.side - 1, args.side - 1, device=device, dtype=torch.float64)),
    ]
    optimizer = torch.optim.Adam(logits, lr=args.learning_rate)
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def evaluate():
        control = layer(*logits)
        query = table.interpolate(control.float().reshape(1, -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear", padding_mode="border", align_corners=True)
        residual = warped - fixed
        image_mse = residual.square().mean()
        gradient_mse = 0.5 * (
            (residual[..., 1:] - residual[..., :-1]).square().mean()
            + (residual[..., 1:, :] - residual[..., :-1, :]).square().mean()
        )
        return image_mse + args.image_gradient_weight * gradient_mse, image_mse, gradient_mse, query, control
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times, records = [], [], []
    began_all = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        total, image_mse, gradient_mse, _, _ = evaluate()
        synchronize(); middle = time.perf_counter()
        total.backward()
        synchronize(); after_backward = time.perf_counter()
        optimizer.step()
        forward_times.append(middle - began)
        backward_times.append(after_backward - middle)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            records.append({"step": step + 1, "image_mse_before_update": image_mse.item(),
                            "image_gradient_mse_before_update": gradient_mse.item(),
                            "objective_before_update": total.item(),
                            "forward_residual": layer.last_forward_stats["true_relative_residual"],
                            "minimum_area_ratio": layer.last_forward_stats["minimum_signed_area_ratio"],
                            "elapsed_seconds": time.perf_counter() - began_all})
    training_seconds = time.perf_counter() - began_all
    with torch.no_grad():
        total, image_mse, gradient_mse, query, control = evaluate()
        source_grid = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32).reshape(args.side, args.side, 2)
        high_basis = torch.sin(64 * math.pi * source_grid[..., 0]) * torch.sin(64 * math.pi * source_grid[..., 1])
        fine_basis_norm = high_basis.square().sum()
        projected = (
            ((control.float()[0] - source_grid) * high_basis[..., None]).sum(dim=(0, 1)) / fine_basis_norm
            if fine_basis_norm > 1e-10 else None
        )
        vertices = source_grid.reshape(-1, 2)
        faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
        centroids = vertices[faces].mean(dim=1)[None]
        _, predicted_jacobian = evaluate_structured_p1_with_jacobian(control.float(), centroids)
        _, target_jacobian = _target_on_faces(centroids, coefficients, 32)
        predicted_mu, target_mu = _mu(predicted_jacobian), _mu(target_jacobian)
        geometry = {
            "face_beltrami_rmse": (predicted_mu - target_mu).abs().square().mean().sqrt().item(),
            "maximum_predicted_beltrami_modulus": predicted_mu.abs().max().item(),
            "maximum_target_beltrami_modulus": target_mu.abs().max().item(),
        }
    if args.save_state:
        torch.save({"logits": [value.detach().cpu() for value in logits], "args": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "per_pair_bounded_sine_pcg_direct_image_latent",
        "not_amortized_inference": True,
        "no_target_geometry_in_loss": True,
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "target_set_size": 32,
        "target_seed": 55101,
        "target_item": args.item,
        "target_coefficients_for_evaluation_only": coefficients[0].tolist(),
        "conductance_interval": [1, 16],
        "image_gradient_weight": args.image_gradient_weight,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "device": str(device),
        "dtype": "float64_solver_float32_images",
        "final_image_mse": image_mse.item(),
        "final_image_gradient_mse": gradient_mse.item(),
        "final_objective": total.item(),
        "final_query_map_rmse": (query - true_map).square().mean().sqrt().item(),
        "projected_fine_x_amplitude": projected[0].item() if projected is not None else None,
        "projected_fine_y_amplitude": projected[1].item() if projected is not None else None,
        "final_geometry": geometry,
        "final_solver_stats": dict(layer.last_forward_stats),
        "median_forward_seconds_after_first": statistics.median(forward_times[1:] or forward_times),
        "median_backward_seconds_after_first": statistics.median(backward_times[1:] or backward_times),
        "training_seconds": training_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
