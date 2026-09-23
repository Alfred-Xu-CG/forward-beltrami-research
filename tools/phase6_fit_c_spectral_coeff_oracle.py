"""Target-coordinate capacity test for 18 Fourier edge conductance latents."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from phase6_local_conductance_ratio_bound import target_vertices
from phase6_spectral_edge_encoder import SpectralEdgeImageEncoder
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--item", type=int, default=29)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    side = args.side
    device = torch.device(args.device)
    mesh = structured_rectangle(side - 1, side - 1)
    grid = np.arange(side**2, dtype=np.int64).reshape(side, side)
    edges = np.concatenate((
        np.stack((grid[:, :-1].ravel(), grid[:, 1:].ravel()), axis=1),
        np.stack((grid[:-1].ravel(), grid[1:].ravel()), axis=1),
        np.stack((grid[:-1, :-1].ravel(), grid[1:, 1:].ravel()), axis=1),
    ))
    dictionary = SpectralEdgeImageEncoder(
        side, mesh.vertices[edges].mean(axis=1),
        frequencies=(1, 2, 4, 8, 16, 32),
    ).to(device)
    solver = SinePreconditionedTutteLayer(
        side, maximum_conductance=16, tolerance=1e-10, max_iterations=120,
    ).to(device)
    target, target_coefficients = target_vertices(side, args.item, device)
    target = target[None]
    coefficient = torch.nn.Parameter(
        torch.zeros(3, len(dictionary.frequencies), device=device, dtype=torch.float64))
    bases = (
        dictionary.horizontal_basis.double(), dictionary.vertical_basis.double(),
        dictionary.diagonal_basis.double(),
    )
    optimizer = torch.optim.Adam([coefficient], lr=args.learning_rate)
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def forward():
        fields = tuple(
            (-0.8 + coefficient[index] @ basis)[None].reshape(shape)
            for index, (basis, shape) in enumerate(zip(
                bases, ((1, side, side - 1), (1, side - 1, side),
                        (1, side - 1, side - 1))))
        )
        return solver(*fields)
    records = []
    synchronize()
    began = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        mapped = forward()
        difference = mapped - target
        coordinate_mse = difference.square().mean()
        gradient_mse = 0.5 * (
            ((side - 1) * (difference[:, :, 1:] - difference[:, :, :-1])).square().mean()
            + ((side - 1) * (difference[:, 1:] - difference[:, :-1])).square().mean()
        )
        loss = coordinate_mse + args.gradient_weight * gradient_mse
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            synchronize()
            records.append({
                "step": step + 1, "coordinate_mse_before_update": float(coordinate_mse),
                "gradient_mse_before_update": float(gradient_mse),
                "objective_before_update": float(loss),
                "elapsed_seconds": time.perf_counter() - began,
            })
    synchronize()
    training_seconds = time.perf_counter() - began
    with torch.no_grad():
        mapped = forward()
        error = mapped - target
        xx, yy = solver._source[..., 0], solver._source[..., 1]
        high = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
        projection = ((mapped - solver._source[None]) * high[None, :, :, None]).sum(
            dim=(1, 2)) / high.square().sum()
        conductances = tuple(
            1 + 15 * (-0.8 + coefficient[index] @ basis).sigmoid()
            for index, basis in enumerate(bases)
        )
    print(json.dumps({
        "method": "C_spectral_edge_latent_target_coordinate_oracle",
        "target_geometry_used_in_loss": True,
        "side": side, "control_vertices": side**2,
        "control_faces": 2 * (side - 1)**2,
        "target_item": args.item, "target_seed": 55101, "target_set_size": 32,
        "target_coefficients": target_coefficients,
        "spectral_frequencies": list(dictionary.frequencies),
        "latent_scalar_count": coefficient.numel(),
        "steps": args.steps, "learning_rate": args.learning_rate,
        "gradient_weight": args.gradient_weight,
        "device": str(device), "dtype": "float64",
        "vertex_map_rmse": float(error.square().mean().sqrt()),
        "maximum_vertex_error": float(torch.linalg.vector_norm(error, dim=-1).max()),
        "projected_fine_x_amplitude": float(projection[0, 0]),
        "projected_fine_y_amplitude": float(projection[0, 1]),
        "conductance_minimums": [float(value.min()) for value in conductances],
        "conductance_maximums": [float(value.max()) for value in conductances],
        "solver_stats": solver.last_forward_stats,
        "learned_coefficients": coefficient.detach().cpu().tolist(),
        "training_seconds": training_seconds,
        "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
