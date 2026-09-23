"""Independent full sparse solve and directional VJP at a declared mesh size."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

from phase6_benchmark_sine_pcg_tutte import make_logits
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer


def independent_direct(side: int, weights: tuple[np.ndarray, ...]) -> tuple[np.ndarray, float, float]:
    """Assemble the full Dirichlet system from edge endpoints, not PCG kernels."""
    horizontal, vertical, diagonal = weights
    ids = np.arange(side * side, dtype=np.int64).reshape(side, side)
    aa = np.concatenate((ids[:, :-1].ravel(), ids[:-1].ravel(), ids[:-1, :-1].ravel()))
    bb = np.concatenate((ids[:, 1:].ravel(), ids[1:].ravel(), ids[1:, 1:].ravel()))
    cc = np.concatenate((horizontal.ravel(), vertical.ravel(), diagonal.ravel()))
    n = side * side
    laplacian = sparse.coo_matrix((np.concatenate((cc, cc, -cc, -cc)),
                                   (np.concatenate((aa, bb, aa, bb)),
                                    np.concatenate((aa, bb, bb, aa)))), shape=(n, n)).tocsr()
    rows, columns = np.meshgrid(np.arange(1, side - 1), np.arange(1, side - 1), indexing="ij")
    interior = ids[rows, columns].ravel()
    mask = np.ones(n, dtype=bool); mask[interior] = False
    boundary = np.flatnonzero(mask)
    source = np.stack(np.meshgrid(np.arange(side)/(side-1), np.arange(side)/(side-1), indexing="xy"), axis=-1).reshape(n, 2)
    matrix = laplacian[interior][:, interior].tocsc()
    rhs = -(laplacian[interior][:, boundary] @ source[boundary])
    began = time.perf_counter()
    interior_map = sparse_linalg.spsolve(matrix, rhs)
    elapsed = time.perf_counter() - began
    mapped = source.copy(); mapped[interior] = interior_map
    residual = np.linalg.norm(matrix @ interior_map - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    return mapped.reshape(side, side, 2), residual, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--mode", choices=("smooth", "random"), default="smooth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = SinePreconditionedTutteLayer(args.side, tolerance=1e-11, max_iterations=100).to(device)
    logits = make_logits(args.side, 1, args.mode, device)
    mapped = layer(*logits)
    initial_stats = dict(layer.last_forward_stats)
    weights = tuple((1 + 3 * value.detach().sigmoid())[0].cpu().numpy() for value in logits)
    direct, direct_residual, direct_seconds = independent_direct(args.side, weights)
    maximum_discrepancy = float(np.abs(mapped.detach()[0].cpu().numpy() - direct).max())
    torch.manual_seed(20260923)
    cotangent = torch.randn_like(mapped) / math.sqrt(mapped.numel())
    gradients = torch.autograd.grad((mapped * cotangent).sum(), logits)
    backward_stats = dict(layer.last_backward_stats)
    directions = tuple(torch.randn_like(value) / math.sqrt(value.numel()) for value in logits)
    analytic = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions)).item()
    with torch.no_grad():
        plus = (layer(*(value + args.epsilon * direction for value, direction in zip(logits, directions))) * cotangent).sum()
        minus = (layer(*(value - args.epsilon * direction for value, direction in zip(logits, directions))) * cotangent).sum()
    numerical = ((plus - minus) / (2 * args.epsilon)).item()
    print(json.dumps({
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "mode": args.mode,
        "device": str(device),
        "dtype": "float64",
        "conductance_interval": [1, 4],
        "pcg_stats_before_directional_check": initial_stats,
        "pcg_adjoint_stats": backward_stats,
        "independent_sparse_direct_seconds": direct_seconds,
        "independent_sparse_direct_relative_residual": float(direct_residual),
        "maximum_pcg_vs_direct_coordinate_difference": maximum_discrepancy,
        "directional_vjp_analytic": analytic,
        "directional_vjp_central_difference": numerical,
        "directional_vjp_absolute_difference": abs(analytic - numerical),
        "directional_vjp_relative_difference": abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-15),
        "epsilon": args.epsilon,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
