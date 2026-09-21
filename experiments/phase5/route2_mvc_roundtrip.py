"""MVC canonical encode/decode benchmark on Route-I positive-Tutte maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import scipy
import scipy.sparse.linalg as sparse_linalg
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.mesh import structured_rectangle  # noqa: E402
from qcopt.neural_bijection.metrics import compute_p1_map_metrics  # noqa: E402
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.incremental import assemble_directed_system  # noqa: E402
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target  # noqa: E402
from qcopt.neural_bijection.tutte.mvc import MeanValueCoordinateEncoder  # noqa: E402


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip().lower()
    return value if result.returncode == 0 and len(value) == 40 else None


def _condition_estimate(matrix) -> float:
    factor = sparse_linalg.splu(matrix.tocsc())
    n = matrix.shape[0]
    inverse = sparse_linalg.LinearOperator(
        (n, n),
        matvec=lambda value: factor.solve(value),
        rmatvec=lambda value: factor.solve(value, trans="T"),
        matmat=lambda value: factor.solve(value),
        rmatmat=lambda value: factor.solve(value, trans="T"),
        dtype=np.float64,
    )
    return float(sparse_linalg.norm(matrix, ord=1) * sparse_linalg.onenormest(inverse))


def _minimum_triangle_quality(mesh, mapped: np.ndarray) -> float:
    triangles = mapped[mesh.faces]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    third = triangles[:, 2] - triangles[:, 1]
    twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    denominator = (
        np.sum(first * first, axis=1)
        + np.sum(second * second, axis=1)
        + np.sum(third * third, axis=1)
    )
    quality = 2.0 * np.sqrt(3.0) * twice_area / denominator
    return float(np.min(quality))


def run_case(*, control_side: int, seed: int, strength: float) -> dict[str, Any]:
    if control_side < 3:
        raise ValueError("control_side must be at least three")
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("strength must be finite and positive")
    mesh = structured_rectangle(control_side - 1, control_side - 1)
    target_start = perf_counter()
    target = build_directed_target(
        mesh,
        image_height=2,
        image_width=2,
        seed=seed,
        strength=strength,
        height=1.0,
    )
    target_seconds = perf_counter() - target_start
    solver = DirectTutteLayer(mesh)
    encoder = MeanValueCoordinateEncoder(mesh)
    mapped = target.control.detach().clone()

    encode_start = perf_counter()
    encoded = encoder(mapped)
    encode_seconds = perf_counter() - encode_start
    decode_start = perf_counter()
    recovered = solver(encoded.logits, encoded.boundary)
    decode_seconds = perf_counter() - decode_start

    mapped_numpy = mapped.detach().numpy()
    recovered_numpy = recovered.detach().numpy()
    metrics = compute_p1_map_metrics(mesh, recovered_numpy, target=mapped_numpy)
    mask = torch.as_tensor(solver.system.valid_mask)
    raw_probabilities = torch.softmax(
        target.interior_logits.masked_fill(~mask, -torch.inf), dim=-1
    )
    supported_logits = encoded.logits.masked_fill(~mask, torch.nan)
    maximums = torch.where(mask, encoded.logits, -torch.inf).amax(dim=-1)
    minimums = torch.where(mask, encoded.logits, torch.inf).amin(dim=-1)
    degree = mask.sum(dim=-1)
    row_means = (encoded.logits * mask).sum(dim=-1) / degree
    matrix, _, _ = assemble_directed_system(
        solver.system,
        encoded.logits.detach().numpy(),
        encoded.boundary.detach().numpy(),
    )
    return {
        "control_side": control_side,
        "control_vertices": mesh.n_vertices,
        "faces": len(mesh.faces),
        "interior_rows": solver.system.n_rows,
        "seed": seed,
        "raw_logit_strength": strength,
        "target_setup_seconds": target_seconds,
        "mvc_encode_seconds": encode_seconds,
        "canonical_decode_seconds": decode_seconds,
        "maximum_vertex_error": metrics.maximum_map_error,
        "map_rmse": metrics.map_rmse,
        "mu_rmse": metrics.mu_rmse,
        "maximum_mu_error": metrics.maximum_mu_error,
        "minimum_area_ratio": metrics.minimum_area_ratio,
        "minimum_signed_area": metrics.minimum_signed_area,
        "topology_certified": metrics.global_injectivity_certificate,
        "minimum_mvc_probability": float(encoded.probabilities[mask].min()),
        "maximum_canonical_logit_spread": float((maximums - minimums).max()),
        "maximum_canonical_supported_row_mean": float(torch.abs(row_means).max()),
        "maximum_barycentric_residual": float(encoded.diagnostics.barycentric_residual.max()),
        "maximum_covariance_condition": float(encoded.diagnostics.covariance_condition.max()),
        "minimum_edge_length": float(encoded.diagnostics.minimum_edge_length.min()),
        "minimum_angle_sine": float(encoded.diagnostics.minimum_angle_sine.min()),
        "maximum_winding_error": float(encoded.diagnostics.winding_error.max()),
        "minimum_triangle_quality": _minimum_triangle_quality(mesh, recovered_numpy),
        "canonical_system_condition_one_estimate": _condition_estimate(matrix),
        "raw_to_canonical_probability_l2": float(
            torch.linalg.vector_norm(raw_probabilities[mask] - encoded.probabilities[mask])
        ),
        "finite_supported_logits": bool(torch.isfinite(supported_logits[mask]).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-sides", default="11,25,49")
    parser.add_argument("--seeds", default="1701,1702,1703")
    parser.add_argument("--strengths", default="0.15,0.5,1.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sides = [int(value) for value in args.control_sides.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    strengths = [float(value) for value in args.strengths.split(",")]
    started = perf_counter()
    rows = [
        run_case(control_side=side, seed=seed, strength=strength)
        for side in sides
        for strength in strengths
        for seed in seeds
    ]
    receipt = {
        "schema": "phase5_route2_mvc_roundtrip_v1",
        "research_question": (
            "Does the differentiable MVC encoder canonically represent and exactly re-decode "
            "Route-I positive-Tutte maps at 11/25/49 control-side resolutions?"
        ),
        "metric_definitions": {
            "map_rmse": "sqrt(mean over vertices of squared Euclidean position error)",
            "triangle_quality": "2*sqrt(3)*twice_signed_area / sum of three squared edge lengths",
            "covariance_condition": "largest/smallest eigenvalue of the 2x2 MVC row covariance",
            "system_condition": "one-norm estimate ||A||_1 ||A^{-1}||_1 using SuperLU inverse products",
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git_commit": _git_commit(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "configuration": {
            "control_sides": sides,
            "seeds": seeds,
            "strengths": strengths,
            "dtype": "float64",
            "boundary_height": 1.0,
        },
        "wall_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

