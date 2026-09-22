"""Explicit fixed-boundary comparison of three decoder-tangent lifts.

This is a deliberately tiny, dense diagnostic.  It constructs the complete
decoder Jacobian on a supported arithmetic-zero-row-mean logit basis and
compares its Euclidean Moore--Penrose lift with the MVC encoder derivative and
the covariance lift.  It is not a scalable layer implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

import numpy as np

# On the supported Windows research host, initializing NumPy's linear algebra
# runtime before importing Torch avoids loading two incompatible OpenMP
# runtimes when the SciPy SuperLU-backed reference decoder is first exercised.
_NUMPY_LINALG_BOOTSTRAP = float(np.linalg.cond(np.eye(1, dtype=np.float64)))

import torch  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.mesh import TriMesh  # noqa: E402
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.mvc import MeanValueCoordinateEncoder  # noqa: E402
from qcopt.neural_bijection.tutte.mvc_retraction import (
    CovarianceLogitLift,
)  # noqa: E402


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip().lower()
    return value if result.returncode == 0 and len(value) == 40 else None


def _variable_degree_disk() -> TriMesh:
    """Two-interior-vertex disk with a genuine degree-five latent fiber."""

    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.15],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 5], [1, 4, 5], [4, 0, 5], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=np.int64,
    )
    return TriMesh(vertices, faces)


def _helmert_zero_sum_basis(mask: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal padded basis for supported zero-row-sum logits."""

    if mask.dtype != torch.bool or mask.ndim != 2:
        raise TypeError("mask must be a two-dimensional Boolean tensor")
    rows, width = mask.shape
    dimension = sum(max(int(mask[row].sum()) - 1, 0) for row in range(rows))
    basis = torch.zeros((rows, width, dimension), dtype=torch.float64)
    column = 0
    for row in range(rows):
        slots = torch.nonzero(mask[row], as_tuple=False).flatten()
        degree = len(slots)
        if degree < 3:
            raise ValueError(
                "every interior decoder row must have degree at least three"
            )
        for index in range(degree - 1):
            denominator = math.sqrt((index + 1) * (index + 2))
            basis[row, slots[: index + 1], column] = 1.0 / denominator
            basis[row, slots[index + 1], column] = -(index + 1) / denominator
            column += 1
    if column != dimension:
        raise RuntimeError("internal gauge-basis dimension mismatch")
    return basis


def _recenter_supported(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    degree = mask.sum(dim=-1, keepdim=True)
    means = (
        torch.sum(
            torch.where(mask, values, torch.zeros_like(values)), dim=-1, keepdim=True
        )
        / degree
    )
    return torch.where(mask, values - means, torch.zeros_like(values))


def _row_sum_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    sums = torch.sum(torch.where(mask, values, torch.zeros_like(values)), dim=-1)
    return float(torch.max(torch.abs(sums)))


def _probability_mean_max(
    values: torch.Tensor, probabilities: torch.Tensor, mask: torch.Tensor
) -> float:
    means = torch.sum(
        torch.where(mask, probabilities * values, torch.zeros_like(values)), dim=-1
    )
    return float(torch.max(torch.abs(means)))


def _supported_max(
    values: torch.Tensor, mask: torch.Tensor, *, supported: bool
) -> float:
    selected = values[mask if supported else ~mask]
    return float(torch.max(torch.abs(selected))) if selected.numel() else 0.0


def _norms(
    values: torch.Tensor, probabilities: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float]:
    euclidean = torch.linalg.vector_norm(values[mask])
    weighted = torch.sqrt(torch.sum(probabilities[mask] * values[mask].square()))
    return float(euclidean), float(weighted)


def _finite_difference_tangent(
    decoder: DirectTutteLayer,
    logits: torch.Tensor,
    boundary: torch.Tensor,
    latent_direction: torch.Tensor,
    step: float,
) -> torch.Tensor:
    plus = decoder(logits + step * latent_direction, boundary)
    minus = decoder(logits - step * latent_direction, boundary)
    return (plus - minus) / (2.0 * step)


def run_comparison() -> dict[str, Any]:
    """Run the deterministic float64 comparison and return a finite receipt."""

    torch.set_num_threads(1)
    mesh = _variable_degree_disk()
    decoder = DirectTutteLayer(mesh)
    encoder = MeanValueCoordinateEncoder(mesh)
    system = decoder.system
    mask = torch.as_tensor(system.valid_mask, dtype=torch.bool)

    raw_logits = torch.tensor(
        [[0.31, -0.17, 0.24, -0.09, 0.11], [-0.28, 0.19, 0.37, 0.0, 0.0]],
        dtype=torch.float64,
    )
    boundary_count = len(system.loop)
    angles = torch.arange(boundary_count, dtype=torch.float64) * (
        2.0 * torch.pi / boundary_count
    )
    boundary = torch.stack((1.2 * torch.cos(angles), 0.83 * torch.sin(angles)), dim=-1)
    initial_control = decoder(raw_logits, boundary)
    canonical = encoder(initial_control)
    logits = canonical.logits
    probabilities = canonical.probabilities
    control = decoder(logits, boundary)

    direction = torch.zeros_like(control)
    direction[torch.as_tensor(system.interior)] = torch.tensor(
        [[0.027, -0.018], [-0.016, 0.031]], dtype=torch.float64
    )
    boundary_direction = direction.index_select(0, torch.as_tensor(system.loop))
    if bool(torch.any(boundary_direction != 0.0)):
        raise ValueError(
            "fixed-boundary comparison requires exactly zero boundary direction"
        )

    basis = _helmert_zero_sum_basis(mask)
    basis_matrix = basis.reshape(system.n_rows * system.max_degree, -1)
    gram = basis_matrix.T @ basis_matrix
    identity = torch.eye(gram.shape[0], dtype=torch.float64)
    basis_orthogonality_error = torch.max(torch.abs(gram - identity))
    basis_row_sum_error = torch.max(torch.abs(torch.sum(basis, dim=1)))
    if float(basis_orthogonality_error) > 8.0 * torch.finfo(torch.float64).eps:
        raise ValueError("zero-row-mean basis is not numerically orthonormal")
    if float(basis_row_sum_error) > 8.0 * torch.finfo(torch.float64).eps:
        raise ValueError("zero-row-mean basis contains a row-shift component")
    if _supported_max(basis, mask[..., None].expand_as(basis), supported=False) != 0.0:
        raise ValueError("zero-row-mean basis touches unsupported padded slots")

    theta_zero = torch.zeros(basis.shape[-1], dtype=torch.float64, requires_grad=True)

    def decode_coordinates(theta: torch.Tensor) -> torch.Tensor:
        represented = logits + torch.einsum("ijd,d->ij", basis, theta)
        return decoder(represented, boundary).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(decode_coordinates, theta_zero)
    if tuple(jacobian.shape) != (2 * system.n_vertices, basis.shape[-1]):
        raise RuntimeError("complete decoder Jacobian has an unexpected shape")
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError("complete decoder Jacobian must be finite")

    finite_difference_step = 2.0e-6
    finite_difference_columns = []
    for column in range(basis.shape[-1]):
        coordinate = torch.zeros(basis.shape[-1], dtype=torch.float64)
        coordinate[column] = 1.0
        finite_difference_columns.append(
            (
                decode_coordinates(finite_difference_step * coordinate)
                - decode_coordinates(-finite_difference_step * coordinate)
            )
            / (2.0 * finite_difference_step)
        )
    finite_difference_jacobian = torch.stack(finite_difference_columns, dim=-1)
    finite_difference_error = finite_difference_jacobian - jacobian
    finite_difference_relative = torch.linalg.matrix_norm(
        finite_difference_error
    ) / torch.linalg.matrix_norm(jacobian)

    u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
    largest = singular_values[0]
    rank_tolerance = (
        64.0 * torch.finfo(torch.float64).eps * max(jacobian.shape) * largest
    )
    retained = singular_values > rank_tolerance
    rank = int(retained.sum())
    expected_rank = 2 * system.n_rows
    if rank != expected_rank:
        raise ValueError(
            f"decoder Jacobian rank {rank} does not equal fixed-boundary tangent dimension {expected_rank}"
        )
    discarded = singular_values[~retained]
    if discarded.numel() == 0:
        raise ValueError("test mesh must expose a nontrivial decoder-Jacobian kernel")
    pseudoinverse = (vh[retained].T / singular_values[retained]) @ u[:, retained].T
    target = direction.reshape(-1)
    theta_moore_penrose = pseudoinverse @ target
    moore_penrose = torch.einsum("ijd,d->ij", basis, theta_moore_penrose)

    _, mvc_derivative = torch.autograd.functional.jvp(
        lambda vertices: encoder(vertices).logits,
        (control,),
        (direction,),
    )
    covariance_native = CovarianceLogitLift(system)(
        control, probabilities, direction
    ).delta_logits
    covariance_common = _recenter_supported(covariance_native, mask)

    common_lifts = {
        "moore_penrose": moore_penrose,
        "mvc_encoder_derivative": mvc_derivative,
        "covariance_common_gauge": covariance_common,
    }
    for name, values in common_lifts.items():
        if _supported_max(values, mask, supported=False) != 0.0:
            raise ValueError(f"{name} touches unsupported padded slots")
        if _row_sum_max(values, mask) > 64.0 * torch.finfo(torch.float64).eps:
            raise ValueError(
                f"{name} is not in the declared arithmetic-zero-row-mean gauge"
            )

    target_norm = torch.linalg.vector_norm(target)

    supported_base = logits[mask].detach().clone().requires_grad_(True)

    def decode_supported(values: torch.Tensor) -> torch.Tensor:
        represented = logits.masked_scatter(mask, values)
        return decoder(represented, boundary).reshape(-1)

    supported_jacobian = torch.autograd.functional.jacobian(
        decode_supported, supported_base
    )
    inverse_sqrt_metric = torch.diag(probabilities[mask].rsqrt())
    weighted_operator = supported_jacobian @ inverse_sqrt_metric
    weighted_u, weighted_singular_values, weighted_vh = torch.linalg.svd(
        weighted_operator, full_matrices=False
    )
    weighted_rank_tolerance = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(weighted_operator.shape)
        * weighted_singular_values[0]
    )
    weighted_retained = weighted_singular_values > weighted_rank_tolerance
    if int(weighted_retained.sum()) != expected_rank:
        raise ValueError("weighted supported decoder Jacobian has unexpected rank")
    weighted_pseudoinverse = (
        weighted_vh[weighted_retained].T / weighted_singular_values[weighted_retained]
    ) @ weighted_u[:, weighted_retained].T
    explicit_weighted_solution = inverse_sqrt_metric @ weighted_pseudoinverse @ target

    def lift_metrics(values: torch.Tensor) -> dict[str, Any]:
        flat = values.reshape(-1)
        gauge_coordinates = basis_matrix.T @ flat
        gauge_projection = basis_matrix @ gauge_coordinates
        gauge_residual = flat - gauge_projection
        decoded = jacobian @ gauge_coordinates
        decoded_residual = decoded - target
        finite_difference = _finite_difference_tangent(
            decoder, logits, boundary, values, finite_difference_step
        ).reshape(-1)
        finite_difference_residual = finite_difference - target
        euclidean, weighted = _norms(values, probabilities, mask)
        return {
            "euclidean_norm": euclidean,
            "weighted_norm": weighted,
            "supported_row_sum_maximum_absolute": _row_sum_max(values, mask),
            "probability_weighted_row_mean_maximum_absolute": _probability_mean_max(
                values, probabilities, mask
            ),
            "unsupported_maximum_absolute": _supported_max(
                values, mask, supported=False
            ),
            "gauge_projection_l2_residual": float(
                torch.linalg.vector_norm(gauge_residual)
            ),
            "decoded_tangent_l2_residual": float(
                torch.linalg.vector_norm(decoded_residual)
            ),
            "decoded_tangent_relative_l2_residual": float(
                torch.linalg.vector_norm(decoded_residual) / target_norm
            ),
            "decoded_boundary_maximum_absolute": float(
                torch.max(
                    torch.abs(
                        decoded.reshape(system.n_vertices, 2).index_select(
                            0, torch.as_tensor(system.loop)
                        )
                    )
                )
            ),
            "finite_difference_tangent_l2_residual": float(
                torch.linalg.vector_norm(finite_difference_residual)
            ),
            "finite_difference_tangent_relative_l2_residual": float(
                torch.linalg.vector_norm(finite_difference_residual) / target_norm
            ),
        }

    lifts = {name: lift_metrics(values) for name, values in common_lifts.items()}
    lifts["covariance_native_probability_gauge"] = lift_metrics(covariance_native)
    explicit_weighted_norm = torch.sqrt(
        torch.sum(probabilities[mask] * explicit_weighted_solution.square())
    )
    lifts["covariance_native_probability_gauge"].update(
        {
            "explicit_weighted_pseudoinverse_l2_difference": float(
                torch.linalg.vector_norm(
                    covariance_native[mask] - explicit_weighted_solution
                )
            ),
            "explicit_weighted_pseudoinverse_weighted_norm": float(
                explicit_weighted_norm
            ),
            "explicit_weighted_pseudoinverse_rank": int(weighted_retained.sum()),
            "explicit_weighted_pseudoinverse_rank_tolerance": float(
                weighted_rank_tolerance
            ),
            "explicit_weighted_pseudoinverse_decoded_l2_residual": float(
                torch.linalg.vector_norm(
                    supported_jacobian @ explicit_weighted_solution - target
                )
            ),
        }
    )

    pair_values = {
        "mvc_encoder_derivative_minus_moore_penrose": mvc_derivative - moore_penrose,
        "covariance_common_gauge_minus_moore_penrose": covariance_common
        - moore_penrose,
        "mvc_encoder_derivative_minus_covariance_common_gauge": mvc_derivative
        - covariance_common,
    }
    pairwise = {}
    for name, difference in pair_values.items():
        coordinates = basis_matrix.T @ difference.reshape(-1)
        kernel_image = jacobian @ coordinates
        euclidean, weighted = _norms(difference, probabilities, mask)
        pairwise[name] = {
            "euclidean_norm": euclidean,
            "weighted_norm": weighted,
            "kernel_l2_residual": float(torch.linalg.vector_norm(kernel_image)),
            "kernel_relative_l2_residual": float(
                torch.linalg.vector_norm(kernel_image) / target_norm
            ),
        }

    boundary_output_rows = torch.cat(
        [2 * torch.as_tensor(system.loop), 2 * torch.as_tensor(system.loop) + 1]
    )
    boundary_jacobian_maximum = torch.max(
        torch.abs(jacobian.index_select(0, boundary_output_rows))
    )
    if float(boundary_jacobian_maximum) != 0.0:
        raise ValueError("fixed-boundary decoder Jacobian has nonzero boundary rows")

    euclidean_slack = 256.0 * torch.finfo(torch.float64).eps
    weighted_slack = 256.0 * torch.finfo(torch.float64).eps
    optimality = {
        "moore_penrose_is_euclidean_minimum_in_common_gauge": bool(
            lifts["moore_penrose"]["euclidean_norm"]
            <= min(
                lifts["mvc_encoder_derivative"]["euclidean_norm"],
                lifts["covariance_common_gauge"]["euclidean_norm"],
            )
            + euclidean_slack
        ),
        "native_covariance_is_weighted_minimum_over_representatives": bool(
            lifts["covariance_native_probability_gauge"][
                "explicit_weighted_pseudoinverse_l2_difference"
            ]
            <= 512.0
            * torch.finfo(torch.float64).eps
            * max(
                1.0,
                lifts["covariance_native_probability_gauge"]["euclidean_norm"],
            )
            and lifts["covariance_native_probability_gauge"]["weighted_norm"]
            <= min(
                lifts["moore_penrose"]["weighted_norm"],
                lifts["mvc_encoder_derivative"]["weighted_norm"],
                lifts["covariance_common_gauge"]["weighted_norm"],
            )
            + weighted_slack
        ),
    }
    if not all(optimality.values()):
        raise ValueError(
            "observed norms contradict a declared minimum-norm characterization"
        )

    receipt: dict[str, Any] = {
        "schema": "phase5_route2_mvc_pseudoinverse_comparison_v1",
        "status": "ok",
        "research_question": (
            "How do J_D^+ d, dE_MVC(Y)[d], and L_Y d differ on the same fixed-boundary "
            "supported arithmetic-zero-row-mean logit gauge?"
        ),
        "scope": (
            "tiny CPU-float64 diagnostic of one variable-degree disk; not a scalable layer, "
            "optimization result, or Route-II completion verdict"
        ),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git_commit": _git_commit(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "mesh": {
            "vertices": system.n_vertices,
            "interior": system.n_rows,
            "boundary": len(system.loop),
            "faces": len(system.faces),
            "maximum_degree": system.max_degree,
            "supported_slots": int(mask.sum()),
            "gauge_dimension": basis.shape[-1],
        },
        "state": {
            "canonical_logits": logits.tolist(),
            "canonical_probabilities": probabilities.tolist(),
            "fixed_boundary": boundary.tolist(),
            "control": control.tolist(),
            "direction": direction.tolist(),
        },
        "gauge": {
            "name": "supported arithmetic-zero-row-mean logits",
            "basis": "row-block orthonormal Helmert basis in sorted supported-slot order",
            "unsupported_slots_fixed_to_zero": True,
            "basis_orthogonality_maximum_absolute_error": float(
                basis_orthogonality_error
            ),
            "basis_row_sum_maximum_absolute": float(basis_row_sum_error),
            "base_logit_row_mean_maximum_absolute": max(
                abs(float(torch.sum(logits[row, mask[row]]) / torch.sum(mask[row])))
                for row in range(system.n_rows)
            ),
        },
        "direction": {
            "full_l2_norm": float(target_norm),
            "interior_l2_norm": float(
                torch.linalg.vector_norm(
                    direction.index_select(0, torch.as_tensor(system.interior))
                )
            ),
            "boundary_maximum_absolute": float(
                torch.max(torch.abs(boundary_direction))
            ),
        },
        "jacobian": {
            "definition": (
                "all 2|V| decoder outputs differentiated with respect to the orthonormal "
                "supported zero-row-mean coordinates; boundary argument held fixed"
            ),
            "shape": list(jacobian.shape),
            "rank": rank,
            "expected_fixed_boundary_rank": expected_rank,
            "kernel_dimension_in_common_gauge": basis.shape[-1] - rank,
            "rank_tolerance_definition": "64*eps_float64*max(shape)*largest_singular_value",
            "rank_tolerance": float(rank_tolerance),
            "singular_values": singular_values.tolist(),
            "largest_singular_value": float(largest),
            "smallest_retained_singular_value": float(singular_values[retained][-1]),
            "largest_discarded_singular_value": float(discarded[0]),
            "retained_condition_number": float(largest / singular_values[retained][-1]),
            "boundary_row_maximum_absolute": float(boundary_jacobian_maximum),
            "finite_difference_step": finite_difference_step,
            "finite_difference_maximum_absolute_error": float(
                torch.max(torch.abs(finite_difference_error))
            ),
            "finite_difference_relative_frobenius_error": float(
                finite_difference_relative
            ),
        },
        "metric_definitions": {
            "decoded_tangent_l2_residual": "||J Q^T z - vec(d)||_2",
            "euclidean_norm": "sqrt(sum over supported slots of z_ij^2)",
            "weighted_norm": "sqrt(sum over supported slots of p_ij z_ij^2)",
            "kernel_l2_residual": "||J Q^T(z_a-z_b)||_2",
            "gauge_projection_l2_residual": "||vec(z)-Q Q^T vec(z)||_2",
            "explicit_weighted_pseudoinverse": (
                "W^(-1/2) (J_supported W^(-1/2))^+ vec(d), with W=diag(p)"
            ),
        },
        "lifts": lifts,
        "pairwise_differences": pairwise,
        "optimality_checks": optimality,
        "interpretation": {
            "moore_penrose": (
                "Euclidean minimum only in the explicitly fixed common zero-row-mean gauge."
            ),
            "covariance_native": (
                "Probability-weighted minimum over all supported logit representatives; it "
                "uses probability-weighted row mean zero and is reported separately."
            ),
            "common_covariance": (
                "The native covariance lift after subtracting each supported arithmetic row "
                "mean; the decoder tangent is unchanged but its weighted norm increases."
            ),
        },
    }
    # A success receipt may never hide a nonfinite diagnostic.
    json.dumps(receipt, allow_nan=False)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_comparison()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
