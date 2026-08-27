"""Coupled augmented LSQC solver for matched multi-chart atlases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from .constraints import LinearConstraints
from .lsqc import assemble_lsqc_operator
from .multichart import Atlas, build_affine_compatibility


@dataclass
class MultiChartSolveResult:
    maps: dict[str, np.ndarray]
    algebraic_residual: float
    constraint_residual: float
    seam_residual: float
    system: sparse.csc_matrix


def solve_coupled_lsqc(
    atlas: Atlas,
    mus: dict[str, np.ndarray],
    constraints: LinearConstraints,
    *,
    weighted: bool = False,
) -> MultiChartSolveResult:
    if constraints.C.shape[1] != atlas.n_coordinates:
        raise ValueError("global constraints have incompatible column count")
    operators = [
        assemble_lsqc_operator(chart.mesh, mus[chart.name], weighted=weighted)
        for chart in atlas.charts
    ]
    operator = sparse.block_diag(operators, format="csr")
    n_residuals, n_coordinates = operator.shape
    n_constraints = constraints.C.shape[0]
    system = sparse.bmat(
        [
            [
                -sparse.eye(n_residuals, format="csr"),
                operator,
                sparse.csr_matrix((n_residuals, n_constraints)),
            ],
            [
                operator.T,
                sparse.csr_matrix((n_coordinates, n_coordinates)),
                constraints.C.T,
            ],
            [
                sparse.csr_matrix((n_constraints, n_residuals)),
                constraints.C,
                sparse.csr_matrix((n_constraints, n_constraints)),
            ],
        ],
        format="csc",
    )
    rhs = np.concatenate((np.zeros(n_residuals + n_coordinates), constraints.d))
    solution = splinalg.splu(system).solve(rhs)
    coordinates = solution[n_residuals : n_residuals + n_coordinates]
    algebraic = float(np.linalg.norm(system @ solution - rhs) / max(np.linalg.norm(rhs), 1.0))
    constraint_residual = float(np.linalg.norm(constraints.C @ coordinates - constraints.d, ord=np.inf))
    seam = build_affine_compatibility(atlas)
    seam_residual = float(np.linalg.norm(seam.C @ coordinates - seam.d, ord=np.inf))
    return MultiChartSolveResult(
        maps=atlas.unpack_maps(coordinates),
        algebraic_residual=algebraic,
        constraint_residual=constraint_residual,
        seam_residual=seam_residual,
        system=system,
    )
