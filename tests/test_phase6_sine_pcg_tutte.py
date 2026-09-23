"""Independent sparse solve, spectral bound, topology, and implicit VJP tests."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer
from qcopt.neural_bijection.dense.sine_pcg_tutte import _interior_action, _sine_inverse


def _random_logits(side: int, batch: int = 1) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(65257)
    return (
        0.4 * torch.randn(batch, side, side - 1, dtype=torch.float64),
        0.4 * torch.randn(batch, side - 1, side, dtype=torch.float64),
        0.4 * torch.randn(batch, side - 1, side - 1, dtype=torch.float64),
    )


def _independent_matrix_and_rhs(side: int, weights: tuple[np.ndarray, ...]) -> tuple[sparse.csc_matrix, np.ndarray]:
    horizontal, vertical, diagonal = weights
    edges = []
    for row in range(side):
        for col in range(side - 1):
            edges.append(((row, col), (row, col + 1), horizontal[row, col]))
    for row in range(side - 1):
        for col in range(side):
            edges.append(((row, col), (row + 1, col), vertical[row, col]))
    for row in range(side - 1):
        for col in range(side - 1):
            edges.append(((row, col), (row + 1, col + 1), diagonal[row, col]))
    n = (side - 2) ** 2
    matrix = sparse.lil_matrix((n, n), dtype=float)
    rhs = np.zeros((n, 2))
    def interior(vertex):
        row, col = vertex
        return (row - 1) * (side - 2) + col - 1 if 0 < row < side - 1 and 0 < col < side - 1 else None
    for a, b, weight in edges:
        ia, ib = interior(a), interior(b)
        if ia is not None:
            matrix[ia, ia] += weight
            if ib is not None:
                matrix[ia, ib] -= weight
            else:
                rhs[ia] += weight * np.asarray((b[1], b[0])) / (side - 1)
        if ib is not None:
            matrix[ib, ib] += weight
            if ia is not None:
                matrix[ib, ia] -= weight
            else:
                rhs[ib] += weight * np.asarray((a[1], a[0])) / (side - 1)
    return matrix.tocsc(), rhs


def test_sine_inverse_of_five_point_dirichlet_operator() -> None:
    side = 9
    layer = SinePreconditionedTutteLayer(side)
    torch.manual_seed(7)
    rhs = torch.randn(2, side - 2, side - 2, 2, dtype=torch.float64)
    solved = _sine_inverse(rhs, layer._eigenvalues)
    ones_h = torch.ones(2, side, side - 1, dtype=torch.float64)
    ones_v = torch.ones(2, side - 1, side, dtype=torch.float64)
    zeros_d = torch.zeros(2, side - 1, side - 1, dtype=torch.float64)
    assert torch.allclose(_interior_action(solved, ones_h, ones_v, zeros_d), rhs, rtol=1e-12, atol=1e-12)


def test_sine_pcg_matches_independent_sparse_solve_and_vjp() -> None:
    side = 9
    layer = SinePreconditionedTutteLayer(side, tolerance=1e-12)
    logits = tuple(value.requires_grad_() for value in _random_logits(side))
    mapped = layer(*logits)
    weights = tuple((1 + 3 * value.sigmoid())[0].detach().numpy() for value in logits)
    matrix, rhs = _independent_matrix_and_rhs(side, weights)
    direct = sparse_linalg.spsolve(matrix, rhs).reshape(side - 2, side - 2, 2)
    assert np.max(np.abs(mapped.detach().numpy()[0, 1:-1, 1:-1] - direct)) < 1e-11
    assert layer.last_forward_stats["true_relative_residual"] < 1e-12
    assert layer.last_forward_stats["minimum_signed_area_ratio"] > 0
    torch.manual_seed(9)
    cotangent = torch.randn_like(mapped)
    loss = (mapped * cotangent).sum()
    gradient = torch.autograd.grad(loss, logits)
    assert layer.last_backward_stats["true_relative_residual"] < 1e-12
    direction = tuple(torch.randn_like(value) for value in logits)
    analytic = sum((g * d).sum() for g, d in zip(gradient, direction))
    epsilon = 1e-5
    with torch.no_grad():
        plus = (layer(*(value + epsilon * d for value, d in zip(logits, direction))) * cotangent).sum()
        minus = (layer(*(value - epsilon * d for value, d in zip(logits, direction))) * cotangent).sum()
    measured = (plus - minus) / (2 * epsilon)
    assert torch.allclose(analytic, measured, rtol=2e-5, atol=1e-7)


def test_mixed_identity_and_random_batch_and_energy_bounds() -> None:
    side = 9
    layer = SinePreconditionedTutteLayer(side, tolerance=1e-11)
    random_logits = _random_logits(side, batch=2)
    logits = tuple(value.clone() for value in random_logits)
    for value in logits:
        value[0].zero_()
    mapped = layer(*logits)
    assert torch.allclose(mapped[0], layer._source, atol=1e-13)
    assert layer.last_forward_stats["minimum_signed_area_ratio"] > 0
    torch.manual_seed(37)
    trial = torch.randn(2, side - 2, side - 2, 2, dtype=torch.float64)
    weights = tuple(1 + 3 * value.sigmoid() for value in logits)
    five_h = torch.ones_like(weights[0])
    five_v = torch.ones_like(weights[1])
    zero_d = torch.zeros_like(weights[2])
    a_energy = (trial * _interior_action(trial, *weights)).sum()
    p_energy = (trial * _interior_action(trial, five_h, five_v, zero_d)).sum()
    assert a_energy >= p_energy - 1e-11
    assert a_energy <= 12 * p_energy + 1e-11
