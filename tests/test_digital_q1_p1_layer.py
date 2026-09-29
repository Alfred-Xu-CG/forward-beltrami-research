"""A factorized fixed-diagonal P1 layer preserves the decoder's gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_p1_eval import p1_map_at_unit_queries
from tools.digital_q1_p1_layer import (
    FactorizedP1Map, SafeSparseMatchP1Layer, evaluate_factorized_p1,
    fixed_sw_ne_faces,
)


def test_procedural_sw_ne_faces_have_the_declared_orientation() -> None:
    faces = fixed_sw_ne_faces(3)
    assert faces.shape == (8, 3)
    assert faces[:2].tolist() == [[0, 1, 4], [0, 4, 3]]
    assert faces[-2:].tolist() == [[4, 5, 8], [4, 8, 7]]


def test_fixed_query_p1_values_match_separate_numpy_evaluator_and_vjp() -> None:
    vertices = torch.tensor([[[[0., 0.], [1., .1]],
                              [[-.1, 1.], [1.1, 1.2]]]],
                            dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor([[[1.03, .07], [-.02, .98]]],
                          dtype=torch.float64, requires_grad=True)
    offset = torch.tensor([[.02, -.01]], dtype=torch.float64, requires_grad=True)
    mapped = FactorizedP1Map(vertices, matrix, offset)
    queries = torch.tensor([[[.12, .18], [.38, .79], [.83, .29]]],
                           dtype=torch.float64)
    actual = evaluate_factorized_p1(mapped, queries)
    expected_residual = p1_map_at_unit_queries(
        vertices.detach().numpy()[0], queries.numpy()[0])
    # Scalar expansion keeps this independent two-coordinate check free of
    # NumPy BLAS, which aborts in some Windows PyTorch/MKL test environments.
    reference_matrix = matrix.detach().numpy()[0]
    reference_offset = offset.detach().numpy()[0]
    expected = np.stack((
        reference_matrix[0, 0] * expected_residual[:, 0]
        + reference_matrix[0, 1] * expected_residual[:, 1] + reference_offset[0],
        reference_matrix[1, 0] * expected_residual[:, 0]
        + reference_matrix[1, 1] * expected_residual[:, 1] + reference_offset[1],
    ), axis=-1)
    np.testing.assert_allclose(actual.detach().numpy()[0], expected, atol=1e-14)
    actual.square().sum().backward()
    for gradient in (vertices.grad, matrix.grad, offset.grad):
        assert gradient is not None and bool(torch.isfinite(gradient).all())


def test_batched_safe_p1_layer_has_independent_outputs_and_full_vjp() -> None:
    source = torch.tensor([[[.17, .23], [.41, .38], [.76, .62], [.53, .84]],
                           [[.19, .25], [.39, .44], [.72, .67], [.55, .81]]],
                          dtype=torch.float64, requires_grad=True)
    motion = torch.tensor([[[.008, -.004]] * 4,
                           [[-.006, .005]] * 4], dtype=torch.float64)
    target = (source + motion).detach().requires_grad_(True)
    matrix = torch.tensor([[[1.02, .03], [-.01, .97]],
                           [[.98, -.02], [.01, 1.01]]],
                          dtype=torch.float64, requires_grad=True)
    offset = torch.tensor([[.01, -.02], [-.005, .014]],
                          dtype=torch.float64, requires_grad=True)
    layer = SafeSparseMatchP1Layer(side=33, update_sigmas=(.12, .06),
                                   proposal_mode="residual")
    mapped = layer(source, target, post_affine_matrix=matrix,
                   post_affine_offset=offset)
    assert mapped.residual_vertices.shape == (2, 33, 33, 2)
    assert mapped.faces().shape == (2 * 32 * 32, 3)
    assert not torch.allclose(mapped.residual_vertices[0],
                              mapped.residual_vertices[1])
    reference = identity_vertices(33, device=torch.device("cpu")).double()
    for item in mapped.residual_vertices:
        check = validate_q1_map(item[None], reference)
        assert check["nonpositive_corners"] == 0
        assert check["boundary_max_error"] == 0
    queries = torch.tensor([[[.12, .28], [.76, .57]],
                            [[.29, .37], [.67, .82]]], dtype=torch.float64)
    values = evaluate_factorized_p1(mapped, queries)
    values.square().sum().backward()
    for gradient in (source.grad, target.grad, matrix.grad, offset.grad):
        assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert bool(torch.any(source.grad[0] != 0))
    assert bool(torch.any(source.grad[1] != 0))


def test_affine_acceptance_uses_exact_stored_binary_sign() -> None:
    # A float32 determinant calculation rounds this negative exact value up.
    matrix = torch.tensor([[[1.7181129455566406, .8094261288642883],
                            [2.147641181945801, 1.0117826461791992]]],
                          dtype=torch.float32)
    assert bool(torch.linalg.det(matrix)[0] > 0)
    source = torch.tensor([[[.25, .25], [.75, .75]]])
    target = source + torch.tensor([.01, 0.])
    layer = SafeSparseMatchP1Layer(side=17, update_sigmas=(.12,))
    with pytest.raises(ValueError, match="exact-positive"):
        layer(source, target, post_affine_matrix=matrix)
