"""Sparse fixed-point P1 sampler matches the trusted geometric evaluator."""

import numpy as np
import torch

from tools.digital_birl_landmark_score import p1_at_queries
from tools.digital_mind_sparse_match_finetune import p1_at_points


def test_sparse_p1_interpolation_and_vertex_vjp():
    generator = np.random.default_rng(17)
    vertices = generator.normal(size=(9, 9, 2)).astype(np.float32)
    query = generator.uniform(0, 1, size=(30, 2)).astype(np.float32)
    query[0] = (0., 0.)
    query[1] = (1., 1.)
    variable = torch.tensor(vertices[None], requires_grad=True)
    result = p1_at_points(variable, torch.tensor(query))
    expected = p1_at_queries(vertices, query)
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=1e-6)
    result.square().mean().backward()
    assert variable.grad is not None
    assert torch.isfinite(variable.grad).all()
