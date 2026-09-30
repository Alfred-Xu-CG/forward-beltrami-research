"""The analytic machine-match pass is differentiable and Q1 safe."""

import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_match_feedback257 import AnalyticMatchFeedback257
from tools.digital_mind_sparse_match_finetune import p1_at_points
from tools.digital_q1_dhr_distill import identity_vertices


def test_one_match_feedback_reduces_error_with_valid_saved_geometry_and_vjp():
    layer = AnalyticMatchFeedback257().double()
    identity = identity_vertices(257, device=torch.device("cpu")).double()
    source = torch.tensor([[.513, .507]], dtype=torch.float64,
                          requires_grad=True)
    target = torch.tensor([[.533, .517]], dtype=torch.float64,
                          requires_grad=True)
    before = torch.linalg.vector_norm(p1_at_points(identity, source) - target)
    mapped = layer(identity, source, target)
    after = torch.linalg.vector_norm(p1_at_points(mapped, source) - target)
    assert after < before
    assert validate_q1_map(mapped, identity)["valid"]
    objective = mapped[0, 130, 131, 0] + .3 * mapped[0, 129, 131, 1]
    source_grad, target_grad = torch.autograd.grad(objective, (source, target))
    assert bool(torch.isfinite(source_grad).all())
    assert bool(torch.isfinite(target_grad).all())
    assert bool(target_grad.abs().sum() > 0)

    epsilon = 1e-4
    positive = target.detach().clone()
    negative = target.detach().clone()
    positive[0, 0] += epsilon
    negative[0, 0] -= epsilon
    finite_difference = (
        layer(identity, source.detach(), positive)[0, 130, 131, 0] -
        layer(identity, source.detach(), negative)[0, 130, 131, 0]
    ) / (2 * epsilon)
    analytic = torch.autograd.grad(
        layer(identity, source.detach(), target)[0, 130, 131, 0], target)[0][0, 0]
    assert torch.allclose(analytic, finite_difference, atol=1e-6, rtol=.03)
