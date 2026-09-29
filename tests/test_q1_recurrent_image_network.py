"""Repeated image-conditioned passes preserve the shared Q1 grid."""

import torch

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from qcopt.neural_bijection.dense.q1_recurrent_image_network import (
    RecurrentQ1ImageRegistrationNetwork,
)


def test_zero_heads_equal_base_and_vjp_reaches_current_image_head() -> None:
    torch.manual_seed(29)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4, flow_hint=False)
    network = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2, 33: 1},
    )
    fixed = torch.rand(1, 1, 64, 64)
    moving = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        expected = base(fixed, moving)[0]
        actual = network(fixed, moving)[0]
    assert torch.equal(actual, expected)
    assert float(q1_corner_determinants(actual).amin()) > 0
    axis = torch.arange(33, dtype=actual.dtype) / 32
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.stack((xx + .05 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy), yy), -1)
    mapped = network(fixed, moving)[0]
    loss = (mapped - target[None]).square().mean()
    loss.backward()
    assert network.heads["17"].output.bias.grad is not None
    assert bool(torch.isfinite(network.heads["17"].output.bias.grad).all())
    assert float(network.heads["17"].output.bias.grad.abs().sum()) > 0


def test_nonzero_recurrent_heads_stay_corner_positive() -> None:
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4, flow_hint=False)
    network = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 3, 33: 2},
    )
    with torch.no_grad():
        network.heads["17"].output.bias.copy_(torch.tensor([2., -1.]))
        network.heads["33"].output.bias.copy_(torch.tensor([-2., 1.]))
    fixed = torch.zeros(1, 1, 64, 64)
    moving = torch.ones_like(fixed)
    result = network(fixed, moving)[0]
    assert bool(torch.isfinite(result).all())
    assert float(q1_corner_determinants(result).amin()) > 0
    assert torch.equal(result[:, 0], base(fixed, moving)[0][:, 0])
