"""Recurrent safe head preserves warm start and finite image-to-weight VJP."""

import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_mind_dense257_head import DenseSafeHead
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices


def test_four_pass_warm_start_and_update_vjp():
    torch.manual_seed(8)
    coarse = identity_vertices(257, device=torch.device("cpu"))
    features = torch.randn((1, 88, 256, 256))
    old = DenseSafeHead(width=8)
    with torch.no_grad():
        old.encoder[-1].weight.normal_(0., 0.001)
    new = RecurrentDenseSafeHead(width=8, passes=4)
    new.initialize_from_one_pass(old)
    expected = old(coarse, features)
    initial = new(coarse, features)
    torch.testing.assert_close(initial, expected, rtol=0, atol=2e-7)
    assert validate_q1_map(initial, coarse)["valid"]
    with torch.no_grad():
        new.heads[1].weight.normal_(0., .001)
    moved = new(coarse, features)
    assert validate_q1_map(moved, coarse)["valid"]
    assert torch.any(torch.abs(moved - initial) > 1e-7)
    moved.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in new.parameters())
