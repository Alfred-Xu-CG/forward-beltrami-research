"""Gaussian match channels and conditional safe head have finite gradients."""

import torch

from tools.digital_mind_dense257_head import DenseSafeHead
from tools.digital_mind_match_conditioned257 import gaussian_match_features
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices


def test_match_raster_and_recurrent_warm_start():
    source = torch.tensor([[.2, .3], [.7, .8]], requires_grad=True)
    target = torch.tensor([[.21, .32], [.72, .81]], requires_grad=True)
    raster = gaussian_match_features(source, target, side=32)
    assert raster.shape == (1, 6, 32, 32)
    assert torch.isfinite(raster).all()
    raster.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert target.grad is not None and torch.isfinite(target.grad).all()

    old = DenseSafeHead(width=8)
    model = RecurrentDenseSafeHead(width=8, passes=4, match_channels=6)
    model.initialize_from_one_pass(old)
    coarse = identity_vertices(257, device=torch.device("cpu"))
    image = torch.randn((1, 88, 256, 256))
    zero_match = torch.zeros((1, 6, 256, 256))
    initial = model(coarse, image, match_feature=zero_match)
    torch.testing.assert_close(initial, old(coarse, image), rtol=0, atol=2e-7)
