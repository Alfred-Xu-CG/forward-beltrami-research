"""The corrected residual student starts exactly at its frozen P1 input."""

import torch

from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices


def test_zero_recurrent_heads_preserve_initial_map_and_have_vjp():
    model = RecurrentDenseSafeHead(passes=4, match_channels=6)
    for head in model.heads:
        head.weight.data.zero_()
        head.bias.data.zero_()
    initial = identity_vertices(257, device=torch.device("cpu"))
    fine = torch.zeros((1, 88, 256, 256))
    matches = torch.zeros((1, 6, 256, 256))
    mapped = model(initial, fine, match_feature=matches)
    assert torch.equal(mapped, initial)
    loss = mapped[:, 1:-1, 1:-1, 0].mean()
    loss.backward()
    assert all(head.bias.grad is not None and
               torch.isfinite(head.bias.grad).all() for head in model.heads)


def test_dilated_zero_heads_preserve_initial_map_and_have_vjp():
    model = RecurrentDenseSafeHead(passes=4, match_channels=6,
                                   context="dilated")
    initial = identity_vertices(257, device=torch.device("cpu"))
    fine = torch.zeros((1, 88, 256, 256))
    matches = torch.zeros((1, 6, 256, 256))
    mapped = model(initial, fine, match_feature=matches)
    assert torch.equal(mapped, initial)
    mapped[:, 1:-1, 1:-1, 0].mean().backward()
    assert all(head.bias.grad is not None and
               torch.isfinite(head.bias.grad).all() for head in model.heads)


def test_image_only_dilated_zero_heads_preserve_map_and_have_vjp():
    model = RecurrentDenseSafeHead(passes=4, match_channels=0,
                                   context="dilated")
    initial = identity_vertices(257, device=torch.device("cpu"))
    fine = torch.zeros((1, 88, 256, 256))
    mapped = model(initial, fine)
    assert torch.equal(mapped, initial)
    mapped[:, 1:-1, 1:-1, 1].mean().backward()
    assert all(head.bias.grad is not None and
               torch.isfinite(head.bias.grad).all() for head in model.heads)


def test_unet_zero_head_preserves_map_and_has_vjp():
    model = RecurrentDenseSafeHead(passes=1, match_channels=6,
                                   context="unet")
    initial = identity_vertices(257, device=torch.device("cpu"))
    fine = torch.zeros((1, 88, 256, 256))
    matches = torch.zeros((1, 6, 256, 256))
    mapped = model(initial, fine, match_feature=matches)
    assert torch.equal(mapped, initial)
    mapped[:, 1:-1, 1:-1, 0].mean().backward()
    assert model.heads[0].bias.grad is not None
    assert torch.isfinite(model.heads[0].bias.grad).all()


def test_multilevel_unet_zero_controls_preserve_map_and_have_vjp():
    model = RecurrentDenseSafeHead(passes=1, match_channels=6,
                                   context="multilevel_unet")
    initial = identity_vertices(257, device=torch.device("cpu"))
    fine = torch.zeros((1, 88, 256, 256))
    matches = torch.zeros((1, 6, 256, 256))
    mapped = model(initial, fine, match_feature=matches)
    assert torch.equal(mapped, initial)
    mapped[:, 1:-1, 1:-1, 0].mean().backward()
    assert all(head.bias.grad is not None and
               torch.isfinite(head.bias.grad).all() and
               torch.any(head.bias.grad != 0) for head in model.heads)


def test_multilevel_unet_directional_vjp_matches_finite_difference():
    model = RecurrentDenseSafeHead(passes=1, match_channels=6,
                                   context="multilevel_unet")
    initial = identity_vertices(257, device=torch.device("cpu"))
    fine = torch.zeros((1, 88, 256, 256))
    matches = torch.zeros((1, 6, 256, 256))
    parameter = model.heads[2].bias
    value = model(initial, fine, match_feature=matches)[0, 128, 128, 0]
    analytical = torch.autograd.grad(value, parameter)[0][0].item()
    epsilon = 1e-3
    with torch.no_grad():
        parameter[0] = epsilon
        plus = model(initial, fine, match_feature=matches)[0, 128, 128, 0].item()
        parameter[0] = -epsilon
        minus = model(initial, fine, match_feature=matches)[0, 128, 128, 0].item()
        parameter[0] = 0
    numerical = (plus - minus) / (2 * epsilon)
    assert abs(analytical) > 1e-4
    assert abs(numerical - analytical) <= .03 * abs(analytical)


if __name__ == "__main__":
    test_zero_recurrent_heads_preserve_initial_map_and_have_vjp()
    test_dilated_zero_heads_preserve_initial_map_and_have_vjp()
    test_image_only_dilated_zero_heads_preserve_map_and_have_vjp()
    test_unet_zero_head_preserves_map_and_has_vjp()
    test_multilevel_unet_zero_controls_preserve_map_and_have_vjp()
    test_multilevel_unet_directional_vjp_matches_finite_difference()
    print("six_passed")
