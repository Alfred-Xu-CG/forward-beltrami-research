"""Descriptor-force cue shape and first-order image-input gradient."""

import torch

from tools.digital_image_force_feature257 import image_force_features
from tools.digital_q1_dhr_distill import identity_vertices


def test_force_shape_finite_and_image_vjp():
    torch.manual_seed(23)
    fixed = torch.rand((1, 8, 256, 256), dtype=torch.float64,
                       requires_grad=True)
    moving = torch.rand((1, 8, 256, 256), dtype=torch.float64)
    mask = torch.ones((1, 1, 256, 256), dtype=torch.float64)
    current = identity_vertices(257, device=torch.device("cpu")).double()
    value = image_force_features(fixed, moving, mask, current)
    assert value.shape == (1, 6, 256, 256)
    assert bool(torch.isfinite(value).all())
    scalar = value[:, :5].square().mean()
    gradient = torch.autograd.grad(scalar, fixed)[0]
    assert bool(torch.isfinite(gradient).all())
    index = (0, 0, 100, 100)
    epsilon = 1e-5
    plus = fixed.detach().clone()
    minus = fixed.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    numerical = (
        image_force_features(plus, moving, mask, current)[:, :5].square().mean()
        - image_force_features(minus, moving, mask, current)[:, :5].square().mean()
    ) / (2 * epsilon)
    torch.testing.assert_close(gradient[index], numerical, rtol=.03, atol=1e-7)


if __name__ == "__main__":
    test_force_shape_finite_and_image_vjp()
    print("passed")
