"""High64 is the same textured task with doubled frequency and halved amplitude."""

import math

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset


def test_high64_reuses_texture_and_preserves_analytic_target():
    low = make_dataset(3, 65, 99317, return_coefficients=True,
                       target_family="high32")
    high = make_dataset(3, 65, 99317, return_coefficients=True,
                        target_family="high64")
    fixed, moving, target, coefficients = high
    assert torch.equal(moving, low[1])
    torch.testing.assert_close(coefficients[:, :2], low[3][:, :2], rtol=0, atol=0)
    torch.testing.assert_close(coefficients[:, 2], low[3][:, 2] / 2, rtol=0, atol=0)
    line = torch.linspace(0, 1, 65)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    low_wave = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    high_wave = torch.sin(128 * math.pi * xx) * torch.sin(128 * math.pi * yy)
    ax, ay, af = (coefficients[:, k, None, None] for k in range(3))
    expected = torch.stack((xx + ax * low_wave + af * high_wave,
                            yy + ay * low_wave + af * high_wave), dim=-1)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        fixed, F.grid_sample(moving, 2 * target - 1, mode="bilinear",
                             padding_mode="border", align_corners=True),
        rtol=0, atol=0)
    # This bound is uniform in x,y and in every coefficient from the generator.
    lower = (1 - 2 * math.pi * (0.015 + 0.025)
             - 0.00125 * 128 * math.pi
             - 0.00125 * 0.020 * 2 * (2 * math.pi) * (128 * math.pi))
    assert lower > 0.119
