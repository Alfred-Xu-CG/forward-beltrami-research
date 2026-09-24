"""OOD texture generation keeps the deformation sample exactly fixed."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from phase7_train_forward_pyramid_image import (  # noqa: E402
    replace_test_appearance, sinusoidal_mode_coefficients,
)


def test_texture_shift_preserves_maps_and_rewarps_images() -> None:
    side = 33
    axis = torch.linspace(0, 1, side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None].expand(2, -1, -1, -1)
    source = torch.rand(2, 1, side, side)
    data = source, source, identity
    for mode in ("spots", "crosswaves"):
        fixed, moving, mapped = replace_test_appearance(data, mode, 123)
        assert mapped is identity
        assert fixed.shape == moving.shape == source.shape
        assert not torch.allclose(moving, source)
        expected = F.grid_sample(
            moving, 2 * mapped - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        assert torch.allclose(fixed, expected, atol=1e-6)


def test_sinusoidal_mode_projection_returns_two_coordinate_coefficient() -> None:
    side = 129
    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    carrier = torch.sin(2 * torch.pi * 7 * xx) * torch.sin(2 * torch.pi * 7 * yy)
    identity = torch.stack((xx, yy), dim=-1)
    mapped = identity + carrier[..., None] * torch.tensor((.002, -.003), dtype=torch.float64)
    coefficient = sinusoidal_mode_coefficients(mapped[None], 7)
    torch.testing.assert_close(
        coefficient[0], torch.tensor((.002, -.003), dtype=torch.float64),
        atol=1e-15, rtol=0,
    )
