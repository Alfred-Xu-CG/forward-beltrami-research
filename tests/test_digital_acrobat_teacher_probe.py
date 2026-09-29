"""Small contract tests for the ACROBAT image-to-safe-residual probe."""

import pytest
import torch

from tools.digital_acrobat_teacher_probe import (
    affine_prewarp,
    factored_residual_target,
    split_case_ids,
)
from tools.digital_acrobat_blank_ablation import actual_and_blank_outputs


def test_affine_prewarp_uses_image_pixel_centers():
    moving = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    matrix = torch.eye(2)[None]
    offset = torch.tensor([[.25, 0.]])
    warped = affine_prewarp(moving, matrix, offset)
    assert warped.shape == moving.shape
    torch.testing.assert_close(warped[0, 0, :, :3], moving[0, 0, :, 1:])
    torch.testing.assert_close(warped[0, 0, :, 3], moving[0, 0, :, 3])


def test_factored_target_recovers_residual_from_external_affine():
    residual = torch.tensor([[[[.0, .0], [.5, .0]], [[.1, 1.], [.9, .9]]]])
    matrix = torch.tensor([[[1.2, .3], [-.1, .8]]])
    offset = torch.tensor([[.2, -.4]])
    full = torch.einsum("bhwi,bji->bhwj", residual, matrix) + offset[:, None, None]
    recovered = factored_residual_target(full, matrix, offset)
    torch.testing.assert_close(recovered, residual, atol=1e-6, rtol=1e-6)


def test_case_level_split_rejects_leakage():
    train, test = split_case_ids([100, 156, 315], [315])
    assert train == [100, 156] and test == [315]
    with pytest.raises(ValueError):
        split_case_ids([100, 100, 156], [156])
    with pytest.raises(ValueError):
        split_case_ids([100, 156], [999])


def test_blank_ablation_changes_only_image_inputs():
    class Probe(torch.nn.Module):
        def forward(self, fixed, moving):
            value = (fixed + moving).mean()
            return value.expand(1, 2, 2, 2), None, None

    fixed = torch.ones(1, 1, 4, 4)
    moving = torch.full_like(fixed, 2.)
    actual, blank = actual_and_blank_outputs(Probe(), fixed, moving)
    assert actual.shape == blank.shape == (1, 2, 2, 2)
    torch.testing.assert_close(actual, torch.full_like(actual, 3.))
    torch.testing.assert_close(blank, torch.zeros_like(blank))
