"""Small-grid topology and whole-feedback first-order gradient check."""

from __future__ import annotations

import torch

from phase6_train_multisample_image import ConvexQuadLocalImageEncoder
from qcopt.neural_bijection.dense import SpectralSafeFeedbackLayer


def _minimum_area(control: torch.Tensor) -> torch.Tensor:
    a = control[:, :-1, :-1]
    b = control[:, :-1, 1:]
    c = control[:, 1:, 1:]
    d = control[:, 1:, :-1]
    def cross(first, second):
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    return torch.minimum(cross(b - a, c - a).amin(), cross(c - a, d - a).amin())


def test_two_stage_spectral_feedback_preserves_original_grid_faces_and_vjp() -> None:
    torch.manual_seed(71129)
    side = 9
    encoder = ConvexQuadLocalImageEncoder(side, width=4, head_mode="multilevel", body_mode="local")
    layer = SpectralSafeFeedbackLayer(side, initial_modes=4, extra_modes=4,
                                      extra_passes=1, extra_gain=0.25, floor_fraction=0.2)
    moving = torch.randn(1, 1, 33, 33)
    fixed = torch.roll(moving, shifts=(1, -1), dims=(-2, -1))
    root, levels, local = encoder(torch.cat((fixed, moving), dim=1))
    local = local.detach().requires_grad_()
    output = layer(fixed, moving, (root.detach(), tuple(tuple(value.detach() for value in group) for group in levels), local))
    assert output.shape == (1, side, side, 2)
    assert _minimum_area(output).item() > 0
    line = torch.linspace(0, 1, side)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    source = torch.stack((xx, yy), dim=-1)[None]
    assert torch.allclose(output[:, 0], source[:, 0])
    assert torch.allclose(output[:, -1], source[:, -1])
    assert torch.allclose(output[:, :, 0], source[:, :, 0])
    assert torch.allclose(output[:, :, -1], source[:, :, -1])
    cotangent = torch.randn_like(output)
    direction = torch.randn_like(local)
    analytic = (torch.autograd.grad((output * cotangent).sum(), local)[0] * direction).sum()
    epsilon = 1e-4
    def value(perturbed):
        return (layer(fixed, moving, (root.detach(), tuple(tuple(x.detach() for x in group) for group in levels), perturbed)) * cotangent).sum()
    measured = (value(local.detach() + epsilon * direction) - value(local.detach() - epsilon * direction)) / (2 * epsilon)
    assert torch.isfinite(analytic)
    assert torch.allclose(analytic, measured, rtol=2e-2, atol=2e-4)


def test_checkpointed_qc_feedback_matches_output_and_first_order_vjp() -> None:
    torch.manual_seed(71130)
    side = 9
    images = torch.randn(1, 2, 33, 33)
    fixed, moving = images[:, :1], images[:, 1:]
    encoder = ConvexQuadLocalImageEncoder(side, width=4, head_mode="multilevel", body_mode="local")
    initial_latent = encoder(images)
    ordinary = SpectralSafeFeedbackLayer(
        side, initial_modes=4, extra_modes=4, extra_passes=2,
        extra_gain=1, extra_qc_cap=0.8, floor_fraction=0.8)
    recomputed = SpectralSafeFeedbackLayer(
        side, initial_modes=4, extra_modes=4, extra_passes=2,
        extra_gain=1, extra_qc_cap=0.8, floor_fraction=0.8,
        extra_checkpoint=True)
    local1 = initial_latent[2].detach().requires_grad_()
    local2 = initial_latent[2].detach().requires_grad_()
    latent1 = (initial_latent[0].detach(),
               tuple(tuple(value.detach() for value in group) for group in initial_latent[1]),
               local1)
    latent2 = (latent1[0], latent1[1], local2)
    output1 = ordinary(fixed, moving, latent1)
    output2 = recomputed(fixed, moving, latent2)
    assert torch.allclose(output1, output2, rtol=0, atol=0)
    cotangent = torch.randn_like(output1)
    gradient1 = torch.autograd.grad((output1 * cotangent).sum(), local1)[0]
    gradient2 = torch.autograd.grad((output2 * cotangent).sum(), local2)[0]
    assert torch.isfinite(gradient1).all()
    assert torch.allclose(gradient1, gradient2, rtol=1e-5, atol=1e-6)
