"""Odd-extension DST and its first-order gradient."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense.sine_spectral import dst2, idst2, keep_vector_sine_modes, spectralize_bounded_logits


def test_dst2_round_trip_and_zero_boundary_basis() -> None:
    torch.manual_seed(672)
    values = torch.randn(2, 3, 7, 9, dtype=torch.float64, requires_grad=True)
    reconstructed = idst2(dst2(values))
    assert torch.allclose(reconstructed, values, rtol=1e-12, atol=1e-12)
    reconstructed.square().mean().backward()
    assert torch.allclose(values.grad, 2 * values.detach() / values.numel(), rtol=1e-11, atol=1e-11)


def test_joint_top_modes_round_trip_and_directional_vjp() -> None:
    torch.manual_seed(841)
    values = (0.2 * torch.randn(1, 7, 7, 2, dtype=torch.float64)).requires_grad_()
    direction = torch.randn_like(values)
    cotangent = torch.randn_like(values)

    def objective(candidate: torch.Tensor) -> torch.Tensor:
        return (keep_vector_sine_modes(candidate, 5) * cotangent).sum()

    derivative = torch.autograd.grad(objective(values), values)[0]
    predicted = (derivative * direction).sum()
    step = 1e-6
    measured = (objective(values + step * direction) - objective(values - step * direction)) / (2 * step)
    assert torch.allclose(predicted, measured, rtol=1e-5, atol=1e-6)
    assert torch.allclose(keep_vector_sine_modes(values, 49), values, rtol=1e-12, atol=1e-12)


def test_spectralized_logits_are_finite_and_differentiable() -> None:
    torch.manual_seed(917)
    logits = torch.randn(1, 7, 7, 2, dtype=torch.float64, requires_grad=True)
    result = spectralize_bounded_logits(logits, side=9, raw_span=2.0, count=6)
    assert result.shape == logits.shape and torch.isfinite(result).all()
    result.square().mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
