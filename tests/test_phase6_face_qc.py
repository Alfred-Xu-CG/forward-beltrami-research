"""Direct original-triangle Beltrami formula and VJP checks."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense import structured_p1_face_beltrami_modulus, structured_p1_qc_tail_penalty


def test_affine_face_beltrami_matches_exact_value() -> None:
    line = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    control = torch.stack((1.2 * xx, 0.8 * yy), dim=-1)[None]
    modulus = structured_p1_face_beltrami_modulus(control)
    assert modulus.shape == (1, 2, 8, 8)
    assert torch.allclose(modulus, torch.full_like(modulus, 0.2), rtol=1e-12, atol=1e-12)


def test_qc_tail_vjp_matches_directional_difference() -> None:
    torch.manual_seed(3257)
    line = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    base = torch.stack((1.1 * xx, 0.9 * yy), dim=-1)[None]
    control = (base + 0.002 * torch.randn_like(base)).requires_grad_()
    direction = torch.randn_like(control)
    value = structured_p1_qc_tail_penalty(control, 0.05)
    gradient = torch.autograd.grad(value, control)[0]
    analytic = (gradient * direction).sum()
    epsilon = 1e-6
    measured = (
        structured_p1_qc_tail_penalty(control + epsilon * direction, 0.05)
        - structured_p1_qc_tail_penalty(control - epsilon * direction, 0.05)
    ) / (2 * epsilon)
    assert torch.allclose(analytic, measured, rtol=1e-5, atol=1e-6)


def test_identity_has_finite_zero_qc_tail_vjp() -> None:
    line = torch.linspace(0.0, 1.0, 9, dtype=torch.float32)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    control = torch.stack((xx, yy), dim=-1)[None].requires_grad_()
    value = structured_p1_qc_tail_penalty(control, 0.6)
    gradient = torch.autograd.grad(value, control)[0]
    assert value.item() == 0.0
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0
