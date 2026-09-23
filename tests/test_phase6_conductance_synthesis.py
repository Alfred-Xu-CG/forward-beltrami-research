"""Linearized cancellation, positive output, and end-to-end VJP checks."""

from __future__ import annotations

import math

import torch

from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer, synthesize_bounded_conductances
from qcopt.neural_bijection.dense.sine_pcg_tutte import _laplacian


def _desired(side: int, amplitude: float) -> torch.Tensor:
    line = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    basis = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    return torch.stack((xx + amplitude * basis, yy + 0.7 * amplitude * basis), dim=-1)[None]


def test_one_pass_cancels_first_order_equilibrium_residual() -> None:
    side = 9
    for gauge in ("centered", "range"):
        final_residuals = []
        for amplitude in (0.001, 0.002):
            desired = _desired(side, amplitude)
            logits, stats = synthesize_bounded_conductances(desired, gauge=gauge, return_stats=True)
            assert stats is not None
            assert all(item["clipped_horizontal_fraction"] == 0 for item in stats["passes"])
            weights = tuple(1 + 15 * value.sigmoid() for value in logits)
            corrected = _laplacian(desired, *weights)[:, 1:-1, 1:-1]
            initial = _laplacian(desired, *(torch.full_like(value, 4.0) for value in weights))[:, 1:-1, 1:-1]
            assert torch.linalg.vector_norm(corrected) < 0.05 * torch.linalg.vector_norm(initial)
            final_residuals.append(float(torch.linalg.vector_norm(corrected)))
        assert 3.5 < final_residuals[1] / final_residuals[0] < 4.5


def test_synthesis_and_equilibrium_have_finite_directional_vjp() -> None:
    side = 9
    layer = SinePreconditionedTutteLayer(side, maximum_conductance=16.0, tolerance=1e-12)
    desired = _desired(side, 0.005).requires_grad_()
    torch.manual_seed(6291)
    cotangent = torch.randn_like(desired)
    direction = torch.randn_like(desired)
    direction[:, 0] = 0; direction[:, -1] = 0
    direction[:, :, 0] = 0; direction[:, :, -1] = 0
    def objective(value):
        logits, _ = synthesize_bounded_conductances(value, passes=2)
        return (layer(*logits) * cotangent).sum()
    value = objective(desired)
    gradient = torch.autograd.grad(value, desired)[0]
    analytic = (gradient * direction).sum()
    epsilon = 1e-6
    measured = (objective(desired.detach() + epsilon * direction) - objective(desired.detach() - epsilon * direction)) / (2 * epsilon)
    assert torch.isfinite(gradient).all()
    assert torch.allclose(analytic, measured, rtol=1e-4, atol=1e-6)
