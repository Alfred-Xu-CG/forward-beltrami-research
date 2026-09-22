"""Exact multilevel P1-homeomorphism composition and first-order gradient."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense import CoarseFineConvexQuadComposition, certify_convex_quad_output


def _zero_levels(layer, dtype: torch.dtype):
    return tuple(
        (torch.zeros(1, n, n - 1, dtype=dtype),
         torch.zeros(1, n - 1, n, dtype=dtype),
         torch.zeros(1, n - 1, n - 1, 2, dtype=dtype))
        for n in layer.latent_sides
    )


def test_coarse_fine_identity_and_factor_topology() -> None:
    layer = CoarseFineConvexQuadComposition(5, 9, 17)
    layer.prepare(device="cpu", dtype=torch.float64)
    coarse_root = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    fine_root = torch.zeros_like(coarse_root)
    result = layer(coarse_root, _zero_levels(layer.coarse, torch.float64), fine_root, _zero_levels(layer.fine, torch.float64))
    line = torch.linspace(0, 1, 17, dtype=torch.float64)
    y, x = torch.meshgrid(line, line, indexing="ij")
    identity = torch.stack((x, y), dim=-1)[None]
    assert torch.max(torch.abs(result.dense - identity)) < 1e-14
    assert all(certify_convex_quad_output(control) > 0 for control in result.controls)


def test_coarse_fine_root_vjp_matches_directional_difference() -> None:
    torch.manual_seed(9871)
    layer = CoarseFineConvexQuadComposition(5, 9, 18)
    layer.prepare(device="cpu", dtype=torch.float64)
    coarse_root = torch.tensor([[[[0.23, -0.19]]]], dtype=torch.float64, requires_grad=True)
    fine_root = torch.tensor([[[[-0.16, 0.28]]]], dtype=torch.float64, requires_grad=True)
    coarse_latents = _zero_levels(layer.coarse, torch.float64)
    fine_latents = _zero_levels(layer.fine, torch.float64)
    cotangent = torch.randn(1, 18, 18, 2, dtype=torch.float64)

    def objective(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (layer(first, coarse_latents, second, fine_latents).dense * cotangent).sum()

    gradients = torch.autograd.grad(objective(coarse_root, fine_root), (coarse_root, fine_root))
    directions = (torch.randn_like(coarse_root), torch.randn_like(fine_root))
    predicted = sum((g * d).sum() for g, d in zip(gradients, directions))
    step = 1e-6
    observed = (
        objective(coarse_root + step * directions[0], fine_root + step * directions[1])
        - objective(coarse_root - step * directions[0], fine_root - step * directions[1])
    ) / (2 * step)
    assert torch.allclose(predicted, observed, rtol=1e-5, atol=1e-6)
