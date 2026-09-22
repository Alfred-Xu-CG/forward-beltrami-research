"""Topology and VJP checks for full-grid alternating monotone composition."""

from __future__ import annotations

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactAlternatingMonotoneComposition
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _minimum_area_ratio(control: torch.Tensor) -> torch.Tensor:
    a = control[..., :-1, :-1, :]
    b = control[..., :-1, 1:, :]
    c = control[..., 1:, 1:, :]
    d = control[..., 1:, :-1, :]
    first = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    second = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
    return torch.minimum(first.min(), second.min()) * (control.shape[-2] - 1) ** 2


def test_two_full_grid_layers_have_positive_faces_and_boundary() -> None:
    side = 17
    table = StructuredDenseQueryTable.from_mesh(structured_rectangle(side - 1, side - 1), height=32, width=32)
    table.prepare(device="cpu", dtype=torch.float64)
    model = ExactAlternatingMonotoneComposition(side, table)
    torch.manual_seed(20260923)
    latents = tuple(((0.2 * torch.randn(1, side - 1, dtype=torch.float64), 0.2 * torch.randn(1, side, side - 1, dtype=torch.float64)),) for _ in range(2))
    result = model(latents)
    assert result.dense.shape == (1, 32, 32, 2)
    for control in result.controls:
        assert _minimum_area_ratio(control) > 0
        assert torch.all(control >= 0) and torch.all(control <= 1)
    assert torch.allclose(result.dense[:, 0, 0], torch.zeros(1, 2, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(result.dense[:, -1, -1], torch.ones(1, 2, dtype=torch.float64), atol=1e-14)


def test_two_layer_vjp_matches_directional_difference() -> None:
    side = 9
    table = StructuredDenseQueryTable.from_mesh(structured_rectangle(side - 1, side - 1), height=20, width=20)
    table.prepare(device="cpu", dtype=torch.float64)
    model = ExactAlternatingMonotoneComposition(side, table)
    torch.manual_seed(21)
    first_global = (0.3 * torch.randn(1, side - 1, dtype=torch.float64)).requires_grad_()
    first_line = 0.3 * torch.randn(1, side, side - 1, dtype=torch.float64)
    second_global = 0.3 * torch.randn(1, side - 1, dtype=torch.float64)
    second_line = 0.3 * torch.randn(1, side, side - 1, dtype=torch.float64)

    def objective(global_logits: torch.Tensor) -> torch.Tensor:
        result = model((((global_logits, first_line),), ((second_global, second_line),)))
        return result.dense[:, 2:-2, 2:-2].square().mean()

    gradient = torch.autograd.grad(objective(first_global), first_global)[0]
    direction = torch.randn_like(first_global)
    direction = direction / torch.linalg.vector_norm(direction)
    step = 1.0e-5
    finite_difference = (objective(first_global.detach() + step * direction) - objective(first_global.detach() - step * direction)) / (2 * step)
    assert abs(gradient.mul(direction).sum().item() - finite_difference.item()) < 1.0e-7
