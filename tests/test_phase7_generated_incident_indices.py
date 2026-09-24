"""Generated incident-face indexing must preserve the safe P1 update."""

import pytest
import torch

from qcopt.neural_bijection.dense.colored_vertex_relaxation import (
    SafeColoredVertexRelaxation,
)


@pytest.mark.parametrize("side", [9, 10])
def test_generated_indices_match_buffered_output_and_vjp(side: int) -> None:
    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    logits = (1.2 * torch.randn(
        1, side - 2, side - 2, 2,
        dtype=torch.float64, generator=torch.Generator().manual_seed(713),
    )).requires_grad_()
    buffered = SafeColoredVertexRelaxation(
        side, motion_mode="radial", raw_span=2., index_mode="buffered",
    )
    generated = SafeColoredVertexRelaxation(
        side, motion_mode="radial", raw_span=2., index_mode="generated",
    )

    output_buffered = buffered(base, logits)
    output_generated = generated(base, logits)
    assert torch.equal(output_buffered, output_generated)
    gradient_buffered = torch.autograd.grad(output_buffered.square().sum(), logits)[0]
    gradient_generated = torch.autograd.grad(output_generated.square().sum(), logits)[0]
    assert torch.equal(gradient_buffered, gradient_generated)
    stored_buffered = sum(t.numel() * t.element_size() for t in buffered.buffers())
    stored_generated = sum(t.numel() * t.element_size() for t in generated.buffers())
    assert stored_generated < stored_buffered


@pytest.mark.parametrize("index_mode", ["buffered", "generated"])
@pytest.mark.parametrize("motion_mode", ["radial", "disk"])
def test_color_checkpoint_matches_direct_vjp(
    index_mode: str, motion_mode: str,
) -> None:
    side = 11
    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None].requires_grad_()
    logits = torch.randn(
        1, side - 2, side - 2, 2, dtype=torch.float64,
        generator=torch.Generator().manual_seed(98),
    ).requires_grad_()
    floor = base.new_tensor([0.001]) if motion_mode == "radial" else None
    direct = SafeColoredVertexRelaxation(
        side, motion_mode=motion_mode, index_mode=index_mode,
    )
    recompute = SafeColoredVertexRelaxation(
        side, motion_mode=motion_mode, index_mode=index_mode,
        checkpoint_colors=True,
    )
    output_direct = direct(base, logits, area_floor=floor)
    output_recompute = recompute(base, logits, area_floor=floor)
    assert torch.equal(output_direct, output_recompute)
    weights = torch.randn(
        output_direct.shape, dtype=torch.float64,
        generator=torch.Generator().manual_seed(101),
    )
    gradients_direct = torch.autograd.grad(
        (output_direct * weights).sum(), (base, logits), retain_graph=False,
    )
    gradients_recompute = torch.autograd.grad(
        (output_recompute * weights).sum(), (base, logits), retain_graph=False,
    )
    for actual, expected in zip(gradients_recompute, gradients_direct):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
