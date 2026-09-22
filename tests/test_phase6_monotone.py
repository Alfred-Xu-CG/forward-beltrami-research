"""Mathematical and VJP checks for the solve-free fixed-grid decoder."""

import torch

from qcopt.neural_bijection.dense import DenseMonotoneGridLayer


def _signed_twice_area(control: torch.Tensor) -> torch.Tensor:
    a = control[:-1, :-1]
    b = control[:-1, 1:]
    c = control[1:, 1:]
    d = control[1:, :-1]
    first = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
    first -= (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    second = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1])
    second -= (c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
    return torch.stack((first, second), dim=-1)


def test_both_axes_give_strictly_positive_fixed_grid_faces() -> None:
    torch.manual_seed(17)
    side = 17
    for axis in ("vertical", "horizontal"):
        layer = DenseMonotoneGridLayer(side, axis=axis)
        global_logits = 2.0 * torch.randn(3, side - 1, dtype=torch.float64)
        line_logits = 2.0 * torch.randn(3, side, side - 1, dtype=torch.float64)
        control = layer(global_logits, line_logits)
        assert control.shape == (3, side, side, 2)
        assert torch.all(_signed_twice_area(control[0]) > 0.0)
        assert torch.all(control[:, 0, :, 1] == 0.0)
        assert torch.all(control[:, -1, :, 1] == 1.0)
        assert torch.all(control[:, :, 0, 0] == 0.0)
        assert torch.all(control[:, :, -1, 0] == 1.0)


def test_extreme_finite_logits_remain_valid_in_float32() -> None:
    side = 257
    layer = DenseMonotoneGridLayer(side)
    global_logits = torch.full((side - 1,), -80.0, dtype=torch.float32)
    global_logits[side // 2] = 80.0
    line_logits = torch.full((side, side - 1), -80.0, dtype=torch.float32)
    line_logits[:, side // 2] = 80.0
    control = layer(global_logits, line_logits)
    assert control.shape == (side, side, 2)
    assert torch.all(_signed_twice_area(control) > 0.0)


def test_backward_matches_directional_finite_difference() -> None:
    torch.manual_seed(29)
    side = 5
    layer = DenseMonotoneGridLayer(side, axis="horizontal")
    global_logits = torch.randn(side - 1, dtype=torch.float64, requires_grad=True)
    line_logits = torch.randn(side, side - 1, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(side, side, 2, dtype=torch.float64)
    loss = (layer(global_logits, line_logits) * cotangent).sum()
    gradients = torch.autograd.grad(loss, (global_logits, line_logits))
    direction_x = torch.randn_like(global_logits)
    direction_y = torch.randn_like(line_logits)
    predicted = (gradients[0] * direction_x).sum() + (gradients[1] * direction_y).sum()
    step = 1.0e-6
    plus = (layer(global_logits + step * direction_x, line_logits + step * direction_y) * cotangent).sum()
    minus = (layer(global_logits - step * direction_x, line_logits - step * direction_y) * cotangent).sum()
    actual = (plus - minus) / (2.0 * step)
    torch.testing.assert_close(predicted, actual, rtol=1.0e-7, atol=1.0e-8)
