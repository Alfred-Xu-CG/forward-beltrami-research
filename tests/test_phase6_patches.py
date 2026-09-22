"""Topology and differentiability of local patch layers and their composition."""

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import LocalPatchComposition, LocalPatchMonotoneLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _face_areas(control: torch.Tensor) -> torch.Tensor:
    a = control[:-1, :-1]
    b = control[:-1, 1:]
    c = control[1:, 1:]
    d = control[1:, :-1]
    first = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
    first -= (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    second = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1])
    second -= (c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
    return torch.stack((first, second), dim=-1)


def test_shifted_patch_layers_preserve_fixed_grid_topology() -> None:
    torch.manual_seed(41)
    side = 33
    line = torch.linspace(0.0, 1.0, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)
    for axis, offset in (("vertical", 0), ("horizontal", 2)):
        layer = LocalPatchMonotoneLayer(
            side, 4, offset_x=offset, offset_y=offset, axis=axis
        )
        latent = torch.randn(layer.patch_count, 3, 4, dtype=torch.float64)
        control = layer(latent)
        assert control.shape == (side, side, 2)
        assert torch.all(_face_areas(control) > 0.0)
        torch.testing.assert_close(control[0], identity[0])
        torch.testing.assert_close(control[-1], identity[-1])
        torch.testing.assert_close(control[:, 0], identity[:, 0])
        torch.testing.assert_close(control[:, -1], identity[:, -1])


def test_exact_composition_has_a_vjp_through_both_layers() -> None:
    torch.manual_seed(43)
    side = 17
    layers = (
        LocalPatchMonotoneLayer(side, 4, axis="vertical"),
        LocalPatchMonotoneLayer(side, 4, offset_x=2, offset_y=2, axis="horizontal"),
    )
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(side - 1, side - 1), height=19, width=19
    )
    composition = LocalPatchComposition(layers, table)
    first = torch.randn(1, layers[0].patch_count, 3, 4, dtype=torch.float64, requires_grad=True)
    second = torch.randn(1, layers[1].patch_count, 3, 4, dtype=torch.float64, requires_grad=True)
    result = composition((first, second))
    assert result.dense.shape == (1, 19, 19, 2)
    assert torch.all((result.dense >= 0.0) & (result.dense <= 1.0))
    assert all(torch.all(_face_areas(control[0]) > 0.0) for control in result.controls)
    cotangent = torch.randn_like(result.dense)
    gradient = torch.autograd.grad((result.dense * cotangent).sum(), (first, second))
    assert all(torch.isfinite(value).all() for value in gradient)
    assert all(value.abs().max() > 0.0 for value in gradient)


def test_257_side_extreme_float32_patch_weights_remain_oriented() -> None:
    layer = LocalPatchMonotoneLayer(257, 16, offset_x=8, offset_y=8, axis="horizontal")
    latent = torch.full((layer.patch_count, 15, 16), -80.0, dtype=torch.float32)
    latent[..., 8] = 80.0
    control = layer(latent)
    assert torch.all(_face_areas(control) > 0.0)
