"""Independent checks of structured P1 value and affine-Jacobian evaluation."""

from __future__ import annotations

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import DenseMonotoneGridLayer, evaluate_structured_p1_with_jacobian
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def test_value_matches_existing_dynamic_query_interpolator() -> None:
    torch.manual_seed(31)
    side = 13
    decoder = DenseMonotoneGridLayer(side, axis="horizontal")
    control = decoder(0.2 * torch.randn(2, side - 1, dtype=torch.float64), 0.2 * torch.randn(2, side, side - 1, dtype=torch.float64))
    points = 0.01 + 0.98 * torch.rand(2, 73, 2, dtype=torch.float64)
    evaluated, jacobian = evaluate_structured_p1_with_jacobian(control, points)
    table = StructuredDenseQueryTable.from_mesh(structured_rectangle(side - 1, side - 1), height=8, width=8)
    independent = table.interpolate_points(control.reshape(2, -1, 2), points)
    assert torch.allclose(evaluated, independent, atol=1e-13, rtol=0)
    assert torch.all(torch.linalg.det(jacobian) > 0)


def test_affine_jacobian_matches_local_coordinate_difference() -> None:
    torch.manual_seed(47)
    side = 9
    decoder = DenseMonotoneGridLayer(side, axis="vertical")
    control = decoder(0.3 * torch.randn(1, side - 1, dtype=torch.float64), 0.3 * torch.randn(1, side, side - 1, dtype=torch.float64))
    point = torch.tensor([[[0.37, 0.41]]], dtype=torch.float64)
    _, jacobian = evaluate_structured_p1_with_jacobian(control, point)
    epsilon = 1.0e-6
    for axis in range(2):
        offset = torch.zeros_like(point)
        offset[..., axis] = epsilon
        plus, _ = evaluate_structured_p1_with_jacobian(control, point + offset)
        minus, _ = evaluate_structured_p1_with_jacobian(control, point - offset)
        finite_difference = (plus - minus) / (2 * epsilon)
        assert torch.allclose(finite_difference, jacobian[..., axis], atol=1e-10, rtol=0)
