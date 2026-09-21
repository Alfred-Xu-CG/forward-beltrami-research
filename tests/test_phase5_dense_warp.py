from __future__ import annotations

import numpy as np
import pytest
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _dense_identity(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def test_precomputed_dense_query_separates_control_and_image_resolution() -> None:
    mesh = structured_rectangle(10, 10)  # 11 x 11 control vertices
    table = StructuredDenseQueryTable.from_mesh(mesh, height=256, width=256)

    dense = table.interpolate(torch.tensor(mesh.vertices.copy(), dtype=torch.float64))

    assert table.control_vertices == 121
    assert table.query_count == 256 * 256
    assert dense.shape == (256, 256, 2)
    torch.testing.assert_close(dense, _dense_identity(256, 256), atol=2e-14, rtol=0.0)


def test_precomputed_dense_query_is_batched_differentiable_and_affine_exact() -> None:
    mesh = structured_rectangle(4, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=19, width=23)
    source = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    matrix = torch.tensor(((1.2, 0.1), (-0.2, 0.8)), dtype=torch.float64)
    offset = torch.tensor((0.3, -0.1), dtype=torch.float64)
    affine = source @ matrix.T + offset
    vertices = torch.stack((source, affine)).requires_grad_(True)

    dense = table.interpolate(vertices)
    expected_identity = _dense_identity(19, 23)
    expected_affine = expected_identity @ matrix.T + offset

    assert dense.shape == (2, 19, 23, 2)
    torch.testing.assert_close(dense[0], expected_identity, atol=2e-14, rtol=0.0)
    torch.testing.assert_close(dense[1], expected_affine, atol=2e-14, rtol=0.0)
    dense.square().mean().backward()
    assert vertices.grad is not None
    assert bool(torch.isfinite(vertices.grad).all())
    assert float(torch.linalg.vector_norm(vertices.grad)) > 0.0


def test_precomputed_dense_query_barycentric_rows_are_convex() -> None:
    mesh = structured_rectangle(6, 5)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=37, width=41)

    barycentric = table.barycentric_numpy

    np.testing.assert_allclose(barycentric.sum(axis=1), 1.0, atol=1e-15)
    assert float(np.min(barycentric)) >= 0.0
    assert float(np.max(barycentric)) <= 1.0


def test_nondefault_dtype_requires_explicit_preparation_before_forward() -> None:
    mesh = structured_rectangle(3, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=16, width=16)
    vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32)

    with pytest.raises(ValueError, match="prepare"):
        table.interpolate(vertices)

    table.prepare(device=vertices.device, dtype=vertices.dtype)
    dense = table.interpolate(vertices)
    assert dense.dtype == torch.float32


def test_prepare_normalizes_indexed_cpu_alias_to_actual_tensor_device() -> None:
    mesh = structured_rectangle(3, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=16, width=16)
    vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32, device="cpu")

    table.prepare(device="cpu:0", dtype=vertices.dtype)

    dense = table.interpolate(vertices)
    assert dense.device == vertices.device
    assert dense.dtype == vertices.dtype


def test_dynamic_structured_interpolation_is_affine_exact_and_differentiable() -> None:
    mesh = structured_rectangle(5, 4)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=17, width=19)
    source = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    matrix = torch.tensor(((0.82, 0.11), (-0.07, 1.13)), dtype=torch.float64)
    offset = torch.tensor((0.04, -0.03), dtype=torch.float64)
    control = (source @ matrix.T + offset).requires_grad_()
    query = torch.tensor(
        ((0.13, 0.22), (0.71, 0.38), (0.47, 0.83), (0.999, 0.001)),
        dtype=torch.float64,
        requires_grad=True,
    )

    result = table.interpolate_points(control, query)
    expected = query @ matrix.T + offset

    torch.testing.assert_close(result, expected, atol=3e-15, rtol=0.0)
    assert torch.autograd.gradcheck(
        table.interpolate_points,
        (control, query),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-6,
    )


def test_dynamic_interpolation_supports_batch_broadcast_and_composition() -> None:
    mesh = structured_rectangle(4, 4)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=11, width=13)
    source = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    first = torch.stack((source, 0.8 * source + 0.1))
    query = torch.tensor(((0.12, 0.34), (0.56, 0.78)), dtype=torch.float64)

    after_first = table.interpolate_points(first, query)
    after_second = table.interpolate_points(source, after_first)

    assert after_first.shape == (2, 2, 2)
    torch.testing.assert_close(after_second, after_first, atol=2e-15, rtol=0.0)
    torch.testing.assert_close(after_first[0], query, atol=2e-15, rtol=0.0)
    torch.testing.assert_close(after_first[1], 0.8 * query + 0.1, atol=2e-15, rtol=0.0)


@pytest.mark.parametrize("kind", ("outside", "nan"))
def test_dynamic_interpolation_rejects_invalid_query_coordinates(kind: str) -> None:
    mesh = structured_rectangle(3, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=9, width=9)
    control = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    query = torch.tensor(((0.2, 0.4), (0.7, 0.8)), dtype=torch.float64)
    if kind == "outside":
        query[0, 0] = 1.01
    else:
        query[1, 1] = torch.nan

    with pytest.raises(ValueError, match="unit square|finite"):
        table.interpolate_points(control, query)


def test_dynamic_interpolation_rejects_unvalidated_low_precision() -> None:
    mesh = structured_rectangle(3, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=9, width=9)
    control = torch.tensor(mesh.vertices.copy(), dtype=torch.float16)
    query = torch.tensor(((0.2, 0.4), (0.7, 0.8)), dtype=torch.float16)

    with pytest.raises(ValueError, match="float32 or float64"):
        table.interpolate_points(control, query)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_dynamic_interpolation_only_clamps_roundoff_sized_exterior_queries(dtype) -> None:
    mesh = structured_rectangle(3, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=9, width=9)
    control = torch.tensor(mesh.vertices.copy(), dtype=dtype)
    tolerance = 64.0 * torch.finfo(dtype).eps
    roundoff_query = torch.tensor(((1.0 + 0.5 * tolerance, 0.4),), dtype=dtype)
    invalid_query = torch.tensor(((1.0 + 2.0 * tolerance, 0.4),), dtype=dtype)

    accepted = table.interpolate_points(control, roundoff_query)

    torch.testing.assert_close(
        accepted,
        torch.tensor(((1.0, 0.4),), dtype=dtype),
        atol=4.0 * torch.finfo(dtype).eps,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match="unit square"):
        table.interpolate_points(control, invalid_query)
