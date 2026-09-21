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
