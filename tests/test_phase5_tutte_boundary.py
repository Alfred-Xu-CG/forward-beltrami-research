from __future__ import annotations

import numpy as np
import pytest
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.boundary import StructuredRectangleBoundary


def test_uniform_logits_and_unit_modulus_recover_identity_boundary() -> None:
    mesh = structured_rectangle(4, 3)
    layer = StructuredRectangleBoundary(mesh, minimum_height=0.05)
    logits = torch.zeros(layer.n_segments, dtype=torch.float64)
    raw_modulus = torch.tensor(layer.raw_modulus_for_height(1.0), dtype=torch.float64)

    boundary = layer(logits, raw_modulus)

    expected = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)
    torch.testing.assert_close(boundary, expected, atol=2e-15, rtol=0.0)


@pytest.mark.parametrize("nx,ny", [(1, 3), (4, 3), (5, 1)])
def test_random_logits_preserve_corners_and_strict_oriented_side_order(nx: int, ny: int) -> None:
    mesh = structured_rectangle(nx, ny)
    layer = StructuredRectangleBoundary(mesh, minimum_height=0.03)
    generator = torch.Generator().manual_seed(704 + nx + 10 * ny)
    logits = torch.randn(3, layer.n_segments, generator=generator, dtype=torch.float64)
    raw_modulus = torch.tensor((-0.4, 0.2, 1.1), dtype=torch.float64)

    boundary = layer(logits, raw_modulus)

    for sample in range(3):
        height = float(layer.height(raw_modulus[sample]))
        np.testing.assert_allclose(
            boundary[sample, layer.corner_loop_positions].detach().numpy(),
            ((0.0, 0.0), (1.0, 0.0), (1.0, height), (0.0, height)),
            atol=2e-15,
        )
        for side, positions in enumerate(layer.side_vertex_loop_positions):
            values = boundary[sample, positions]
            coordinate = values[:, 0] if side in (0, 2) else values[:, 1]
            differences = torch.diff(coordinate)
            if side in (0, 1):
                assert bool(torch.all(differences > 0.0))
            else:
                assert bool(torch.all(differences < 0.0))


def test_boundary_logits_and_modulus_pass_double_gradcheck() -> None:
    mesh = structured_rectangle(3, 2)
    layer = StructuredRectangleBoundary(mesh, minimum_height=0.07)
    logits = torch.linspace(-0.4, 0.6, layer.n_segments, dtype=torch.float64).requires_grad_()
    raw_modulus = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(layer, (logits, raw_modulus), eps=1e-6, atol=1e-7, rtol=1e-6)


def test_batched_logits_and_scalar_modulus_broadcast() -> None:
    mesh = structured_rectangle(2, 3)
    layer = StructuredRectangleBoundary(mesh)
    logits = torch.zeros((4, layer.n_segments), dtype=torch.float32)
    raw_modulus = torch.tensor(layer.raw_modulus_for_height(0.8), dtype=torch.float32)

    output = layer(logits, raw_modulus)

    assert output.shape == (4, layer.n_segments, 2)
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output[:, layer.corner_loop_positions[2], 1],
        torch.full((4,), 0.8, dtype=torch.float32),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_represented_height_is_exact_and_broadcasts(dtype) -> None:
    layer = StructuredRectangleBoundary(structured_rectangle(3, 2))
    logits = torch.zeros((3, layer.n_segments), dtype=dtype)

    output = layer.at_height(logits, 1.0)

    assert output.shape == (3, layer.n_segments, 2)
    assert bool(torch.all(output[..., 1].amin(dim=-1) == 0.0))
    assert bool(torch.all(output[..., 1].amax(dim=-1) == 1.0))


def test_fixed_height_rejects_nonpositive_nonfinite_and_mismatched_tensors() -> None:
    layer = StructuredRectangleBoundary(structured_rectangle(2, 2))
    logits = torch.zeros(layer.n_segments, dtype=torch.float32)

    for height in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            layer.at_height(logits, height)
    with pytest.raises(ValueError, match="dtype"):
        layer.at_height(logits, torch.tensor(1.0, dtype=torch.float64))


def test_segment_underflow_and_invalid_inputs_are_rejected() -> None:
    mesh = structured_rectangle(3, 3)
    layer = StructuredRectangleBoundary(mesh)
    logits = torch.zeros(layer.n_segments, dtype=torch.float32)
    logits[0] = -1000.0
    with pytest.raises(ValueError, match="strictly positive"):
        layer(logits, torch.tensor(0.0, dtype=torch.float32))

    with pytest.raises(ValueError, match="finite"):
        layer(torch.full_like(logits, torch.nan), torch.tensor(0.0, dtype=torch.float32))
    with pytest.raises(ValueError, match="dtype"):
        layer(torch.zeros_like(logits), torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        layer(torch.zeros(layer.n_segments - 1), torch.tensor(0.0))


def test_positive_probabilities_that_round_to_duplicate_vertices_are_rejected() -> None:
    mesh = structured_rectangle(3, 3)
    layer = StructuredRectangleBoundary(mesh)
    logits = torch.zeros(layer.n_segments, dtype=torch.float64)
    bottom_edges = layer.side_vertex_loop_positions[0][:-1]
    logits[bottom_edges] = torch.tensor((0.0, -100.0, -100.0), dtype=torch.float64)

    with pytest.raises(ValueError, match="strict.*order|strictly ordered"):
        layer(logits, torch.tensor(0.0, dtype=torch.float64))


def test_large_height_inverse_is_finite_and_stable() -> None:
    layer = StructuredRectangleBoundary(structured_rectangle(2, 2), minimum_height=0.05)

    raw = layer.raw_modulus_for_height(1000.0)

    assert np.isfinite(raw)
    recovered = layer.height(torch.tensor(raw, dtype=torch.float64))
    torch.testing.assert_close(recovered, torch.tensor(1000.0, dtype=torch.float64))


def test_nonfinite_realized_height_is_rejected() -> None:
    layer = StructuredRectangleBoundary(
        structured_rectangle(2, 2), minimum_height=1.0e38
    )
    logits = torch.zeros(layer.n_segments, dtype=torch.float32)
    raw = torch.tensor(3.0e38, dtype=torch.float32)

    with pytest.raises(ValueError, match="height.*finite|finite.*height"):
        layer(logits, raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_registered_side_indices_follow_module_to_cuda() -> None:
    layer = StructuredRectangleBoundary(structured_rectangle(3, 2)).to("cuda")
    logits = torch.zeros(layer.n_segments, dtype=torch.float32, device="cuda")
    raw = torch.tensor(0.0, dtype=torch.float32, device="cuda")

    output = layer(logits, raw)

    assert output.device.type == "cuda"
    assert all(positions.device.type == "cuda" for positions in layer.side_vertex_loop_positions)
