from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest
import torch

from qcopt.mesh import TriMesh
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer


def _mvc_module():
    name = "qcopt.neural_bijection.tutte.mvc"
    assert importlib.util.find_spec(name) is not None, "Route-II MVC layer is missing"
    return importlib.import_module(name)


def _variable_degree_disk() -> TriMesh:
    """Square fan with a degree-five center and degree-three inserted vertex."""
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.15],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 5],
            [1, 4, 5],
            [4, 0, 5],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int64,
    )
    return TriMesh(vertices, faces)


def _convex_boundary(mesh: TriMesh, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    count = len(mesh.boundary_loops[0])
    angles = torch.arange(count, dtype=dtype) * (2.0 * torch.pi / count)
    return torch.stack((1.3 * torch.cos(angles), 0.8 * torch.sin(angles)), dim=-1)


def test_mvc_encoder_uses_oriented_cycles_but_returns_existing_padded_slot_layout() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    encoder = mvc.MeanValueCoordinateEncoder(mesh)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)

    encoded = encoder(vertices)

    assert encoded.logits.shape == (2, 5)
    assert encoded.probabilities.shape == (2, 5)
    np.testing.assert_array_equal(encoder.valid_mask.sum(axis=1), [5, 3])
    assert np.all(encoder.cyclic_slots[0] >= 0)
    assert np.all(encoder.cyclic_slots[1, :3] >= 0)
    assert np.all(encoder.cyclic_slots[1, 3:] == -1)
    assert bool(torch.all(encoded.probabilities[encoder.valid_mask] > 0.0))
    torch.testing.assert_close(
        encoded.probabilities.masked_select(
            ~torch.as_tensor(encoder.valid_mask)
        ),
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        encoded.logits.masked_select(~torch.as_tensor(encoder.valid_mask)),
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(encoded.probabilities.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    represented = torch.softmax(
        encoded.logits.masked_fill(~torch.as_tensor(encoder.valid_mask), -torch.inf), dim=-1
    )
    torch.testing.assert_close(represented, encoded.probabilities, atol=2e-16, rtol=2e-15)
    supported_mean = (
        encoded.logits * torch.as_tensor(encoder.valid_mask, dtype=torch.float64)
    ).sum(dim=-1) / torch.as_tensor(encoder.valid_mask.sum(axis=1), dtype=torch.float64)
    torch.testing.assert_close(supported_mean, torch.zeros_like(supported_mean), atol=2e-15, rtol=0.0)

    neighbors = torch.as_tensor(encoder.neighbor_vertices)
    reconstructed = (
        encoded.probabilities[..., None]
        * vertices[neighbors.clamp_min(0)]
        * torch.as_tensor(encoder.valid_mask)[..., None]
    ).sum(dim=1)
    torch.testing.assert_close(
        reconstructed,
        vertices[torch.as_tensor(encoder.interior_vertices)],
        atol=3e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        encoded.boundary,
        vertices[torch.as_tensor(mesh.boundary_loops[0].copy())],
        atol=0.0,
        rtol=0.0,
    )
    diagnostics = encoded.diagnostics
    assert diagnostics.minimum_edge_length.shape == (2,)
    assert diagnostics.minimum_angle_sine.shape == (2,)
    assert diagnostics.winding_error.shape == (2,)
    assert diagnostics.barycentric_residual.shape == (2,)
    assert diagnostics.covariance_condition.shape == (2,)
    assert bool(torch.all(diagnostics.minimum_edge_length > 0.0))
    assert bool(torch.all(diagnostics.minimum_angle_sine > 0.0))
    assert float(diagnostics.winding_error.max()) < 2.0e-15
    assert float(diagnostics.barycentric_residual.max()) < 3.0e-16
    assert bool(torch.all(torch.isfinite(diagnostics.covariance_condition)))
    assert bool(torch.all(diagnostics.covariance_condition >= 1.0))


def test_mvc_encode_decode_round_trip_is_machine_precision_on_nonsymmetric_map() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    solver = DirectTutteLayer(mesh)
    encoder = mvc.MeanValueCoordinateEncoder(mesh)
    generator = torch.Generator().manual_seed(5203)
    logits = 0.7 * torch.randn(
        solver.system.n_rows,
        solver.system.max_degree,
        generator=generator,
        dtype=torch.float64,
    )
    boundary = _convex_boundary(mesh)
    mapped = solver(logits, boundary)

    encoded = encoder(mapped)
    recovered = solver(encoded.logits, encoded.boundary)

    torch.testing.assert_close(recovered, mapped, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(encoded.boundary, boundary, atol=0.0, rtol=0.0)


def test_canonicalization_changes_redundant_logits_without_changing_decoded_map() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    solver = DirectTutteLayer(mesh)
    layer = mvc.MVCCanonicalizationLayer(mesh, solver)
    generator = torch.Generator().manual_seed(7301)
    logits = 0.8 * torch.randn(
        solver.system.n_rows,
        solver.system.max_degree,
        generator=generator,
        dtype=torch.float64,
    )
    boundary = _convex_boundary(mesh)

    result = layer(logits, boundary)
    canonical_map = solver(result.logits, result.boundary)

    torch.testing.assert_close(canonical_map, result.control, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(result.boundary, boundary, atol=0.0, rtol=0.0)
    original_probabilities = torch.softmax(
        logits.masked_fill(~torch.as_tensor(layer.encoder.valid_mask), -torch.inf), dim=-1
    )
    assert float(
        torch.max(
            torch.abs(
                original_probabilities[0, :5] - result.probabilities[0, :5]
            )
        )
    ) > 1.0e-3


def test_mvc_encoder_batches_maps_and_preserves_each_boundary_exactly() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    encoder = mvc.MeanValueCoordinateEncoder(mesh)
    first = torch.tensor(mesh.vertices, dtype=torch.float64)
    affine = torch.tensor([[1.2, 0.15], [0.05, 0.9]], dtype=torch.float64)
    second = first @ affine.T + torch.tensor([2.0, -3.0], dtype=torch.float64)

    encoded = encoder(torch.stack((first, second)))

    assert encoded.logits.shape == (2, 2, 5)
    expected_boundary = torch.stack((first, second))[:, torch.as_tensor(mesh.boundary_loops[0].copy())]
    torch.testing.assert_close(encoded.boundary, expected_boundary, atol=0.0, rtol=0.0)


def test_mvc_encoder_passes_gradcheck_and_jvp_matches_centered_difference() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    encoder = mvc.MeanValueCoordinateEncoder(mesh)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda value: encoder(value).logits,
        (vertices,),
        eps=1.0e-6,
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    direction = torch.arange(vertices.numel(), dtype=torch.float64).reshape_as(vertices).cos() / 20.0
    _, tangent = torch.autograd.functional.jvp(
        lambda value: encoder(value).logits,
        (vertices,),
        (direction,),
    )
    epsilon = 1.0e-6
    difference = (
        encoder(vertices.detach() + epsilon * direction).logits
        - encoder(vertices.detach() - epsilon * direction).logits
    ) / (2.0 * epsilon)
    torch.testing.assert_close(tangent, difference, atol=2.0e-8, rtol=2.0e-7)


def test_canonicalization_layer_passes_first_order_gradcheck() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    solver = DirectTutteLayer(mesh)
    layer = mvc.MVCCanonicalizationLayer(mesh, solver)
    logits = torch.linspace(
        -0.3,
        0.4,
        solver.system.n_rows * solver.system.max_degree,
        dtype=torch.float64,
    ).reshape(solver.system.n_rows, solver.system.max_degree).requires_grad_()
    boundary = _convex_boundary(mesh).requires_grad_()

    assert torch.autograd.gradcheck(
        lambda latent, fixed_boundary: layer(latent, fixed_boundary).logits,
        (logits, boundary),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("nonfinite", "finite"),
        ("zero_edge", "edge"),
        ("wrong_orientation", "angle|orientation"),
        ("near_pi", "angle"),
    ],
)
def test_mvc_encoder_fails_closed_outside_its_local_geometric_domain(case: str, message: str) -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    encoder = mvc.MeanValueCoordinateEncoder(mesh)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
    if case == "nonfinite":
        vertices[4, 0] = torch.nan
    elif case == "zero_edge":
        vertices[4] = vertices[0]
    elif case == "wrong_orientation":
        vertices[:, 0] *= -1.0
    elif case == "near_pi":
        vertices[5, 1] = 1.0e-16
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    with pytest.raises(ValueError, match=message):
        encoder(vertices)


def test_mvc_encoder_rank_screen_rejects_an_extremely_flat_but_oriented_star() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    encoder = mvc.MeanValueCoordinateEncoder(
        mesh,
        angle_sine_tolerance=0.0,
        covariance_rank_tolerance=1.0e-5,
    )
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
    vertices[:, 1] *= 1.0e-4

    with pytest.raises(ValueError, match="rank|covariance"):
        encoder(vertices)


@pytest.mark.parametrize("case", ["shape", "dtype", "empty_batch"])
def test_mvc_encoder_rejects_invalid_tensor_contracts(case: str) -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    encoder = mvc.MeanValueCoordinateEncoder(mesh)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
    if case == "shape":
        vertices = vertices[:-1]
    elif case == "dtype":
        vertices = vertices.to(torch.int64)
    elif case == "empty_batch":
        vertices = vertices.unsqueeze(0)[:0]
    with pytest.raises((TypeError, ValueError)):
        encoder(vertices)


def test_mvc_encoder_handles_a_disk_with_no_interior_vertices() -> None:
    mvc = _mvc_module()
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 1.0]], dtype=np.float64)
    mesh = TriMesh(vertices, np.array([[0, 1, 2]], dtype=np.int64))
    encoder = mvc.MeanValueCoordinateEncoder(mesh)

    encoded = encoder(torch.tensor(vertices, dtype=torch.float64))

    assert encoded.logits.shape == (0, 0)
    assert encoded.probabilities.shape == (0, 0)
    torch.testing.assert_close(encoded.boundary, torch.tensor(vertices, dtype=torch.float64))


def test_canonicalization_constructor_rejects_a_solver_for_another_mesh() -> None:
    mvc = _mvc_module()
    mesh = _variable_degree_disk()
    vertices = mesh.vertices.copy()
    vertices[5, 0] += 0.01
    other = TriMesh(vertices, mesh.faces.copy())

    with pytest.raises(ValueError, match="mesh"):
        mvc.MVCCanonicalizationLayer(mesh, DirectTutteLayer(other))
