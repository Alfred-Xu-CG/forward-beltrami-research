"""Independent Route-II tests for covariance lifts and legal Tutte retractions."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer
from qcopt.neural_bijection.tutte.mvc import MeanValueCoordinateEncoder


@pytest.fixture
def api() -> SimpleNamespace:
    name = "qcopt.neural_bijection.tutte.mvc_retraction"
    assert importlib.util.find_spec(name) is not None, "Route-II covariance retraction is missing"
    module = importlib.import_module(name)
    return SimpleNamespace(
        covariance_logit_lift=module.covariance_logit_lift,
        CovarianceLogitLift=module.CovarianceLogitLift,
        CovarianceTutteRetraction=module.CovarianceTutteRetraction,
    )


def _state(nx: int = 3, ny: int = 3):
    mesh = structured_rectangle(nx, ny)
    decoder = DirectTutteLayer(mesh)
    generator = torch.Generator().manual_seed(9321 + 10 * nx + ny)
    logits = 0.18 * torch.randn(
        decoder.system.n_rows,
        decoder.system.max_degree,
        generator=generator,
        dtype=torch.float64,
    )
    boundary = torch.tensor(
        mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64
    )
    control = decoder(logits, boundary)
    return mesh, decoder, logits, boundary, control


def _global_neighbor_vertices(system) -> np.ndarray:
    result = np.full_like(system.neighbors, -1)
    rows, slots = np.nonzero(system.valid_mask)
    stored = system.neighbors[rows, slots]
    is_boundary = system.neighbor_is_boundary[rows, slots]
    result[rows[is_boundary], slots[is_boundary]] = system.loop[stored[is_boundary]]
    result[rows[~is_boundary], slots[~is_boundary]] = system.interior[stored[~is_boundary]]
    return result


def test_covariance_lift_has_the_derived_sign_and_global_neighbor_coupling(api) -> None:
    _, decoder, logits, _, control = _state()
    system = decoder.system
    direction = torch.zeros_like(control)
    moved_vertex = int(system.interior[1])
    direction[moved_vertex] = torch.tensor((0.07, -0.04), dtype=torch.float64)

    result = api.covariance_logit_lift(system, logits, control, direction)
    mask = torch.from_numpy(system.valid_mask)
    neighbors = _global_neighbor_vertices(system)
    independently_recovered = torch.zeros((system.n_rows, 2), dtype=torch.float64)
    for row in range(system.n_rows):
        valid = mask[row]
        p = result.probabilities[row, valid]
        delta = result.delta_logits[row, valid]
        delta_p = p * (delta - torch.sum(p * delta))
        neighbor_ids = torch.from_numpy(neighbors[row, system.valid_mask[row]])
        independently_recovered[row] = torch.sum(
            delta_p[:, None] * control.index_select(0, neighbor_ids), dim=0
        )
    rhs = direction.index_select(0, torch.from_numpy(system.interior)).clone()
    for row in range(system.n_rows):
        valid_neighbors = torch.from_numpy(neighbors[row, system.valid_mask[row]])
        rhs[row] -= torch.sum(
            result.probabilities[row, mask[row], None]
            * direction.index_select(0, valid_neighbors),
            dim=0,
        )

    torch.testing.assert_close(independently_recovered, rhs, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(result.linearized_residual, torch.zeros_like(rhs), atol=2e-13, rtol=0)
    coupled_rows = [
        row
        for row in range(system.n_rows)
        if int(system.interior[row]) != moved_vertex
        and moved_vertex in neighbors[row, system.valid_mask[row]]
    ]
    assert coupled_rows
    assert all(float(torch.linalg.vector_norm(result.delta_logits[row])) > 0 for row in coupled_rows)
    assert bool(torch.all(result.delta_logits[~mask] == 0))


def test_probability_api_accepts_mvc_encoder_shaped_rows(api) -> None:
    _, decoder, logits, _, control = _state(3, 2)
    system = decoder.system
    mask = torch.from_numpy(system.valid_mask)
    probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    direction = torch.zeros_like(control)
    direction[torch.from_numpy(system.interior)] = torch.tensor(
        ((0.011, -0.017), (-0.023, 0.019)), dtype=torch.float64
    )

    explicit = api.CovarianceLogitLift(system)(control, probabilities, direction)
    from_logits = api.covariance_logit_lift(system, logits, control, direction)
    torch.testing.assert_close(explicit.delta_logits, from_logits.delta_logits)
    torch.testing.assert_close(explicit.barycentric_rhs, from_logits.barycentric_rhs)


def test_softmax_centering_preserves_right_inverse_off_exact_equilibrium(api) -> None:
    """The plan's uncentered r-formula assumes exact barycentric equilibrium."""
    _, decoder, logits, _, control = _state(2, 2)
    system = decoder.system
    mask = torch.from_numpy(system.valid_mask[0])
    probabilities = torch.softmax(logits[0, mask], dim=0)
    represented_probabilities = torch.zeros_like(logits)
    represented_probabilities[0, mask] = probabilities
    perturbed = control.clone()
    perturbed[system.interior[0]] += torch.tensor((0.09, -0.06), dtype=torch.float64)
    direction = torch.zeros_like(control)
    direction[system.interior[0]] = torch.tensor((0.027, 0.019), dtype=torch.float64)

    centered = api.CovarianceLogitLift(system)(
        perturbed, represented_probabilities, direction
    )
    assert float(torch.linalg.vector_norm(centered.equilibrium_residual)) > 0.05
    assert float(torch.linalg.vector_norm(centered.linearized_residual)) < 2e-16

    neighbor_ids = torch.from_numpy(_global_neighbor_vertices(system)[0, system.valid_mask[0]])
    neighbor_points = perturbed.index_select(0, neighbor_ids)
    plan_offsets = neighbor_points - perturbed[system.interior[0]]
    plan_covariance = torch.einsum("j,jd,je->de", probabilities, plan_offsets, plan_offsets)
    rhs = direction[system.interior[0]]
    plan_delta = plan_offsets @ torch.linalg.solve(plan_covariance, rhs)
    plan_delta_p = probabilities * (plan_delta - torch.sum(probabilities * plan_delta))
    plan_recovered = torch.sum(plan_delta_p[:, None] * neighbor_points, dim=0)
    assert float(torch.linalg.vector_norm(plan_recovered - rhs)) > 1e-4


def test_covariance_lift_matches_explicit_global_weighted_pseudoinverse(api) -> None:
    _, decoder, logits, boundary, control = _state(3, 2)
    system = decoder.system
    direction = torch.zeros_like(control)
    direction[torch.from_numpy(system.interior)] = torch.tensor(
        ((0.031, -0.047), (-0.022, 0.019)), dtype=torch.float64
    )
    result = api.covariance_logit_lift(system, logits, control, direction)

    mask = torch.from_numpy(system.valid_mask)
    supported = logits[mask].detach().requires_grad_()

    def decode_supported(values: torch.Tensor) -> torch.Tensor:
        represented = logits.masked_scatter(mask, values)
        decoded = decoder(represented, boundary)
        return decoded.index_select(0, torch.from_numpy(system.interior)).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(decode_supported, supported)
    probabilities = result.probabilities[mask]
    inverse_sqrt_metric = torch.diag(probabilities.rsqrt())
    target = direction.index_select(0, torch.from_numpy(system.interior)).reshape(-1)
    explicit = inverse_sqrt_metric @ torch.linalg.pinv(
        jacobian @ inverse_sqrt_metric
    ) @ target

    actual = result.delta_logits[mask]
    torch.testing.assert_close(actual, explicit, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(jacobian @ actual, target, atol=2e-13, rtol=2e-12)

    ordinary_euclidean = torch.linalg.pinv(jacobian) @ target
    assert float(torch.linalg.vector_norm(actual - ordinary_euclidean)) > 1e-5
    weighted_norm = torch.sum(probabilities * actual.square())
    ordinary_weighted_norm = torch.sum(probabilities * ordinary_euclidean.square())
    assert float(weighted_norm) < float(ordinary_weighted_norm)


def test_retraction_is_decoder_output_and_has_requested_first_variation(api) -> None:
    _, decoder, logits, boundary, control = _state(3, 3)
    system = decoder.system
    generator = torch.Generator().manual_seed(8304)
    direction = torch.zeros_like(control)
    direction[torch.from_numpy(system.interior)] = 0.025 * torch.randn(
        (system.n_rows, 2), generator=generator, dtype=torch.float64
    )
    retraction = api.CovarianceTutteRetraction(decoder)

    at_zero = retraction(logits, boundary, direction, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(at_zero.base_control, control, atol=2e-14, rtol=0)
    torch.testing.assert_close(at_zero.updated_control, control, atol=2e-14, rtol=0)

    epsilon = 2e-5
    plus = retraction(logits, boundary, direction, torch.tensor(epsilon, dtype=torch.float64))
    minus = retraction(logits, boundary, direction, torch.tensor(-epsilon, dtype=torch.float64))
    finite_difference = (plus.updated_control - minus.updated_control) / (2 * epsilon)
    torch.testing.assert_close(finite_difference, direction, atol=8e-10, rtol=2e-8)

    differentiable_alpha = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    differentiable = retraction(logits, boundary, direction, differentiable_alpha)
    cotangent = torch.linspace(
        -0.7, 0.9, differentiable.updated_control.numel(), dtype=torch.float64
    ).reshape_as(differentiable.updated_control)
    alpha_vjp = torch.autograd.grad(
        torch.sum(differentiable.updated_control * cotangent), differentiable_alpha
    )[0]
    torch.testing.assert_close(alpha_vjp, torch.sum(direction * cotangent), atol=3e-13, rtol=2e-12)

    finite_step = retraction(logits, boundary, direction, torch.tensor(0.15, dtype=torch.float64))
    independently_decoded = decoder(finite_step.updated_logits, finite_step.updated_boundary)
    torch.testing.assert_close(finite_step.updated_control, independently_decoded, atol=0, rtol=0)
    euler = finite_step.base_control + 0.15 * direction
    assert float(torch.max(torch.abs(finite_step.updated_control - euler))) > 1e-9


def test_m2_canonicalizes_nontrivially_redundant_positive_rows_before_update(api) -> None:
    mesh, decoder, _, boundary, control = _state(2, 2)
    system = decoder.system
    encoder = MeanValueCoordinateEncoder(mesh)
    encoded = encoder(control)
    mask = torch.from_numpy(system.valid_mask[0])
    probabilities = encoded.probabilities[0, mask]
    neighbor_ids = torch.from_numpy(_global_neighbor_vertices(system)[0, system.valid_mask[0]])
    neighbor_points = control.index_select(0, neighbor_ids)
    affine_constraints = torch.cat(
        (torch.ones((1, len(neighbor_ids)), dtype=torch.float64), neighbor_points.T), dim=0
    )
    _, _, vh = torch.linalg.svd(affine_constraints, full_matrices=True)
    fiber_direction = vh[-1]
    assert float(torch.linalg.vector_norm(affine_constraints @ fiber_direction)) < 1e-13
    admissible = probabilities / torch.clamp(torch.abs(fiber_direction), min=1e-30)
    epsilon = 0.2 * torch.min(admissible[torch.abs(fiber_direction) > 1e-12])
    alternative = probabilities + epsilon * fiber_direction
    assert bool(torch.all(alternative > 0))
    torch.testing.assert_close(affine_constraints @ alternative, affine_constraints @ probabilities)

    first_logits = torch.zeros_like(encoded.logits)
    second_logits = torch.zeros_like(encoded.logits)
    first_logits[0, mask] = torch.log(probabilities)
    second_logits[0, mask] = torch.log(alternative)
    assert float(torch.linalg.vector_norm(first_logits - second_logits)) > 1e-3
    first_map = decoder(first_logits, boundary)
    second_map = decoder(second_logits, boundary)
    torch.testing.assert_close(first_map, second_map, atol=3e-14, rtol=0)

    direction = torch.zeros_like(control)
    direction[system.interior[0]] = torch.tensor((0.02, -0.013), dtype=torch.float64)
    retraction = api.CovarianceTutteRetraction(decoder)
    first = retraction(first_logits, boundary, direction, 0.12)
    second = retraction(second_logits, boundary, direction, 0.12)
    torch.testing.assert_close(first.lift.probabilities, second.lift.probabilities, atol=2e-14, rtol=0)
    torch.testing.assert_close(first.updated_logits, second.updated_logits, atol=3e-14, rtol=0)
    torch.testing.assert_close(first.updated_control, second.updated_control, atol=3e-14, rtol=0)


def test_boundary_tangent_is_explicit_and_translation_uses_no_logit_change(api) -> None:
    _, decoder, logits, boundary, control = _state(3, 2)
    translation = torch.tensor((0.13, -0.08), dtype=torch.float64)
    direction = translation.expand_as(control).clone()

    fixed = api.CovarianceTutteRetraction(decoder)
    with pytest.raises(ValueError, match="fixed boundary"):
        fixed(logits, boundary, direction, torch.tensor(0.1, dtype=torch.float64))

    moving = api.CovarianceTutteRetraction(decoder, allow_boundary_motion=True)
    result = moving(logits, boundary, direction, torch.tensor(0.3, dtype=torch.float64))
    torch.testing.assert_close(result.lift.delta_logits, torch.zeros_like(logits), atol=2e-13, rtol=0)
    torch.testing.assert_close(result.updated_boundary, boundary + 0.3 * translation, atol=2e-15, rtol=0)
    torch.testing.assert_close(result.updated_control, control + 0.3 * translation, atol=3e-13, rtol=0)


@pytest.mark.parametrize("near_rank", [False, True])
def test_covariance_rank_or_conditioning_failure_is_fail_closed(api, near_rank: bool) -> None:
    mesh = structured_rectangle(2, 2)
    decoder = DirectTutteLayer(mesh)
    system = decoder.system
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)
    control = torch.tensor(mesh.vertices, dtype=torch.float64)
    if near_rank:
        control[:, 1] *= 1e-6
    else:
        control[:, 1] = 0.0
    direction = torch.zeros_like(control)

    with pytest.raises(ValueError, match="rank|condition"):
        api.covariance_logit_lift(
            system,
            logits,
            control,
            direction,
            max_covariance_condition=1e8,
        )


def test_retraction_backward_is_finite_for_logits_boundary_direction_and_step(api) -> None:
    _, decoder, logits, boundary, control = _state(3, 2)
    direction = torch.zeros_like(control)
    direction[torch.from_numpy(decoder.system.interior)] = torch.tensor(
        ((0.01, -0.02), (-0.015, 0.012)), dtype=torch.float64
    )
    z = logits.requires_grad_()
    b = boundary.requires_grad_()
    d = direction.requires_grad_()
    alpha = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
    result = api.CovarianceTutteRetraction(decoder)(z, b, d, alpha)
    weights = torch.linspace(0.2, 1.1, result.updated_control.numel(), dtype=torch.float64).reshape_as(
        result.updated_control
    )
    gradients = torch.autograd.grad((result.updated_control.square() * weights).sum(), (z, b, d, alpha))
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_retraction_batches_logits_and_directions_with_shared_boundary(api) -> None:
    _, decoder, logits, boundary, control = _state(3, 2)
    first_direction = torch.zeros_like(control)
    first_direction[torch.from_numpy(decoder.system.interior)] = torch.tensor(
        ((0.012, -0.008), (-0.009, 0.014)), dtype=torch.float64
    )
    directions = torch.stack((first_direction, -0.7 * first_direction))
    batched_logits = torch.stack((logits, 0.6 * logits))
    retraction = api.CovarianceTutteRetraction(decoder)

    batched = retraction(batched_logits, boundary, directions, 0.03)
    separate = [
        retraction(batched_logits[index], boundary, directions[index], 0.03)
        for index in range(2)
    ]
    assert batched.updated_control.shape == (2, decoder.system.n_vertices, 2)
    assert batched.updated_logits.shape == (
        2,
        decoder.system.n_rows,
        decoder.system.max_degree,
    )
    torch.testing.assert_close(
        batched.updated_control, torch.stack([item.updated_control for item in separate])
    )
    torch.testing.assert_close(
        batched.updated_logits, torch.stack([item.updated_logits for item in separate])
    )


@pytest.mark.parametrize("bad", ["nan_logits", "inf_control", "wrong_direction_shape"])
def test_covariance_lift_rejects_nonfinite_or_mismatched_state(api, bad: str) -> None:
    _, decoder, logits, _, control = _state(2, 2)
    direction = torch.zeros_like(control)
    if bad == "nan_logits":
        logits[0, 0] = float("nan")
    elif bad == "inf_control":
        control[0, 0] = float("inf")
    else:
        direction = direction[:-1]
    with pytest.raises((TypeError, ValueError)):
        api.covariance_logit_lift(decoder.system, logits, control, direction)
