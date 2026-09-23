"""Original-face QC cap and first-order gradient checks for local radial steps."""

from __future__ import annotations

import torch

from phase6_evaluate_heldout_beltrami import _mu
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    SafeColoredVertexRelaxation, evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.dense.qc_radial_relaxation import (
    SafeColoredQCRadialRelaxation, _complex_derivative_norms,
)


def test_complex_derivative_norms_match_direct_beltrami() -> None:
    torch.manual_seed(702)
    jacobian = torch.eye(2, dtype=torch.float64) + 0.2 * torch.randn(100, 2, 2, dtype=torch.float64)
    alpha, beta = _complex_derivative_norms(jacobian)
    assert torch.allclose(beta / alpha, _mu(jacobian).abs(), rtol=1e-13, atol=1e-13)
    assert torch.allclose(alpha.square() - beta.square(), torch.linalg.det(jacobian),
                          rtol=1e-13, atol=1e-13)


def test_qc_radial_all_faces_and_vjp_on_original_grid() -> None:
    torch.manual_seed(2403)
    side, cap = 9, 0.8
    mesh = structured_rectangle(side - 1, side - 1)
    source = torch.tensor(mesh.vertices, dtype=torch.float64).reshape(1, side, side, 2)
    centroids = torch.tensor(mesh.vertices[mesh.faces].mean(axis=1), dtype=torch.float64)[None]
    layer = SafeColoredQCRadialRelaxation(side, qc_cap=cap, safety_fraction=0.9)
    latent = (2 * torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64)).requires_grad_()
    mapped = layer(source, latent)
    _, jacobian = evaluate_structured_p1_with_jacobian(mapped, centroids)
    assert float(_mu(jacobian).abs().max()) < cap
    assert float(torch.linalg.det(jacobian).min()) > 0
    assert torch.equal(mapped[:, 0], source[:, 0])
    assert torch.equal(mapped[:, -1], source[:, -1])
    assert torch.equal(mapped[:, :, 0], source[:, :, 0])
    assert torch.equal(mapped[:, :, -1], source[:, :, -1])
    cotangent = torch.randn_like(mapped)
    direction = torch.randn_like(latent)
    analytic = (torch.autograd.grad((mapped * cotangent).sum(), latent)[0] * direction).sum()
    epsilon = 1e-6
    value = lambda x: (layer(source, x) * cotangent).sum()
    measured = (value(latent.detach() + epsilon * direction)
                - value(latent.detach() - epsilon * direction)) / (2 * epsilon)
    assert torch.allclose(analytic, measured, rtol=1e-5, atol=1e-7)


def test_qc_radial_preserves_absolute_face_area_floor() -> None:
    torch.manual_seed(2404)
    side = 9
    mesh = structured_rectangle(side - 1, side - 1)
    source = torch.tensor(mesh.vertices, dtype=torch.float64).reshape(1, side, side, 2)
    centroids = torch.tensor(mesh.vertices[mesh.faces].mean(axis=1), dtype=torch.float64)[None]
    layer = SafeColoredQCRadialRelaxation(side, qc_cap=0.8, floor_fraction=0.8)
    mapped = layer(source, 5 * torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64))
    _, jacobian = evaluate_structured_p1_with_jacobian(mapped, centroids)
    assert float(_mu(jacobian).abs().max()) < 0.8
    assert float(torch.linalg.det(jacobian).min()) >= 0.8 - 1e-12


def test_qc_radial_zero_step_has_finite_vjp() -> None:
    side = 9
    mesh = structured_rectangle(side - 1, side - 1)
    source = torch.tensor(mesh.vertices, dtype=torch.float64).reshape(1, side, side, 2)
    layer = SafeColoredQCRadialRelaxation(side, qc_cap=0.8, floor_fraction=0.8)
    latent = torch.zeros(1, side - 2, side - 2, 2, dtype=torch.float64, requires_grad=True)
    mapped = layer(source, latent)
    gradient = torch.autograd.grad(mapped.square().sum(), latent)[0]
    assert torch.isfinite(gradient).all()


def test_preexisting_cap_violation_is_not_misreported_as_repaired() -> None:
    torch.manual_seed(2405)
    side = 17
    mesh = structured_rectangle(side - 1, side - 1)
    source = torch.tensor(mesh.vertices, dtype=torch.float64).reshape(1, side, side, 2)
    centroids = torch.tensor(mesh.vertices[mesh.faces].mean(axis=1), dtype=torch.float64)[None]
    base = SafeColoredVertexRelaxation(side, motion_mode="radial")(
        source, 3 * torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64))
    _, initial_jacobian = evaluate_structured_p1_with_jacobian(base, centroids)
    initial_maximum = float(_mu(initial_jacobian).abs().max())
    assert initial_maximum > 0.8
    mapped = SafeColoredQCRadialRelaxation(side, qc_cap=0.8)(
        base, 3 * torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64))
    _, jacobian = evaluate_structured_p1_with_jacobian(mapped, centroids)
    assert float(torch.linalg.det(jacobian).min()) > 0
    assert float(_mu(jacobian).abs().max()) >= 0.8
