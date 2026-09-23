"""Initial identity homotopy metric-cap checks on every original P1 face."""

import torch

from qcopt.neural_bijection.dense.qc_initial_homotopy import cap_initial_map


def _face_mu_and_det(mapped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = 1.0 / (mapped.shape[1] - 1)
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    x = torch.stack(((b-a)/h, (c-d)/h), dim=1)
    y = torch.stack(((c-b)/h, (d-a)/h), dim=1)
    alpha = torch.complex(0.5*(x[..., 0]+y[..., 1]),
                          0.5*(x[..., 1]-y[..., 0]))
    beta = torch.complex(0.5*(x[..., 0]-y[..., 1]),
                         0.5*(x[..., 1]+y[..., 0]))
    det = x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    return (beta/alpha).abs(), det


def _distorted_map(side: int, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(12345)
    line = torch.linspace(0, 1, side, dtype=dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    mapped = torch.stack((xx, yy), dim=-1)[None]
    mapped[:, 1:-1, 1:-1] += 0.20*torch.randn(1, side-2, side-2, 2, dtype=dtype)
    return mapped


def test_strongly_distorted_input_receives_strict_face_cap_and_fixed_boundary():
    original = _distorted_map(9)
    capped = cap_initial_map(original, qc_cap=0.8)
    original_mu, _ = _face_mu_and_det(original)
    capped_mu, capped_det = _face_mu_and_det(capped)
    assert original_mu.amax() > 0.8
    assert capped_mu.amax() < 0.8
    assert capped_det.amin() > 0
    assert torch.equal(capped[:, 0], original[:, 0])
    assert torch.equal(capped[:, -1], original[:, -1])
    assert torch.equal(capped[:, :, 0], original[:, :, 0])
    assert torch.equal(capped[:, :, -1], original[:, :, -1])


def test_already_eligible_input_is_identical():
    line = torch.linspace(0, 1, 9, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    assert torch.equal(cap_initial_map(identity), identity)


def test_homotopy_vjp_matches_directional_difference():
    original = _distorted_map(9).requires_grad_()
    torch.manual_seed(546)
    direction = torch.randn_like(original)
    direction[:, 0] = 0
    direction[:, -1] = 0
    direction[:, :, 0] = 0
    direction[:, :, -1] = 0
    weights = torch.randn_like(original)

    def objective(value):
        return (cap_initial_map(value)*weights).sum()

    vjp = torch.autograd.grad(objective(original), original)[0]
    step = 1e-6
    numerical = (objective(original.detach()+step*direction)-
                 objective(original.detach()-step*direction))/(2*step)
    analytical = (vjp*direction).sum()
    assert torch.isfinite(vjp).all()
    assert torch.allclose(analytical, numerical, atol=1e-7, rtol=1e-6)
