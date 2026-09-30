"""Exact sparse hat basis and match-coordinate gradient for dual P1 layer."""

import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_dual_multilevel_match_safe257 import (
    dual_fit_map, loo_confidence, multilevel_design,
)
from tools.digital_mind_sparse_match_finetune import p1_at_points
from tools.digital_q1_dhr_distill import identity_vertices


def test_multilevel_design_equals_explicit_p1_prolongation():
    torch.manual_seed(7)
    query = torch.cat((.05 + .9 * torch.rand((19, 2), dtype=torch.float64),
                       torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.],
                                     [0., .5], [1., .5], [.5, 0.], [.5, 1.],
                                     [.5, .5], [20.5 / 256, 20.5 / 256]],
                                    dtype=torch.float64)))
    levels = (17, 33, 65)
    controls = [torch.randn((1, 2, s - 2, s - 2), dtype=torch.float64) * .002
                for s in levels]
    nodal = torch.zeros((1, 2, 257, 257), dtype=torch.float64)
    for control in controls:
        nodal = nodal + F.interpolate(F.pad(control, (1, 1, 1, 1)),
                                      size=(257, 257), mode="bilinear",
                                      align_corners=True)
    design = multilevel_design(query, levels)
    vector = torch.cat([c[0].permute(1, 2, 0).reshape(-1, 2)
                        for c in controls], dim=0)
    expected = p1_at_points(nodal.permute(0, 2, 3, 1), query)
    torch.testing.assert_close(design @ vector, expected, rtol=0, atol=1e-12)


def test_dual_fit_safe_map_and_match_vjp():
    baseline = identity_vertices(257, device=torch.device("cpu"))
    torch.manual_seed(9)
    source = (.1 + .8 * torch.rand((12, 2))).requires_grad_(True)
    target = (source.detach() + torch.tensor([.002, -.001])).requires_grad_(True)
    mapped, controls = dual_fit_map(baseline, source, target, ridge=.1)
    assert controls.shape == (5155, 2)
    assert validate_q1_map(mapped.detach(), baseline)["valid"]
    (mapped - baseline).square().mean().backward()
    assert source.grad is not None and target.grad is not None
    assert torch.isfinite(source.grad).all() and torch.isfinite(target.grad).all()
    assert float(target.grad.abs().sum()) > 0


def test_dual_target_vjp_matches_finite_difference():
    baseline = identity_vertices(257, device=torch.device("cpu")).double()
    torch.manual_seed(11)
    source = .1 + .8 * torch.rand((9, 2), dtype=torch.float64)
    target = (source + torch.tensor([.001, -.0007], dtype=torch.float64)).requires_grad_(True)

    def objective(candidate):
        mapped, _ = dual_fit_map(baseline, source, candidate, ridge=.1)
        return (mapped - baseline).square().sum()

    derivative = torch.autograd.grad(objective(target), target)[0][0, 0]
    delta = torch.zeros_like(target)
    delta[0, 0] = 1e-5
    numerical = (objective(target.detach() + delta) -
                 objective(target.detach() - delta)) / (2e-5)
    torch.testing.assert_close(derivative, numerical, rtol=.03, atol=2e-6)


def test_dual_rejects_nonfinite_target():
    baseline = identity_vertices(257, device=torch.device("cpu"))
    source = torch.full((8, 2), .5)
    target = source.clone()
    target[0, 0] = float("nan")
    try:
        dual_fit_map(baseline, source, target, ridge=.1)
    except ValueError as error:
        assert "finite target" in str(error)
    else:
        raise AssertionError("nonfinite target unexpectedly accepted")


def test_loo_identity_and_weighted_safe_vjp():
    torch.manual_seed(17)
    design = torch.randn((7, 11), dtype=torch.float64)
    desired = torch.randn((7, 2), dtype=torch.float64) * .001
    ridge = .3
    gram = design @ design.T
    weights, error_px = loo_confidence(
        gram, desired, ridge=ridge, scale_px=4.)
    for i in range(len(design)):
        keep = torch.arange(len(design)) != i
        left = design[keep].T @ design[keep] + ridge * torch.eye(
            design.shape[1], dtype=design.dtype)
        coefficient = torch.linalg.solve(left, design[keep].T @ desired[keep])
        direct_error = 512 * torch.linalg.vector_norm(
            desired[i] - design[i] @ coefficient)
        torch.testing.assert_close(error_px[i], direct_error, rtol=1e-10, atol=1e-10)
    assert bool(((weights > 0) & (weights <= 1)).all())

    baseline = identity_vertices(257, device=torch.device("cpu"))
    source = .1 + .8 * torch.rand((9, 2))
    target = (source + torch.tensor([.002, -.001])).requires_grad_(True)
    mapped, _ = dual_fit_map(
        baseline, source, target, ridge=.1, loo_scale_px=4.)
    assert validate_q1_map(mapped.detach(), baseline)["valid"]
    (mapped - baseline).square().mean().backward()
    assert target.grad is not None and bool(torch.isfinite(target.grad).all())


if __name__ == "__main__":
    test_multilevel_design_equals_explicit_p1_prolongation()
    test_dual_fit_safe_map_and_match_vjp()
    test_dual_target_vjp_matches_finite_difference()
    test_dual_rejects_nonfinite_target()
    test_loo_identity_and_weighted_safe_vjp()
    print("five_passed")
