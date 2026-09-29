"""Exact sparse hat basis and match-coordinate gradient for dual P1 layer."""

import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_dual_multilevel_match_safe257 import (
    dual_fit_map, multilevel_design,
)
from tools.digital_mind_sparse_match_finetune import p1_at_points
from tools.digital_q1_dhr_distill import identity_vertices


def test_multilevel_design_equals_explicit_p1_prolongation():
    torch.manual_seed(7)
    query = .05 + .9 * torch.rand((19, 2), dtype=torch.float64)
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


if __name__ == "__main__":
    test_multilevel_design_equals_explicit_p1_prolongation()
    test_dual_fit_safe_map_and_match_vjp()
    print("two_passed")
