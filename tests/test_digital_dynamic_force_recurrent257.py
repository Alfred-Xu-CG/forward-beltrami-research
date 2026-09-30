"""Dynamic evidence is recomputed at each topology-safe decoder pass."""

import pytest
import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices


def test_cell_to_vertex_interpolation_formula():
    torch.manual_seed(21)
    cells = torch.randn((1, 1, 1, 256), dtype=torch.float64)
    observed = F.interpolate(cells, size=(1, 257), mode="bilinear",
                             align_corners=False)[0, 0, 0, 1:-1]
    j = torch.arange(1, 256, dtype=torch.float64)
    a = (j + .5) / 257.
    expected = a * cells[0, 0, 0, :-1] + (1 - a) * cells[0, 0, 0, 1:]
    assert torch.allclose(observed, expected, atol=1e-13, rtol=0)


def test_zero_initialized_direct_vertex_readout_preserves_baseline_and_trains():
    torch.manual_seed(19)
    original = RecurrentDenseSafeHead(width=4, passes=1)
    expanded = RecurrentDenseSafeHead(width=4, passes=1,
                                      direct_vertex_residual=True)
    expanded.load_state_dict(original.state_dict(), strict=False)
    identity = identity_vertices(257, device=torch.device("cpu"))
    feature = torch.randn((1, 88, 256, 256)) * .01
    baseline = original(identity, feature)
    mapped = expanded(identity, feature)
    assert torch.equal(mapped, baseline)
    assert validate_q1_map(mapped, identity)["valid"]
    loss = mapped[0, 128, 128, 0]
    loss.backward()
    gradient = expanded.vertex_heads[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool(gradient.abs().sum() > 0)


def test_dynamic_evidence_four_current_map_calls_and_vjp():
    torch.manual_seed(7)
    model = RecurrentDenseSafeHead(width=4, passes=4, match_channels=6)
    with torch.no_grad():
        for head in model.heads:
            head.weight.fill_(.002)
            head.bias.fill_(.004)
    identity = identity_vertices(257, device=torch.device("cpu"))
    feature = torch.zeros((1, 88, 256, 256))
    cue = torch.ones((1, 6, 256, 256), requires_grad=True)
    seen = []

    def evidence(current):
        seen.append(current)
        return cue * (.5 + current[:, :-1, :-1, :1].permute(0, 3, 1, 2))

    mapped = model(identity, feature, evidence_fn=evidence)
    assert len(seen) == 4
    assert all(torch.isfinite(item).all() for item in seen)
    assert validate_q1_map(mapped, identity)["valid"]
    mapped[:, 128, 128].sum().backward()
    assert cue.grad is not None and torch.isfinite(cue.grad).all()
    assert bool(cue.grad.abs().sum() > 0)


def test_dynamic_evidence_rejects_bad_shape_and_mixed_inputs():
    model = RecurrentDenseSafeHead(width=4, passes=1, match_channels=6)
    identity = identity_vertices(257, device=torch.device("cpu"))
    feature = torch.zeros((1, 88, 256, 256))
    with pytest.raises(ValueError, match="dynamic evidence has wrong shape"):
        model(identity, feature, evidence_fn=lambda _: torch.zeros(1, 5, 256, 256))
    with pytest.raises(ValueError, match="dynamic evidence requires"):
        model(identity, feature, match_feature=torch.zeros(1, 6, 256, 256),
              evidence_fn=lambda _: torch.zeros(1, 6, 256, 256))


def test_logit_gain_is_default_preserving_and_remains_topology_safe():
    model = RecurrentDenseSafeHead(width=4, passes=1, match_channels=6)
    with torch.no_grad():
        model.heads[0].bias.copy_(torch.tensor([.04, -.03]))
    identity = identity_vertices(257, device=torch.device("cpu"))
    feature = torch.zeros((1, 88, 256, 256))
    evidence = lambda _: torch.zeros((1, 6, 256, 256))
    original = model(identity, feature, evidence_fn=evidence)
    gain_one = model(identity, feature, evidence_fn=evidence, logit_gain=1.)
    gain_two = model(identity, feature, evidence_fn=evidence, logit_gain=2.)
    assert torch.equal(original, gain_one)
    assert not torch.equal(original, gain_two)
    assert validate_q1_map(gain_two, identity)["valid"]
    with pytest.raises(ValueError, match="positive finite logit gain"):
        model(identity, feature, evidence_fn=evidence, logit_gain=0.)


def test_twelve_channel_fixed_match_plus_current_image_cue_vjp():
    model = RecurrentDenseSafeHead(width=4, passes=2, match_channels=12)
    with torch.no_grad():
        model.trunk[0].weight.fill_(.001)
        for head in model.heads:
            head.weight.fill_(.002)
            head.bias.fill_(.004)
    identity = identity_vertices(257, device=torch.device("cpu"))
    feature = torch.zeros((1, 88, 256, 256))
    raster = torch.ones((1, 6, 256, 256), requires_grad=True)
    cue = torch.ones((1, 6, 256, 256), requires_grad=True)
    seen = []

    def evidence(current):
        seen.append(current)
        force = cue * (.5 + current[:, :-1, :-1, :1].permute(0, 3, 1, 2))
        return torch.cat((raster, force), dim=1)

    mapped = model(identity, feature, evidence_fn=evidence)
    assert len(seen) == 2
    assert validate_q1_map(mapped, identity)["valid"]
    mapped[:, 128, 128].sum().backward()
    assert raster.grad is not None and bool(torch.isfinite(raster.grad).all())
    assert cue.grad is not None and bool(torch.isfinite(cue.grad).all())
    assert bool(raster.grad.abs().sum() > 0)
    assert bool(cue.grad.abs().sum() > 0)


def test_twelve_channel_recurrent_evidence_directional_derivatives():
    torch.manual_seed(11)
    model = RecurrentDenseSafeHead(width=4, passes=2,
                                   match_channels=12).double()
    with torch.no_grad():
        for head in model.heads:
            head.weight.fill_(.02)
            head.bias.copy_(torch.tensor([.03, -.02], dtype=torch.float64))
    identity = identity_vertices(257, device=torch.device("cpu")).double()
    feature = torch.zeros((1, 88, 256, 256), dtype=torch.float64)
    ones = torch.ones((1, 6, 256, 256), dtype=torch.float64)

    def value(match_scale, force_scale):
        def evidence(current):
            local = .5 + current[:, :-1, :-1, :1].permute(0, 3, 1, 2)
            return torch.cat((match_scale * ones,
                              force_scale * ones * local), dim=1)
        mapped = model(identity, feature, evidence_fn=evidence)
        return mapped[0, 128, 128, 0] + .7 * mapped[0, 129, 128, 1]

    match_scale = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    force_scale = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.grad(value(match_scale, force_scale),
                                   (match_scale, force_scale))
    epsilon = 1e-4
    for index, analytic in enumerate(expected):
        plus = [match_scale.detach(), force_scale.detach()]
        minus = [match_scale.detach(), force_scale.detach()]
        plus[index] = plus[index] + epsilon
        minus[index] = minus[index] - epsilon
        finite_difference = (value(*plus) - value(*minus)) / (2 * epsilon)
        assert torch.isfinite(analytic)
        assert torch.allclose(analytic, finite_difference, atol=1e-7, rtol=.03)
