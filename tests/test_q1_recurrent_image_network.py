"""Repeated image-conditioned passes preserve the shared Q1 grid."""

import copy

import torch
import torch.nn.functional as F
import pytest

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from qcopt.neural_bijection.dense.q1_recurrent_image_network import (
    RecurrentQ1ImageRegistrationNetwork,
)
from tools.digital_acrobat_recurrent_probe import parse_rounds


def test_recurrent_ablation_schedule_parser() -> None:
    assert parse_rounds("17:1,33:1,65:1") == {17: 1, 33: 1, 65: 1}
    assert parse_rounds("17:4,33:2,65:1,257:4")[257] == 4
    with pytest.raises(Exception):
        parse_rounds("17:1,17:4")
    with pytest.raises(Exception):
        parse_rounds("17:0")


def test_zero_heads_equal_base_and_vjp_reaches_current_image_head() -> None:
    torch.manual_seed(29)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4, flow_hint=False)
    network = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2, 33: 1},
    )
    fixed = torch.rand(1, 1, 64, 64)
    moving = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        expected = base(fixed, moving)[0]
        actual = network(fixed, moving)[0]
    assert torch.equal(actual, expected)
    assert float(q1_corner_determinants(actual).amin()) > 0
    axis = torch.arange(33, dtype=actual.dtype) / 32
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.stack((xx + .05 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy), yy), -1)
    mapped = network(fixed, moving)[0]
    loss = (mapped - target[None]).square().mean()
    loss.backward()
    assert network.heads["17"].output.bias.grad is not None
    assert bool(torch.isfinite(network.heads["17"].output.bias.grad).all())
    assert float(network.heads["17"].output.bias.grad.abs().sum()) > 0


def test_nonzero_recurrent_heads_stay_corner_positive() -> None:
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4, flow_hint=False)
    network = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 3, 33: 2},
    )
    with torch.no_grad():
        network.heads["17"].output.bias.copy_(torch.tensor([2., -1.]))
        network.heads["33"].output.bias.copy_(torch.tensor([-2., 1.]))
    fixed = torch.zeros(1, 1, 64, 64)
    moving = torch.ones_like(fixed)
    result = network(fixed, moving)[0]
    assert bool(torch.isfinite(result).all())
    assert float(q1_corner_determinants(result).amin()) > 0
    assert torch.equal(result[:, 0], base(fixed, moving)[0][:, 0])


def test_current_image_evidence_uses_returned_affine_position() -> None:
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    with torch.no_grad():
        base.affine_head.output.bias[4] = torch.atanh(torch.tensor(.5))
    network = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 1},
    )
    fixed = torch.zeros(1, 1, 64, 64)
    moving = ((torch.arange(64, dtype=torch.float32) + .5) / 64
              ).reshape(1, 1, 1, 64).expand(1, 1, 64, 64)
    captured = []

    def capture(_module, arguments):
        captured.append(arguments[2].detach().clone())

    handle = network.heads["17"].register_forward_pre_hook(capture)
    with torch.no_grad():
        residual, matrix, offset = network(fixed, moving)
    handle.remove()
    query = Q1ImageRegistrationNetwork.apply_affine(residual, matrix, offset)
    expected = F.grid_sample(moving, 2 * query - 1,
                             padding_mode="border", align_corners=False)
    raw = F.grid_sample(moving, 2 * residual - 1,
                        padding_mode="border", align_corners=False)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=1e-6)
    assert float((captured[0] - raw).abs().amax()) > .09


def test_freezing_level_image_evidence_is_a_distinct_recurrent_ablation() -> None:
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    dynamic = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2}, refresh_moving_evidence=True,
    )
    frozen = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2}, refresh_moving_evidence=False,
    )
    fixed = torch.zeros(1, 1, 64, 64)
    moving = ((torch.arange(64, dtype=torch.float32) + .5) / 64
              ).reshape(1, 1, 1, 64).expand(1, 1, 64, 64)
    captured = {"dynamic": [], "frozen": []}
    handles = []
    for name, model in (("dynamic", dynamic), ("frozen", frozen)):
        with torch.no_grad():
            model.heads["17"].output.bias.copy_(torch.tensor([1., 0.]))
        handles.append(model.heads["17"].register_forward_pre_hook(
            lambda _module, args, name=name: captured[name].append(args[2].detach().clone())
        ))
    with torch.no_grad():
        dynamic(fixed, moving)
        frozen(fixed, moving)
    for handle in handles:
        handle.remove()
    assert len(captured["dynamic"]) == len(captured["frozen"]) == 2
    torch.testing.assert_close(captured["dynamic"][0], captured["frozen"][0],
                               rtol=0, atol=0)
    torch.testing.assert_close(captured["frozen"][0], captured["frozen"][1],
                               rtol=0, atol=0)
    assert float((captured["dynamic"][0] - captured["dynamic"][1]).abs().amax()) > 1e-5


def test_fixed_h_recurrent_proposal_is_safe_control() -> None:
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    model = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2}, proposal_mode="fixed_h",
    )
    with torch.no_grad():
        model.heads["17"].output.bias.copy_(torch.tensor([2., -1.]))
        mapped = model(torch.zeros(1, 1, 64, 64), torch.ones(1, 1, 64, 64))[0]
    assert float(q1_corner_determinants(mapped).amin()) > 0


def test_proposal_modes_can_change_only_new_fine_level() -> None:
    from qcopt.neural_bijection.dense.digital_q1 import (
        AdaptiveSoftRadialQ1Relaxation, FixedSpanSoftRadialQ1Relaxation,
    )

    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4, flow_hint=False)
    model = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2, 33: 2},
        proposal_mode="fixed_h_soft",
        proposal_modes_by_side={17: "current_edge", 33: "fixed_h_soft"},
    )
    assert type(model.updates["17"]) is AdaptiveSoftRadialQ1Relaxation
    assert type(model.updates["33"]) is FixedSpanSoftRadialQ1Relaxation
    with torch.no_grad():
        model.heads["17"].output.bias.copy_(torch.tensor([.6, -.3]))
        model.heads["33"].output.bias.copy_(torch.tensor([-.5, .4]))
        mapped = model(torch.zeros(1, 1, 64, 64), torch.ones(1, 1, 64, 64))[0]
    assert float(q1_corner_determinants(mapped).amin()) > 0
    with pytest.raises(ValueError, match="one valid proposal mode"):
        RecurrentQ1ImageRegistrationNetwork(
            base, rounds_by_side={17: 2, 33: 2},
            proposal_modes_by_side={17: "current_edge"},
        )


def test_round_checkpoint_replays_same_map_and_vjp() -> None:
    torch.manual_seed(941)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    ordinary = RecurrentQ1ImageRegistrationNetwork(base, rounds_by_side={17: 2})
    with torch.no_grad():
        ordinary.heads["17"].output.weight.normal_(0, .01)
        ordinary.heads["17"].output.bias.copy_(torch.tensor([.7, -.4]))
    replay = copy.deepcopy(ordinary)
    replay.checkpoint_rounds = True
    first = torch.rand(1, 1, 48, 48)
    second = torch.rand(1, 1, 48, 48)
    results = []
    for network in (ordinary, replay):
        fixed = first.clone().requires_grad_(True)
        moving = second.clone().requires_grad_(True)
        mapped = network(fixed, moving)[0]
        mapped.square().mean().backward()
        results.append((mapped.detach().clone(), fixed.grad.clone(), moving.grad.clone(),
                        network.heads["17"].output.weight.grad.clone()))
    for old, new in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=1e-7)
    assert float(q1_corner_determinants(results[1][0]).amin()) > 0


def test_image_conditioned_f2_repeats_on_current_map_with_vjp() -> None:
    torch.manual_seed(943)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    model = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2},
        update_families_by_side={17: "f2"},
    )
    fixed = torch.rand(1, 1, 48, 48)
    moving = ((torch.arange(48, dtype=torch.float32) + .5) / 48
              ).reshape(1, 1, 1, 48).expand(1, 1, 48, 48).clone()
    with torch.no_grad():
        parent = base(fixed, moving)[0]
        neutral = model(fixed, moving)[0]
    assert torch.equal(parent, neutral)
    with torch.no_grad():
        model.heads["17"].output.bias.copy_(torch.tensor([.8, -.3]))
    evidence = []
    handle = model.heads["17"].register_forward_pre_hook(
        lambda _module, args: evidence.append(args[2].detach().clone())
    )
    result = model(fixed, moving)[0]
    handle.remove()
    assert len(evidence) == 8
    assert float((evidence[0] - evidence[1]).abs().amax()) > 1e-6
    assert float((result - parent).abs().amax()) > 1e-6
    assert float(q1_corner_determinants(result).amin()) > 0
    model.zero_grad(set_to_none=True)
    result.square().mean().backward()
    gradient = model.heads["17"].output.bias.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0


def test_image_conditioned_mixed_f2_f1_checkpoint_and_vjp() -> None:
    torch.manual_seed(290930)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    ordinary = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2},
        update_families_by_side={17: "f2_f1"},
        f2_patch_cells=4, f2_raw_span=1.0,
    )
    fixed = torch.rand(1, 1, 48, 48)
    moving = torch.rand(1, 1, 48, 48)
    with torch.no_grad():
        assert torch.equal(ordinary(fixed, moving)[0], base(fixed, moving)[0])
        ordinary.heads["17"]["f2"].output.bias.copy_(torch.tensor([.2, -.1]))
        ordinary.heads["17"]["f1"].output.bias.copy_(torch.tensor([.2, .1]))
        ordinary.heads["17"]["f2"].output.weight.normal_(0, .01)
        ordinary.heads["17"]["f1"].output.weight.normal_(0, .01)
    evidence = {"f2": [], "f1": []}
    handles = [ordinary.heads["17"][kind].register_forward_pre_hook(
        lambda _module, args, kind=kind: evidence[kind].append(args[2].detach().clone())
    ) for kind in ("f2", "f1")]
    with torch.no_grad():
        ordinary(fixed, moving)
    for handle in handles:
        handle.remove()
    assert len(evidence["f2"]) == 8 and len(evidence["f1"]) == 2
    assert float((evidence["f2"][0] - evidence["f1"][0]).abs().amax()) > 1e-6
    replay = copy.deepcopy(ordinary)
    replay.checkpoint_rounds = True
    outputs = []
    for model in (ordinary, replay):
        moving_input = moving.clone().requires_grad_(True)
        warped_inputs = {"f2": [], "f1": []}
        if model is ordinary:
            def retain_warped(_module, args, kind):
                args[2].retain_grad()
                warped_inputs[kind].append(args[2])
            handles = [model.heads["17"][kind].register_forward_pre_hook(
                lambda module, args, kind=kind: retain_warped(module, args, kind)
            ) for kind in ("f2", "f1")]
        result = model(fixed, moving_input)[0]
        if model is ordinary:
            for handle in handles:
                handle.remove()
        assert float(q1_corner_determinants(result).amin()) > 0
        result.square().mean().backward()
        assert moving_input.grad is not None
        assert bool(torch.isfinite(moving_input.grad).all())
        if model is ordinary:
            assert len(warped_inputs["f2"]) == 8 and len(warped_inputs["f1"]) == 2
            assert all(any(item.grad is not None and float(item.grad.abs().sum()) > 0
                           for item in warped_inputs[kind]) for kind in ("f2", "f1"))
        gradients = tuple(model.heads["17"][kind].output.bias.grad.clone()
                          for kind in ("f2", "f1"))
        assert all(bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
                   for grad in gradients)
        outputs.append((result.detach(), gradients))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    for first, second in zip(outputs[0][1], outputs[1][1], strict=True):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-8)


def test_image_conditioned_f2_checkpoint_vjp_matches_ordinary() -> None:
    torch.manual_seed(947)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    plain = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side={17: 2}, update_families_by_side={17: "f2"},
    )
    with torch.no_grad():
        plain.heads["17"].output.weight.normal_(0, .01)
        plain.heads["17"].output.bias.copy_(torch.tensor([.8, -.3]))
    saved = copy.deepcopy(plain)
    saved.checkpoint_rounds = True
    inputs = (torch.rand(1, 1, 48, 48), torch.rand(1, 1, 48, 48))
    results = []
    for network in (plain, saved):
        fixed, moving = (image.clone().requires_grad_(True) for image in inputs)
        mapped = network(fixed, moving)[0]
        mapped.square().mean().backward()
        results.append((mapped.detach(), fixed.grad, moving.grad,
                        network.heads["17"].output.bias.grad))
    for ordinary, checkpointed in zip(*results, strict=True):
        torch.testing.assert_close(ordinary, checkpointed, rtol=0, atol=1e-7)
    assert float(results[0][1].abs().sum()) > 0
    assert float(results[0][2].abs().sum()) > 0
    assert float(q1_corner_determinants(results[1][0]).amin()) > 0


def test_nondefault_f2_span_changes_motion_not_topology() -> None:
    torch.manual_seed(953)
    base = Q1ImageRegistrationNetwork(seed_side=17, final_side=17,
                                       feature_side=33, width=4, flow_hint=False)
    common = dict(rounds_by_side={17: 2},
                  update_families_by_side={17: "f2"})
    large = RecurrentQ1ImageRegistrationNetwork(base, f2_raw_span=.5, **common)
    small = RecurrentQ1ImageRegistrationNetwork(base, f2_raw_span=.125, **common)
    small.heads.load_state_dict(large.heads.state_dict())
    with torch.no_grad():
        large.heads["17"].output.bias.copy_(torch.tensor([.8, -.3]))
        small.heads["17"].output.bias.copy_(torch.tensor([.8, -.3]))
    fixed = torch.zeros(1, 1, 48, 48)
    moving = torch.ones_like(fixed)
    parent = base(fixed, moving)[0]
    wide = large(fixed, moving)[0]
    narrow = small(fixed, moving)[0]
    assert float((wide - narrow).abs().amax()) > 1e-7
    assert float((narrow - parent).abs().amax()) > 0
    assert float(q1_corner_determinants(narrow).amin()) > 0
    narrow.square().mean().backward()
    assert float(small.heads["17"].output.bias.grad.abs().sum()) > 0
