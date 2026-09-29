"""Focused checks of match-conditioned, safe Q1 forward and VJP."""

import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_match_neural_decoder import (
    MatchSpatialDecoder, gaussian_features, match_rmse, split_matches,
)


def _matches():
    axis = torch.linspace(.12, .88, 5)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack((x, y), dim=-1).reshape(-1, 2)
    target = source + torch.stack((.02 * torch.sin(3 * y), .015 * torch.cos(2 * x)), dim=-1).reshape(-1, 2)
    return source, target


def test_gaussian_features_shapes_and_zero_motion():
    source, target = _matches()
    fields, support = gaussian_features(17, source, source)
    assert fields.shape == (1, 3, 17, 17, 2)
    assert support.shape == (1, 3, 17, 17)
    assert torch.count_nonzero(fields) == 0
    assert torch.isfinite(support).all()


def test_network_forward_and_vjp_are_finite_and_safe():
    source, target = _matches()
    model = MatchSpatialDecoder(update_sides=(17, 33), final_side=65, width=8)
    result = model(source, target)
    assert result.shape == (1, 65, 65, 2)
    check = validate_q1_map(result, identity_vertices(65, device=source.device))
    assert check["valid"] and check["boundary_ordered_rectangle"]
    error = match_rmse(result, source, target)
    assert error < match_rmse(identity_vertices(65, device=source.device), source, target)
    error.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.net.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.net.parameters()) > 0


def test_split_matches_no_overlap_and_no_empty_half():
    source, target = _matches()
    input_source, input_target, held_source, held_target = split_matches(
        source, target, seed=12, input_fraction=.7,
    )
    assert len(input_source) + len(held_source) == len(source)
    assert len(input_source) and len(held_source)
    assert set(map(tuple, input_source.tolist())).isdisjoint(set(map(tuple, held_source.tolist())))
    assert input_target.shape == input_source.shape
    assert held_target.shape == held_source.shape


def test_exact_match_has_finite_zero_vjp():
    source, _ = _matches()
    identity = identity_vertices(17, device=source.device).requires_grad_(True)
    loss = match_rmse(identity, source, source)
    loss.backward()
    assert torch.isfinite(identity.grad).all()
