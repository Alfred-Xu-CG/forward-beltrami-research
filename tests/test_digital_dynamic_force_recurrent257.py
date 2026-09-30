"""Dynamic evidence is recomputed at each topology-safe decoder pass."""

import pytest
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_q1_dhr_distill import identity_vertices


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
