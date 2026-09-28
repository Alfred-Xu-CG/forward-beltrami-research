"""An isolated DHR adapter must preserve the original matched transform."""

import numpy as np
import torch

import deeperhistreg  # noqa: F401 - installs DHR's legacy import paths
import superpoint_superglue as dhr_matcher

from tools.digital_dhr_cached_matcher import CachedDHRMatcher


class _FakeWeights:
    def load_state_dict(self, state):
        assert state == {"weights": 1}


class _FakeMatching:
    def __init__(self, config):
        self.config = config
        self.superpoint = _FakeWeights()
        self.superglue = _FakeWeights()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, images):
        points = torch.tensor([[[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [4.0, 4.0]]])
        return {
            "keypoints0": points,
            "keypoints1": points + torch.tensor([[[1.0, 2.0]]]),
            "matches0": torch.tensor([[0, 1, 2, 3]]),
        }


def test_cached_adapter_matches_original_and_loads_once(monkeypatch):
    constructions = []
    loads = []

    def make_model(config):
        constructions.append(config)
        return _FakeMatching(config)

    def load_weights(path):
        loads.append(path)
        return {"weights": 1}

    monkeypatch.setattr(dhr_matcher.sg, "Matching", make_model)
    monkeypatch.setattr(dhr_matcher.tc, "load", load_weights)
    source = np.zeros((8, 8), dtype=np.float32)
    params = {"device": "cpu", "echo": False, "transform_type": "rigid"}
    original = dhr_matcher.perform_registration(source, source, params)
    assert len(constructions) == 1 and len(loads) == 2

    cache = CachedDHRMatcher()
    first = cache.perform_registration(source, source, params)
    second = cache.perform_registration(source, source, params)
    assert len(constructions) == 2 and len(loads) == 4
    assert cache.model_builds == 1
    for candidate in (first, second):
        np.testing.assert_allclose(candidate[0], original[0], atol=0, rtol=0)
        assert candidate[1] == original[1]
        np.testing.assert_array_equal(candidate[2], original[2])


def test_cached_adapter_rebuilds_when_model_config_changes(monkeypatch):
    constructions = []
    monkeypatch.setattr(dhr_matcher.sg, "Matching", lambda config: constructions.append(config) or _FakeMatching(config))
    monkeypatch.setattr(dhr_matcher.tc, "load", lambda path: {"weights": 1})
    cache = CachedDHRMatcher()
    source = np.zeros((8, 8), dtype=np.float32)
    common = {"device": "cpu", "echo": False, "transform_type": "rigid"}
    cache.perform_registration(source, source, common)
    cache.perform_registration(source, source, {**common, "match_threshold": 0.8})
    assert len(constructions) == 2
