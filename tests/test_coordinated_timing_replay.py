from pathlib import Path

import pytest

from tools.coordinated_timing_replay import replay_configuration,sampling_order


def test_counterbalanced_order_and_only_sampling_changes():
    report={"configuration":dict(fixed="f.png",moving="m.png",affine="a.npz",matches=None,
        output="old.npz",p1_sampling="existing",interpolation="p1_ac",strain_weight=3.)}
    config=replay_configuration(report,Path("new.npz"),"frozen")
    assert config.fixed==Path("f.png") and config.output==Path("new.npz")
    assert config.matches is None and config.strain_weight==3.
    assert report["configuration"]["p1_sampling"]=="existing"
    assert config.p1_sampling=="frozen"
    assert sampling_order(0)==sampling_order(2)==("existing","frozen")
    assert sampling_order(1)==sampling_order(3)==("frozen","existing")


def test_rejects_q1_template():
    with pytest.raises(ValueError,match="actual P1"):
        replay_configuration({"configuration":{"interpolation":"q1"}},Path("x.npz"),"frozen")
