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


def test_geometry_comparison_preserves_frozen_sampler_and_all_evidence():
    report={"configuration":dict(fixed="f.png",moving="m.png",affine="a.npz",matches="raw.json",
        output="old.npz",p1_sampling="frozen",interpolation="p1_ac",strain_weight=3.,method="analytic")}
    config=replay_configuration(report,Path("new.npz"),"stage_cache","geometry")
    assert config.geometry_backend=="stage_cache" and config.p1_sampling=="frozen"
    assert config.strain_weight==3. and config.matches==Path("raw.json")
    assert "geometry_backend" not in report["configuration"]


def test_hierarchy_comparison_preserves_geometry_and_all_fine_evidence():
    report={"configuration":dict(p1_sampling="frozen",geometry_backend="stage_cache",
        interpolation="p1_ac",strain_weight=3.,method="analytic",levels=[17,33,65,129,257],
        cycles=1,grid_side=257,matches="image.json")}
    config=replay_configuration(report,Path("nested.npz"),"nested_p1","hierarchy")
    assert config.control_hierarchy=="nested_p1" and config.geometry_backend=="stage_cache"
    assert config.p1_sampling=="frozen" and config.grid_side==257 and config.strain_weight==3.
    assert "control_hierarchy" not in report["configuration"]
