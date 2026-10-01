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
def test_nested_evidence_replay_sets_only_declared_evaluation(tmp_path):
    from tools.coordinated_timing_replay import replay_configuration
    report={"configuration":dict(fixed="f",moving="m",affine="a",output="old",
        interpolation="p1_ac",control_hierarchy="fixed",p1_sampling="frozen")}
    config=replay_configuration(report,tmp_path/"new.npz","coarse_exact","nested_evidence")
    assert config.control_hierarchy=="nested_p1" and config.nested_evaluation=="coarse_exact"
    assert config.p1_sampling=="frozen" and report["configuration"]["control_hierarchy"]=="fixed"


def test_larger_actual_control_grid_keeps_image_resolution_and_middle_levels():
    report={"configuration":dict(interpolation="p1_ac",grid_side=257,image_side=512,
        levels=[17,33,65,129,257],image_levels=[32,64,128,256,512])}
    config=replay_configuration(report,Path("new.npz"),"full_fine","nested_evidence",1025)
    assert config.grid_side==1025 and config.levels==[17,33,65,129,1025]
    assert config.image_side==512 and config.image_levels==report["configuration"]["image_levels"]
    assert report["configuration"]["levels"][-1]==257
    with pytest.raises(ValueError):replay_configuration(report,Path("bad.npz"),"full_fine","nested_evidence",129)
