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


def test_patch_backend_comparison_changes_only_backend_and_output(tmp_path):
    source=dict(fixed="f.png",moving="m.png",affine="a.npz",matches="matches.json",
        output="old.npz",p1_sampling="frozen",interpolation="p1_ac",method="analytic",
        geometry_backend="stage_cache",fine_patch_cells=16,fine_patch_backend="ordinary",
        grid_side=1025,image_side=1024,levels=[17,33,65,129,1025],strain_weight=3.)
    report={"configuration":source.copy()}
    for backend in ("ordinary","manual"):
        config=replay_configuration(report,tmp_path/"new.npz",backend,"fine_patch_backend")
        expected=source.copy()
        for key in ("fixed","moving","affine","matches"):
            expected[key]=Path(expected[key])
        expected.update(output=tmp_path/"new.npz",fine_patch_backend=backend)
        assert vars(config)==expected
    assert report["configuration"]==source


@pytest.mark.parametrize("cells",[None,0,-2,True])
def test_patch_backend_rejects_inactive_patch_configuration(tmp_path,cells):
    config=dict(interpolation="p1_ac")
    if cells is not None:config["fine_patch_cells"]=cells
    with pytest.raises(ValueError,match="requires active fine_patch_cells"):
        replay_configuration({"configuration":config},tmp_path/"x.npz","manual","fine_patch_backend")


def test_patch_backend_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError,match="ordinary or manual"):
        replay_configuration({"configuration":dict(interpolation="p1_ac",fine_patch_cells=16)},
            tmp_path/"x.npz","stage_cache","fine_patch_backend")


def test_patch_backend_complete_calls_warm_both_then_counterbalance(tmp_path,monkeypatch):
    import argparse
    import json
    from tools import coordinated_timing_replay as replay

    template=tmp_path/"report.json"
    configuration=dict(interpolation="p1_ac",fine_patch_cells=16,fine_patch_backend="ordinary",
        grid_side=1025,image_side=1024,levels=[17,33,65,129,1025],matches="same_matches.json")
    template.write_text(json.dumps({"configuration":configuration}))
    calls=[]
    def fake_optimize(config):
        calls.append(vars(config).copy())
        return dict(optimize_seconds=2.,end_to_end_seconds=3.,feature_seconds=.5,
            peak_allocated_bytes=100,final={"objective":.25},gradient_steps=100,failed_trials=0,
            saved_binary_certificate={"valid":True})
    monkeypatch.setattr(replay,"optimize",fake_optimize)
    result=replay.run(argparse.Namespace(report=template,output=tmp_path/"timing.json",
        repeats=4,comparison="fine_patch_backend",grid_side=None))
    assert [call["fine_patch_backend"] for call in calls]==[
        "ordinary","manual", # both complete configurations warm up
        "ordinary","manual","manual","ordinary", # AB / BA
        "ordinary","manual","manual","ordinary", # AB / BA
    ]
    assert len({call["output"] for call in calls})==10
    assert all(call["fine_patch_cells"]==16 and call["matches"]==Path("same_matches.json")
        and call["grid_side"]==1025 and call["image_side"]==1024 for call in calls)
    assert [row["warmup"] for row in result["rows"]]==[True,True]+[False]*8
    assert set(result["medians"])=={"ordinary","manual"}
    assert result["configuration"]==configuration
    assert all(row["certificate_valid"] and row["gradient_steps"]==100 for row in result["rows"])
    assert "same configured patch geometry" in result["precision_caution"]
