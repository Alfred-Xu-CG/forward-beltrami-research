import sys

import pytest

import tools.coordinated_development_sweep as sweep


@pytest.mark.parametrize("inner,cycles",[(60,1),(30,2)])
def test_equal_gradient_budget_and_supported_cache_only(tmp_path,monkeypatch,inner,cycles):
    captured=[]
    def optimize(config):
        captured.append(config)
        gradients=config.cycles*len(config.levels)*config.inner_steps*(2 if config.method in ("radial","analytic") else 1)
        return dict(final={},failed_trials=0,gradient_steps=gradients,optimize_seconds=.1,peak_allocated_bytes=None)
    monkeypatch.setattr(sweep,"optimize",optimize)
    monkeypatch.setattr(sys,"argv",["sweep","--data",str(tmp_path),"--output",str(tmp_path/"out"),
        "--cases","histo","--methods","radial","analytic","f1","f2","--loss","mind",
        "--inner-steps",str(inner),"--base-cycles",str(cycles),"--geometry-backend","stage_cache",
        "--p1-sampling","frozen","--precision","float64","--image-precision","float32",
        "--interpolation","p1_ac","--image-levels","32","64","128","256","512",
        "--strain-weight","3","--shape-weight",".0001","--match-weight",".1",
        "--matches-dir",str(tmp_path),"--output-selection","best_full","--continuation-scope","first_cycle"])
    sweep.main()
    assert len(captured)==4
    for config in captured:
        assert config.levels==[17,33,65,129,257]
        assert config.grid_side==257 and config.image_side==512
        assert config.strain_weight==3 and config.shape_weight==.0001 and config.match_weight==.1
        assert config.output_selection=="best_full" and config.p1_sampling=="frozen"
        assert config.continuation_scope=="first_cycle"
        assert config.cycles==cycles*(2 if config.method in ("f1","f2") else 1)
        assert config.geometry_backend==("stage_cache" if config.method in ("radial","analytic") else "existing")
        count=config.cycles*len(config.levels)*config.inner_steps*(2 if config.method in ("radial","analytic") else 1)
        assert count==600
    assert (tmp_path/"out"/"mind_summary.json").exists()


@pytest.mark.parametrize("calibration",["edge","physical"])
def test_joint_sweep_has_two_fields_but_one_objective_gradient_per_stage_step(tmp_path,monkeypatch,calibration):
    captured=[]
    def mock(config):
        captured.append(config)
        count=config.cycles*len(config.levels)*config.inner_steps
        return dict(final={},failed_trials=0,gradient_steps=count,optimize_seconds=.1,peak_allocated_bytes=None)
    monkeypatch.setattr(sweep,"optimize",mock)
    monkeypatch.setattr(sys,"argv",["sweep","--data",str(tmp_path),"--output",str(tmp_path/"joint"),
        "--cases","histo","--methods","radial","analytic","--loss","mind",
        "--coordinate-mode","joint","--joint-backend","cached_manual",
        "--inner-steps","60","--base-cycles","1","--lr-calibration",calibration])
    sweep.main()
    assert len(captured)==2
    assert all(c.coordinate_mode=="joint" and c.joint_backend=="cached_manual" and
        c.geometry_backend=="existing" and c.inner_steps==60 and c.cycles==1 for c in captured)
    assert all(c.lr_calibration==calibration and c.output.name.endswith(calibration+"257.npz") for c in captured)


def test_filter_configuration_reaches_application_without_changing_calibration(tmp_path,monkeypatch):
    captured=[]
    def mock(config):
        captured.append(config)
        return dict(final={},failed_trials=0,gradient_steps=300,optimize_seconds=.1,peak_allocated_bytes=None)
    monkeypatch.setattr(sweep,"optimize",mock)
    monkeypatch.setattr(sys,"argv",["sweep","--data",str(tmp_path),"--output",str(tmp_path/"filter"),
        "--cases","histo","--methods","analytic","--loss","mind","--lr-calibration","physical",
        "--proposal-filter-steps","4","--proposal-filter-min-level","129"])
    sweep.main()
    assert len(captured)==1
    assert captured[0].proposal_filter_steps==4 and captured[0].proposal_filter_min_level==129
    assert captured[0].lr_calibration=="physical"
