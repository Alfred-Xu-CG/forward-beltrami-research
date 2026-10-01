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
