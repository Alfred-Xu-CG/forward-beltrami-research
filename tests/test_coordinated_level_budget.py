"""Unequal fixed stage budgets do not change the default registration recipe."""
import argparse

import numpy as np
import pytest

from tools.coordinated_lung_all20 import make_configuration
from tools.coordinated_real_case import optimize
from test_coordinated_lung_all20 import assets


def configuration(tmp_path,method="analytic"):
    source=assets(tmp_path)
    pair=dict(name="he_to_cc10",fixed=source.canvas/"cc10_fixed512.png",
        moving=source.canvas/"cc10_moving512.png",
        affine=source.affines_from/"he_to_cc10_affine.npz")
    config=make_configuration(pair,method,argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    config.matches=None; config.match_weight=0.
    return config


def test_explicit_uniform_budget_preserves_default_map_and_objective(tmp_path):
    config=configuration(tmp_path)
    config.output=tmp_path/"default.npz"
    original=optimize(config)
    config.output=tmp_path/"uniform.npz"
    config.inner_steps_by_level=[1,1]
    explicit=optimize(config)
    assert original["final"]==explicit["final"]
    with np.load(tmp_path/"default.npz") as a,np.load(config.output) as b:
        assert all(np.array_equal(a[key],b[key]) for key in a.files)
    assert original["inner_steps_by_level"]==explicit["inner_steps_by_level"]==[1,1]


@pytest.mark.parametrize("method,cycles,gradients",[("analytic",1,8),("analytic",2,16),("f2",2,8)])
def test_per_level_counts_stage_refresh_and_actual_certificate(tmp_path,method,cycles,gradients):
    config=configuration(tmp_path,method)
    config.cycles=cycles; config.inner_steps_by_level=[1,3]
    config.f2_floor_safety_fraction=.95
    report=optimize(config)
    stages=cycles*2*(2 if method=="analytic" else 1)
    assert report["gradient_steps"]==gradients and report["failed_trials"]==0
    assert report["evaluations"]==gradients+stages
    assert report["objective_evaluations"]==gradients+3*stages+2
    assert report["saved_binary_certificate"]["valid"]
    assert report["configuration"]["inner_steps_by_level"]==[1,3]
    for stage in report["stages"]:
        expected=1 if stage["level"]==5 else 3
        assert stage["inner_steps"]==expected
        trials=[row for row in report["trace"] if row["cycle"]==stage["cycle"]
                and row["level"]==stage["level"] and row["direction"]==stage["direction"]]
        assert [row["step"] for row in trials]==list(range(expected+1))


@pytest.mark.parametrize("budget",[[],[1],[1,1,1],[0,1],[1,-1],[True,1],[1,1.5],"1,1"])
def test_invalid_budget_rejected_before_input_loading(tmp_path,budget):
    config=make_configuration(dict(name="missing",fixed=tmp_path/"missing.png",
        moving=tmp_path/"missing.png",affine=tmp_path/"missing.npz"),"analytic",
        argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    config.inner_steps_by_level=budget
    with pytest.raises(ValueError,match="inner_steps_by_level"):
        optimize(config)
