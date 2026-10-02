"""Frozen machine-point query dispatch preserves the complete application."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tools import coordinated_real_case as app
from tools.coordinated_application_profile import load_configuration
from test_coordinated_application_profile import fixture_report


def test_actual_tiny_optimizer_prepares_one_shared_point_cache(tmp_path,monkeypatch):
    source=fixture_report(tmp_path)
    config=load_configuration(source,tmp_path/"ordinary.npz",production=False)
    config.device="cpu"
    ordinary=app.optimize(config)
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    original=ImageCorrespondences.prepare_fixed_p1_sampling
    calls=[]
    def prepare(self,*values,**kwargs):
        calls.append(self)
        return original(self,*values,**kwargs)
    monkeypatch.setattr(ImageCorrespondences,"prepare_fixed_p1_sampling",prepare)
    config.output=tmp_path/"frozen.npz";config.match_p1_sampling="frozen"
    cached=app.optimize(config)
    assert len(calls)==1
    assert cached["match_p1_sampling"]==cached["configuration"]["match_p1_sampling"]=="frozen"
    assert cached["image_match_evidence"]["fixed_p1_sampling"]["queries"]==9
    assert cached["gradient_steps"]==ordinary["gradient_steps"]==4
    assert cached["failed_trials"]==ordinary["failed_trials"]==0
    assert cached["saved_binary_certificate"]["valid"]
    for key,value in ordinary["final"].items():
        assert cached["final"][key]==pytest.approx(value,rel=1e-11,abs=1e-13)
    with np.load(tmp_path/"ordinary.npz") as a,np.load(config.output) as b:
        np.testing.assert_allclose(a["vertices"],b["vertices"],rtol=1e-11,atol=1e-13)


@pytest.mark.parametrize("change",[
    {"match_p1_sampling":"bad"},{"interpolation":"q1"},
    {"match_weight":0.},{"matches":None},{"control_hierarchy":"nested_p1"},
])
def test_unsupported_frozen_point_dispatch_rejected_before_input_loading(tmp_path,change):
    from tools.coordinated_lung_all20 import make_configuration
    config=make_configuration(dict(name="missing",fixed=tmp_path/"missing.png",
        moving=tmp_path/"missing.png",affine=tmp_path/"missing.npz"),"analytic",
        argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    config.match_p1_sampling="frozen"
    for key,value in change.items():setattr(config,key,value)
    with pytest.raises(ValueError,match="match_p1_sampling|frozen point sampling|p1_arap"):
        app.optimize(config)
