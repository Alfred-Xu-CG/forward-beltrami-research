"""Detached logging equivalence; CPU fixtures do not establish GPU speed."""
import argparse
from dataclasses import replace
import math

import pytest
import torch
import numpy as np

from tools.coordinated_trial_diagnostics import pack_trial_scalars
from tools import coordinated_real_case as app
from tools.coordinated_lung_all20 import make_configuration
from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from test_coordinated_lung_all20 import assets


def test_literal_conversions_and_one_copy(monkeypatch):
    values={"f32":torch.tensor(.1,dtype=torch.float32),"f64":torch.tensor(1.2345678901234567,dtype=torch.float64),
        "false":torch.tensor(False),"true":True,"int32":torch.tensor(-2147483648,dtype=torch.int32),
        "int":2**53,"float":-.123,"minus_zero":torch.tensor(-0.,dtype=torch.float32),
        "nan":torch.tensor(float("nan")),"inf":float("inf"),"negative_inf":torch.tensor(-float("inf"))}
    original=torch.Tensor.cpu;copies=[]
    def copy(tensor,*args,**kwargs):
        copies.append(tensor.clone());return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,"cpu",copy)
    result=pack_trial_scalars(values)
    assert len(copies)==1 and copies[0].dtype==torch.float64
    assert list(result)==list(values)
    for key,value in values.items():
        kind=bool if isinstance(value,bool) or isinstance(value,torch.Tensor) and value.dtype==torch.bool else (
            int if isinstance(value,int) or isinstance(value,torch.Tensor) and value.dtype==torch.int32 else float)
        expected=kind(value)
        assert type(result[key]) is kind
        assert math.isnan(result[key]) if key=="nan" else result[key]==expected
    assert math.copysign(1.,result["minus_zero"])==-1.
    assert pack_trial_scalars({})=={} and len(copies)==1


@pytest.mark.parametrize("values,error",[
    ([],TypeError),({1:0.},TypeError),({"x":"1"},TypeError),
    ({"x":torch.ones(2)},ValueError),({"x":torch.tensor(1.,requires_grad=True)},ValueError),
    ({"x":torch.tensor(1,dtype=torch.int64)},TypeError),
    ({"x":torch.tensor(1.,dtype=torch.float16)},TypeError),
    ({"x":torch.tensor(1j)},TypeError),({"x":2**53+1},ValueError),
    ({"x":torch.tensor(1.),"y":torch.empty((),device="meta")},ValueError),
])
def test_scalar_contract(values,error):
    with pytest.raises(error):pack_trial_scalars(values)


def test_does_not_detach_trainable_graph_or_change_inputs():
    x=torch.tensor(2.,requires_grad=True);y=x.square()
    with pytest.raises(ValueError):pack_trial_scalars({"y":y})
    assert pack_trial_scalars({"y":y.detach()})=={"y":4.}
    y.backward();assert x.grad.item()==4.


def configuration(tmp_path,source,path):
    pair=dict(name="he_to_cc10",fixed=source.canvas/"cc10_fixed512.png",
        moving=source.canvas/"cc10_moving512.png",affine=source.affines_from/"he_to_cc10_affine.npz")
    config=make_configuration(pair,"analytic",argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    config.matches=None;config.match_weight=0.;config.inner_steps=3
    config.output=tmp_path/(path+".npz")
    return config


def assert_same_records(a,b):
    # Timing/configuration strings intentionally differ; scientific trajectories do not.
    def same(x,y):
        if isinstance(x,float) and math.isnan(x):return isinstance(y,float) and math.isnan(y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return type(x) is type(y) and len(x)==len(y) and all(same(i,j) for i,j in zip(x,y,strict=True))
        return type(x) is type(y) and x==y
    for key in ("initial","final","trace","stages","failures","evaluations","gradient_steps",
        "failed_trials","objective_evaluations","selected_stage","coordinated_substep_evaluations"):
        assert same(a[key],b[key]),key


def test_tiny_whole_trajectory_maps_and_all_adam_gradients_bitwise(tmp_path,monkeypatch):
    source=assets(tmp_path);recorded=[];original=torch.optim.Adam
    class RecordingAdam(original):
        def step(self,*args,**kwargs):
            recorded.append(self.param_groups[0]["params"][0].grad.detach().clone())
            return super().step(*args,**kwargs)
    monkeypatch.setattr(torch.optim,"Adam",RecordingAdam)
    reports=[];gradients=[]
    for mode in ("existing","packed"):
        config=configuration(tmp_path,source,mode)
        if mode=="packed":config.trial_diagnostics=mode  # omitted Namespace defaults to existing
        recorded.clear();reports.append(app.optimize(config));gradients.append(list(recorded))
    assert_same_records(*reports)
    assert reports[0]["gradient_steps"]==12 and reports[0]["failed_trials"]==0
    assert reports[0]["diagnostic_pack_calls"]==0
    assert reports[1]["diagnostic_pack_calls"]==reports[1]["evaluations"]==16
    assert len(gradients[0])==len(gradients[1])==12
    assert all(torch.equal(a,b) for a,b in zip(*gradients,strict=True))
    with np.load(tmp_path/"existing.npz") as a,np.load(tmp_path/"packed.npz") as b:
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize("bad",["margin","coordinates","objective","gradient"])
def test_rejected_trials_remain_immediate_and_keep_same_best_map(tmp_path,monkeypatch,bad):
    source=assets(tmp_path);original=FrozenAnchorCoordinatedUpdate.__call__
    def corrupted(self,proposal,**kwargs):
        result=original(self,proposal,**kwargs)
        if bad=="margin":return replace(result,normalized_margin_min=torch.zeros_like(result.normalized_margin_min))
        if bad=="coordinates":
            vertices=result.vertices.clone();vertices[:,4,4,0]=float("inf")
            return replace(result,vertices=vertices)
        if bad=="gradient":result.vertices.register_hook(lambda grad:torch.full_like(grad,float("nan")))
        return result
    monkeypatch.setattr(FrozenAnchorCoordinatedUpdate,"__call__",corrupted)
    if bad=="objective":
        evidence_call=app.Evidence.__call__
        def objective(self,vertices):
            total,parts=evidence_call(self,vertices)
            return (total*float("inf"),parts) if vertices.requires_grad else (total,parts)
        monkeypatch.setattr(app.Evidence,"__call__",objective)
    reports=[]
    for mode in ("existing","packed"):
        config=configuration(tmp_path,source,mode);config.trial_diagnostics=mode
        reports.append(app.optimize(config))
    assert_same_records(*reports)
    assert all(r["failed_trials"]==4 and r["gradient_steps"]==0 for r in reports)
    assert all(r["saved_binary_certificate"]["valid"] for r in reports)
    assert reports[0]["evaluations"]==reports[1]["evaluations"]==4
    with np.load(tmp_path/"existing.npz") as a,np.load(tmp_path/"packed.npz") as b:
        np.testing.assert_array_equal(a["vertices"],b["vertices"])


@pytest.mark.parametrize("change",[
    {"trial_diagnostics":"wrong"},{"geometry_backend":"existing"},{"method":"f2"},
    {"control_hierarchy":"nested_p1"},{"coordinate_mode":"joint"},{"fine_patch_cells":2},
    {"proposal_filter_steps":1},{"interpolation":"p1_bd"},{"precision":"float32"},
    {"loss":"local_ncc"},{"mind_order":"after_warp"},{"image_weight":0.},
])
def test_unsupported_packed_configuration_refused_before_loading(tmp_path,monkeypatch,change):
    source=assets(tmp_path);config=configuration(tmp_path,source,"unused");config.trial_diagnostics="packed"
    for key,value in change.items():setattr(config,key,value)
    monkeypatch.setattr(app,"load_registration_evidence",lambda *a,**kw:pytest.fail("loaded unsupported configuration"))
    with pytest.raises(ValueError):app.optimize(config)
