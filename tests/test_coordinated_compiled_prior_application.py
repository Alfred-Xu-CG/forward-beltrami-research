"""Application dispatch correctness; eager compiler capture is NOT Inductor speed evidence."""
import argparse

import pytest
import torch

from tools import coordinated_real_case as app
from tools import coordinated_compiled_priors as prior
from tools.coordinated_lung_all20 import make_configuration
from test_coordinated_lung_all20 import assets


def capture_factory(monkeypatch):
    original=prior.make_joint_priors
    calls=[]
    def capture(backend):
        assert backend=="inductor"
        function=original("eager")
        calls.append(function)
        return function
    monkeypatch.setattr(prior,"make_joint_priors",capture)
    return calls


def test_complete_evidence_values_vertex_vjp_and_all_parts_unchanged(monkeypatch):
    calls=capture_factory(monkeypatch)
    generator=torch.Generator().manual_seed(433)
    fixed=torch.rand((1,1,16,16),generator=generator)
    moving=torch.rand((1,1,16,16),generator=generator)
    matrix=torch.eye(2,dtype=torch.float64)
    offset=torch.tensor([.01,-.01],dtype=torch.float64)
    y,x=torch.meshgrid(torch.linspace(0,1,9,dtype=torch.float64),
        torch.linspace(0,1,9,dtype=torch.float64),indexing="ij")
    vertices=torch.stack((x,y),-1)[None]
    vertices[:,1:-1,1:-1]+=torch.randn((1,7,7,2),generator=generator,dtype=vertices.dtype)*.002
    objects=[app.Evidence(fixed,moving,matrix,offset,"mind",3.,1.,1e-4,
        interpolation="p1_ac",strain_model="p1_arap",joint_prior_backend=backend)
        for backend in ("eager","inductor")]
    results=[]
    for evidence in objects:
        evidence.prepare_fixed_p1_sampling(9,9,dtype=vertices.dtype,device=vertices.device)
        argument=vertices.clone().requires_grad_()
        total,parts=evidence(argument)
        gradient,=torch.autograd.grad(total,argument)
        results.append((total.detach(),{k:v.detach() for k,v in parts.items()},gradient))
    assert len(calls)==1
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-12,atol=1e-14)
    assert results[0][1].keys()==results[1][1].keys()
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key],results[1][1][key],rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(results[0][2],results[1][2],rtol=1e-12,atol=1e-14)


def test_tiny_whole_optimizer_shares_one_callable_and_retains_all_guards(tmp_path,monkeypatch):
    source=assets(tmp_path)
    pair=dict(name="he_to_cc10",fixed=source.canvas/"cc10_fixed512.png",
        moving=source.canvas/"cc10_moving512.png",
        affine=source.affines_from/"he_to_cc10_affine.npz")
    calls=capture_factory(monkeypatch)
    settings=argparse.Namespace(output=tmp_path,device="cpu",threads=1)
    reports=[]
    for backend in ("eager","inductor"):
        config=make_configuration(pair,"analytic",settings,production=False)
        config.matches=None;config.match_weight=0.
        config.output=tmp_path/(backend+".npz")
        config.joint_prior_backend=backend
        reports.append(app.optimize(config))
    assert len(calls)==1  # Not one factory per image-resolution Evidence object.
    assert all(report["gradient_steps"]==4 and report["failed_trials"]==0 for report in reports)
    assert all(report["saved_binary_certificate"]["valid"] for report in reports)
    assert reports[1]["configuration"]["joint_prior_backend"]=="inductor"
    assert reports[1]["joint_prior_backend"]=="inductor"
    assert reports[0]["final"]==reports[1]["final"]


@pytest.mark.parametrize("change",[
    {"joint_prior_backend":"bad"},
    {"interpolation":"p1_bd"},
    {"shape_weight":0.},
    {"strain_model":"displacement_gradient"},
    {"precision":"float32"},
    {"control_hierarchy":"nested_p1"},
])
def test_unsupported_compiled_configuration_rejected_before_image_loading(tmp_path,change):
    settings=argparse.Namespace(output=tmp_path,device="cpu",threads=1)
    pair=dict(name="unused",fixed=tmp_path/"missing.png",moving=tmp_path/"missing.png",
        affine=tmp_path/"missing.npz")
    config=make_configuration(pair,"analytic",settings,production=False)
    config.joint_prior_backend="inductor"
    for key,value in change.items():setattr(config,key,value)
    with pytest.raises(ValueError,match="joint priors|joint_prior_backend"):
        app.optimize(config)


def test_compilation_failure_is_not_silently_replaced_by_eager(monkeypatch):
    def fail(backend):raise RuntimeError("deliberate compiler failure")
    monkeypatch.setattr(prior,"make_joint_priors",fail)
    raster=torch.ones((1,1,8,8))
    with pytest.raises(RuntimeError,match="compiler failure"):
        app.Evidence(raster,raster,torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
            "mind",3.,1.,1e-4,interpolation="p1_ac",strain_model="p1_arap",joint_prior_backend="inductor")
