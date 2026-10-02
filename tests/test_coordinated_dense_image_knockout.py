"""Structural image-term knockout retains the other functional and map guards."""
import argparse

import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools import coordinated_real_case as app
from tools.coordinated_application_profile import load_configuration
from test_coordinated_application_profile import fixture_report


def test_complete_value_and_full_vertex_vjp_difference_is_only_dense_image():
    generator=torch.Generator().manual_seed(841)
    fixed=torch.rand(1,1,16,16,generator=generator)
    moving=torch.rand(1,1,16,16,generator=generator)
    axis=torch.linspace(0,1,9,dtype=torch.float64)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    vertices=torch.stack((x,y),-1)[None]
    vertices[:,1:-1,1:-1]+=torch.randn(1,7,7,2,generator=generator,dtype=torch.float64)*.002
    points=torch.tensor([[.2,.2],[.4,.7],[.8,.8]],dtype=torch.float64)
    matches=ImageCorrespondences(points,points+.01,torch.ones(3,dtype=torch.float64))
    matrix=torch.tensor([[1.1,.05],[-.02,.98]],dtype=torch.float64)
    offset=torch.tensor([.01,-.01],dtype=torch.float64)
    results=[]
    for beta in (1.,0.):
        evidence=app.Evidence(fixed,moving,matrix,offset,"mind",3.,1.,1e-4,
            interpolation="p1_ac",strain_model="p1_arap",matches=matches,match_weight=.1,image_weight=beta)
        evidence.prepare_fixed_p1_sampling(9,9,dtype=vertices.dtype,device=vertices.device)
        value=vertices.clone().requires_grad_()
        total,parts=evidence(value)
        grad,=torch.autograd.grad(total,value,retain_graph=True)
        image_grad,=torch.autograd.grad(parts["image"],value)
        results.append((total,parts,grad,image_grad))
    hybrid,knockout=results
    torch.testing.assert_close(knockout[0],hybrid[0]-hybrid[1]["image"],rtol=1e-13,atol=1e-15)
    torch.testing.assert_close(knockout[2],hybrid[2]-hybrid[3],rtol=1e-11,atol=1e-13)
    for key in hybrid[1]:
        assert torch.equal(hybrid[1][key],knockout[1][key])
    assert hybrid[1]["image"].item()>0 and knockout[1]["match"].item()>0


def test_tiny_optimizer_dense_knockout_budget_shared_pyramid_and_default(tmp_path):
    source=fixture_report(tmp_path)
    config=load_configuration(source,tmp_path/"default.npz",production=False)
    config.device="cpu"
    default=app.optimize(config)
    config.output=tmp_path/"one.npz";config.image_weight=1.
    explicit=app.optimize(config)
    assert default["final"]==explicit["final"]
    with np.load(tmp_path/"default.npz") as a,np.load(config.output) as b:
        assert np.array_equal(a["vertices"],b["vertices"])
    config.output=tmp_path/"zero.npz";config.image_weight=0.
    knockout=app.optimize(config)
    assert knockout["gradient_steps"]==4 and knockout["failed_trials"]==0
    assert knockout["saved_binary_certificate"]["valid"]
    assert knockout["image_weight"]==knockout["configuration"]["image_weight"]==0.
    assert knockout["final"]["image"]>0
    parts=knockout["final"]
    assert parts["total"]==pytest.approx(3*parts["strain"]+parts["oob"]+1e-4*parts["shape"]+.1*parts["match"],rel=1e-13)
    # This fixture's machine points are identity correspondences. With the
    # dense image term removed, retaining identity is the correct optimum.
    assert len(knockout["stages"])==4
    assert knockout["selected_stage"] is None
    with np.load(config.output) as saved:
        assert np.array_equal(saved["vertices"],saved["boundary_reference"])


@pytest.mark.parametrize("value",[-1.,float("nan"),float("inf"),True,"zero"])
def test_invalid_image_weight_rejected_before_input_loading(tmp_path,value):
    from tools.coordinated_lung_all20 import make_configuration
    config=make_configuration(dict(name="missing",fixed=tmp_path/"missing.png",
        moving=tmp_path/"missing.png",affine=tmp_path/"missing.npz"),"analytic",
        argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    config.image_weight=value
    with pytest.raises(ValueError,match="image_weight"):
        app.optimize(config)
