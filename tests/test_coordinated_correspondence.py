import json
from pathlib import Path

import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools.coordinated_real_case import Evidence,load_image_matches


def reference(side=9):
    axis=torch.linspace(0,1,side,dtype=torch.float64)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    return torch.stack((x,y),-1)[None]


@pytest.mark.parametrize("interpretation",["q1","p1_ac","p1_bd"])
def test_independent_physical_affine_pseudohuber(interpretation):
    source=torch.tensor([[.13,.27],[.64,.76]],dtype=torch.float64)
    target=source+torch.tensor([[.02,-.01],[-.03,.04]],dtype=torch.float64)
    confidence=torch.tensor([.3,.9],dtype=torch.float64)
    matrix=torch.tensor([[1.2,.2],[-.1,.8]],dtype=torch.float64)
    layer=ImageCorrespondences(source,target,confidence,pixel_scale=512.,robust_scale=8.)
    residual=(source.numpy()-target.numpy())@matrix.numpy().T*64
    expected=((np.sqrt(1+(residual**2).sum(-1))-1)*np.array([.25,.75])).sum()
    assert float(layer(reference(),matrix,interpretation))==pytest.approx(expected,abs=1e-13)


@pytest.mark.parametrize("interpretation",["q1","p1_ac","p1_bd"])
def test_correspondence_vjp_directional_fd(interpretation):
    torch.manual_seed(477)
    source=torch.tensor([[.137,.279],[.643,.761]],dtype=torch.float64)
    target=source+torch.tensor([[.02,-.01],[-.03,.04]],dtype=torch.float64)
    layer=ImageCorrespondences(source,target,torch.tensor([.6,.4],dtype=torch.float64))
    vertices=(reference()+.001*torch.randn_like(reference())).requires_grad_()
    direction=.005*torch.randn_like(vertices)
    matrix=torch.tensor([[.9,.1],[-.2,1.1]],dtype=torch.float64)
    grad,=torch.autograd.grad(layer(vertices,matrix,interpretation),vertices)
    step=1e-6
    finite=(layer(vertices+step*direction,matrix,interpretation)-layer(vertices-step*direction,matrix,interpretation))/(2*step)
    torch.testing.assert_close((grad*direction).sum(),finite,rtol=2e-7,atol=2e-9)


def test_match_exact_fit_stationary_and_confidence_fixed():
    points=torch.tensor([[.13,.27],[.64,.76]],dtype=torch.float64,requires_grad=True)
    confidence=torch.tensor([.3,.9],dtype=torch.float64,requires_grad=True)
    layer=ImageCorrespondences(points,points,confidence)
    vertices=reference().requires_grad_()
    value=layer(vertices,torch.eye(2,dtype=torch.float64),"p1_ac")
    grad,=torch.autograd.grad(value,vertices)
    assert float(value)<1e-28 and float(grad.abs().max())<1e-11
    assert not layer.source.requires_grad and not layer.weights.requires_grad


def match_record(matrix,offset):
    source=np.tile(np.array([[.2,.3]]),(9,1))
    target=source.copy();target[-1]=[.99,.3]
    return dict(targets_manual_landmarks_or_dense_teacher_loaded=False,image_side=512,
        fixed="data/fixed.png",moving="data/moving.png",post_affine_matrix=matrix.tolist(),
        post_affine_offset=offset.tolist(),source_points_unit=source.tolist(),
        target_points_unit=target.tolist(),confidence=[.9]*9)


def test_loader_static_domain_and_float_affine_cast(tmp_path):
    matrix=np.array([[1.111111111111,0.],[0.,1.]],dtype=np.float64)
    offset=np.array([.02,0.],dtype=np.float64)
    path=tmp_path/"matches.json";path.write_text(json.dumps(match_record(matrix,offset)))
    module,metadata=load_image_matches(path,matrix.astype(np.float32),offset.astype(np.float32),
        fixed_path=Path("another/fixed.png"),moving_path=Path("another/moving.png"),
        image_side=512,device=torch.device("cpu"),dtype=torch.float64,robust_scale=8.)
    assert metadata["raw_matches"]==9 and metadata["eligible_matches"]==8
    assert metadata["static_original_moving_domain_excluded"]==1
    assert float(module.weights[-1])==0 and float(module.weights.sum())==pytest.approx(1.)
    assert "not content identity proof" in metadata["path_check"]


@pytest.mark.parametrize("prefix",["D:\\data\\native\\","/home/research/native/"])
def test_cross_host_transported_record_and_robust_scale_invariance(tmp_path,prefix):
    matrix=np.eye(2,dtype=np.float64);offset=np.zeros(2,dtype=np.float64)
    record=match_record(matrix,offset)
    old=tmp_path/"old.json";old.write_text(json.dumps(record))
    base,_=load_image_matches(old,matrix,offset,fixed_path=Path("fixed.png"),moving_path=Path("moving.png"),
        image_side=512,device="cpu",dtype=torch.float64,robust_scale=8.)
    record.update(image_side=1024,fixed=prefix+"fixed.png",moving=prefix+"moving.png",
        prediction_side=512,origin_image_side=512,transport_factor=2,
        transport_method="exact integer normalized-frame transport")
    new=tmp_path/"new.json";new.write_text(json.dumps(record))
    transported,metadata=load_image_matches(new,matrix,offset,fixed_path=Path("fixed.png"),moving_path=Path("moving.png"),
        image_side=1024,device="cpu",dtype=torch.float64,robust_scale=16.)
    vertices=reference().requires_grad_()
    before=base(vertices,torch.eye(2,dtype=torch.float64),"p1_ac")
    after=transported(vertices,torch.eye(2,dtype=torch.float64),"p1_ac")
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(before,vertices)[0],torch.autograd.grad(after,vertices)[0],rtol=0,atol=0)
    assert metadata["prediction_side"]==512 and metadata["transport_factor"]==2


@pytest.mark.parametrize("corruption",["affine","provenance","image","confidence"])
def test_loader_invalid_evidence_rejected(tmp_path,corruption):
    matrix=np.eye(2,dtype=np.float32);offset=np.zeros(2,dtype=np.float32)
    record=match_record(matrix,offset)
    if corruption=="affine":record["post_affine_matrix"][0][0]=1.1
    if corruption=="provenance":record["targets_manual_landmarks_or_dense_teacher_loaded"]=True
    if corruption=="image":record["fixed"]="wrong.png"
    if corruption=="confidence":record["confidence"][-1]=1.2
    path=tmp_path/"matches.json";path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_image_matches(path,matrix,offset,fixed_path=Path("fixed.png"),moving_path=Path("moving.png"),
            image_side=512,device=torch.device("cpu"),dtype=torch.float64,robust_scale=8.)


def test_shared_evidence_includes_same_frozen_match_term():
    image=torch.rand(1,1,16,16,dtype=torch.float64)
    source=torch.tensor([[.13,.27],[.64,.76]],dtype=torch.float64)
    matches=ImageCorrespondences(source,source+.01,torch.ones(2,dtype=torch.float64))
    matrix=torch.eye(2,dtype=torch.float64);offset=torch.zeros(2,dtype=torch.float64)
    base=Evidence(image,image,matrix,offset,"mind",.05,1.,interpolation="p1_ac")
    augmented=Evidence(image,image,matrix,offset,"mind",.05,1.,interpolation="p1_ac",matches=matches,match_weight=.1)
    plain,parts=base(reference());total,augparts=augmented(reference())
    torch.testing.assert_close(total-plain,.1*augparts["match"],rtol=1e-12,atol=1e-14)
    for name in parts:torch.testing.assert_close(parts[name],augparts[name],rtol=0,atol=0)
