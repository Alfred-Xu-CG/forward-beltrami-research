"""Only source query geometry is cached; gradients remain through map/affine."""
import copy
import argparse
import json

import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries


def geometry(rows=5,columns=7):
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),
                       torch.linspace(0,1,columns,dtype=torch.float64),indexing="ij")
    return torch.stack((x+.03*x*y,y+.02*x*(1-x)),dim=-1)[None]


def evidence():
    # Corners, grid vertices, horizontal/vertical edges, both diagonal edges,
    # and strict triangle interiors on a nonsquare 5x7 grid.
    source=torch.tensor([[0.,0.],[1.,1.],[0.,1.],[1.,0.],[.5,.5],
        [1/6,.25],[.09,.25],[.5,.31],[1/12,.125],[.11,.08],[.04,.18]],dtype=torch.float64,requires_grad=True)
    target=(source.detach()*.96+.01).requires_grad_()
    confidence=torch.linspace(.2,1.,len(source),dtype=torch.float64,requires_grad=True)
    return ImageCorrespondences(source,target,confidence),source,target,confidence


def value_grads(layer,vertices,matrix,interpolation):
    vertices=vertices.detach().clone().requires_grad_()
    matrix=matrix.detach().clone().requires_grad_()
    value=layer(vertices,matrix,interpolation)
    gradients=torch.autograd.grad(value,(vertices,matrix))
    return value.detach(),tuple(gradient.detach() for gradient in gradients)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_values_full_vertex_and_affine_gradients_edges_and_changed_maps(diagonal):
    baseline,source,target,confidence=evidence()
    cached=copy.deepcopy(baseline)
    metadata=cached.prepare_fixed_p1_sampling(5,7,diagonal)
    assert metadata["queries"]==len(source) and metadata["resident_bytes"]>0
    assert not any(buffer.requires_grad for buffer in cached.buffers())
    assert "fixed_p1_evaluator.weights" in dict(cached.named_buffers())
    matrix=torch.tensor([[1.1,.13],[-.07,.9]],dtype=torch.float64)
    for vertices in (geometry(),geometry()+torch.tensor([.007,-.004])):
        left=value_grads(baseline,vertices,matrix,"p1_"+diagonal)
        right=value_grads(cached,vertices,matrix,"p1_"+diagonal)
        torch.testing.assert_close(left[0],right[0],rtol=2e-13,atol=2e-13)
        for a,b in zip(left[1],right[1]):
            torch.testing.assert_close(a,b,rtol=3e-12,atol=3e-12)
    assert source.grad is None and target.grad is None and confidence.grad is None
    assert cached.fixed_p1_evaluator.vertex_ids.shape==(len(source),3)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_joint_vertex_affine_directional_finite_difference(diagonal):
    layer,*_=evidence();layer.prepare_fixed_p1_sampling(5,7,diagonal)
    vertices=geometry().requires_grad_()
    matrix=torch.tensor([[1.1,.13],[-.07,.9]],dtype=torch.float64,requires_grad=True)
    generator=torch.Generator().manual_seed(203)
    dv=torch.randn(vertices.shape,generator=generator,dtype=torch.float64)*.01
    da=torch.randn(matrix.shape,generator=generator,dtype=torch.float64)*.03
    gv,ga=torch.autograd.grad(layer(vertices,matrix,"p1_"+diagonal),(vertices,matrix))
    h=1e-6
    finite=(layer(vertices+h*dv,matrix+h*da,"p1_"+diagonal)-
            layer(vertices-h*dv,matrix-h*da,"p1_"+diagonal))/(2*h)
    torch.testing.assert_close((gv*dv).sum()+(ga*da).sum(),finite,rtol=2e-7,atol=2e-9)


@pytest.mark.parametrize("interpretation",["q1","p1_bd"])
def test_prepared_diagonal_or_q1_misuse_errors(interpretation):
    layer,*_=evidence();layer.prepare_fixed_p1_sampling(5,7,"ac")
    with pytest.raises(ValueError,match="grid/diagonal"):
        layer(geometry(),torch.eye(2,dtype=torch.float64),interpretation)


def test_wrong_grid_and_explicit_reprepare_and_buffer_cast():
    layer,*_=evidence();layer.prepare_fixed_p1_sampling(5,7,"ac")
    with pytest.raises(ValueError,match="grid/diagonal"):
        layer(geometry(6,7),torch.eye(2,dtype=torch.float64),"p1_ac")
    layer.prepare_fixed_p1_sampling(6,7,"bd")
    layer=layer.float()
    assert layer.fixed_p1_evaluator.weights.dtype==torch.float32
    assert torch.isfinite(layer(geometry(6,7).float(),torch.eye(2),"p1_bd"))
    with pytest.raises(ValueError,match="matching"):
        layer(geometry(6,7),torch.eye(2,dtype=torch.float64),"p1_bd")


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_default_is_exact_old_unprepared_arithmetic(diagonal):
    layer,*_=evidence();vertices=geometry();matrix=torch.eye(2,dtype=torch.float64)
    mapped=p1_map_at_queries(vertices,layer.source,diagonal,validate_queries=False)
    error=(mapped-layer.target)@matrix.T*(layer.pixel_scale/layer.robust_scale)
    squared=error.square().sum(-1)[0]
    old=(squared/(torch.sqrt(1+squared)+1)*layer.weights).sum()
    assert torch.equal(old,layer(vertices,matrix,"p1_"+diagonal))
    assert layer.fixed_p1_sampling_report()==dict(prepared=False,backend="existing")


def test_operator_benchmark_frozen_raw_fixture_reports_setup_and_ten_pairs(tmp_path):
    from test_coordinated_application_profile import fixture_report
    from tools import coordinated_frozen_correspondence_benchmark as benchmark
    predictions=fixture_report(tmp_path)
    args=argparse.Namespace(anchor=predictions.parent/"he_to_cc10_analytic.npz",
        matches=predictions.parent/"he_to_cc10_raw_matches.json",output=tmp_path/"point_benchmark.json",
        device="cpu",threads=1)
    report=benchmark.run(args)
    assert report["status"]=="complete" and report["inductor_used"] is False
    assert report["images_annotations_matcher_read"] is False
    assert report["preparation"]["queries"]==9
    assert report["preparation_seconds"]>0 and report["loading_seconds"]>0
    assert len(report["samples"])==28 and len(report["comparisons"])==14
    assert all(row["passed"] for row in report["comparisons"])
    assert len([row for row in report["samples"] if row["phase"]=="timed"])==20
    assert json.loads(args.output.read_text())["geometry_shape"]==[1,9,9,2]
    with pytest.raises(FileExistsError):
        benchmark.run(args)


@pytest.mark.parametrize("requested,index",[("cuda",0),("cuda:3",3)])
def test_benchmark_cuda_initialization_resolves_default_and_explicit_index(monkeypatch,requested,index):
    from tools import coordinated_frozen_correspondence_benchmark as benchmark
    calls=[]
    monkeypatch.setattr(torch.cuda,"set_device",lambda value:calls.append(("set_device",value)))
    monkeypatch.setattr(torch.cuda,"init",lambda:calls.append(("init",)))
    assert benchmark.initialize_device(torch.device(requested))==torch.device("cuda",index)
    assert calls==[("set_device",index),("init",)]
