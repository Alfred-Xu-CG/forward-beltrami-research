"""Reusable nested P1 layer certifies its inputs/outputs and transmits VJP."""

import math

import pytest
import torch

from qcopt.neural_bijection.dense import NestedP1PhotometricFeedbackLayer


def test_nested_p1_feedback_certificate_and_vjp():
    torch.set_num_threads(4)
    side=9
    line=torch.linspace(0,1,side)
    yy,xx=torch.meshgrid(line,line,indexing="ij")
    envelope=torch.sin(math.pi*xx)*torch.sin(math.pi*yy)
    coarse=torch.stack((xx+0.01*envelope,yy-0.01*envelope),dim=-1)[None]
    coarse=coarse.detach().requires_grad_()
    grid=torch.linspace(0,1,64)
    gy,gx=torch.meshgrid(grid,grid,indexing="ij")
    moving=(torch.sin(8*gx)+torch.cos(6*gy))[None,None].float().requires_grad_()
    fixed=(moving.detach()+0.01*torch.sin(4*gx)[None,None]).requires_grad_()
    layer=NestedP1PhotometricFeedbackLayer(
        side,33,fine_passes=1,qc_cap=0.795,
        spectral_modes=16)
    with pytest.raises(RuntimeError,match="prepare"):
        layer(fixed,moving,coarse)
    layer.prepare(device="cpu")
    mapped=layer(fixed,moving,coarse)
    assert mapped.shape==(1,33,33,2)
    assert layer.last_stats["minimum_signed_area_ratio"]>0
    assert layer.last_stats["maximum_beltrami_modulus"]<0.795
    gradient=torch.autograd.grad(
        mapped.square().mean(),(coarse,fixed,moving))
    assert all(torch.isfinite(value).all() for value in gradient)
    assert all(value.abs().amax()>0 for value in gradient)
    invalid=coarse.detach().clone()
    invalid[:,0,1,1]+=0.01
    with pytest.raises(RuntimeError,match="boundary"):
        layer(fixed.detach(),moving.detach(),invalid)
    strict=NestedP1PhotometricFeedbackLayer(
        side,33,fine_passes=1,qc_cap=0.01,
        spectral_modes=16)
    strict.prepare(device="cpu")
    with pytest.raises(RuntimeError,match="QC cap"):
        strict(fixed.detach(),moving.detach(),coarse.detach())


def test_checkpoint_modes_preserve_output_and_input_vjp():
    torch.set_num_threads(4)
    axis=torch.linspace(0,1,9)
    yy,xx=torch.meshgrid(axis,axis,indexing="ij")
    envelope=torch.sin(math.pi*xx)*torch.sin(math.pi*yy)
    coarse_base=torch.stack((xx+0.01*envelope,yy-0.01*envelope),dim=-1)[None]
    grid=torch.linspace(0,1,64)
    gy,gx=torch.meshgrid(grid,grid,indexing="ij")
    moving_base=(torch.sin(8*gx)+torch.cos(6*gy))[None,None].float()
    fixed_base=moving_base+0.01*torch.sin(4*gx)[None,None]
    outputs=[]
    for mode in ("refiner","full","none"):
        coarse=coarse_base.detach().clone().requires_grad_()
        fixed=fixed_base.detach().clone().requires_grad_()
        moving=moving_base.detach().clone().requires_grad_()
        layer=NestedP1PhotometricFeedbackLayer(
            9,33,fine_passes=1,qc_cap=0.795,spectral_modes=16,
            checkpoint_refiner=mode=="refiner",
            checkpoint_full_pass=mode=="full")
        layer.prepare(device="cpu")
        mapped=layer(fixed,moving,coarse)
        gradients=torch.autograd.grad(
            mapped.square().mean()+0.1*mapped[...,0].mean(),
            (coarse,fixed,moving))
        outputs.append((mapped.detach(),tuple(value.detach() for value in gradients)))
    reference,reference_gradients=outputs[0]
    for mapped,gradients in outputs[1:]:
        torch.testing.assert_close(mapped,reference,rtol=0,atol=1e-6)
        for actual,expected in zip(gradients,reference_gradients):
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prepare_generic_cuda_name_uses_current_device():
    layer=NestedP1PhotometricFeedbackLayer(5,9,fine_passes=0).cuda()
    layer.prepare(device="cuda")
    assert layer._prepared_device==torch.device("cuda",torch.cuda.current_device())
    axis=torch.linspace(0,1,5,device="cuda")
    yy,xx=torch.meshgrid(axis,axis,indexing="ij")
    coarse=torch.stack((xx,yy),dim=-1)[None]
    image=torch.zeros((1,1,16,16),device="cuda")
    mapped=layer(image,image,coarse)
    assert mapped.shape==(1,9,9,2)
    assert layer.last_stats["minimum_signed_area_ratio"]>0
