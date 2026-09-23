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
