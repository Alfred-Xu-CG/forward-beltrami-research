import importlib.util

import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from tools.coordinated_real_case import corner_symmetric_dirichlet
from tools.digital_mind_safe_optimize import strain_penalty


def prior_api():
    name="qcopt.neural_bijection.dense.coordinated_nested_priors"
    assert importlib.util.find_spec(name) is not None,"nested priors not implemented"
    from qcopt.neural_bijection.dense.coordinated_nested_priors import exact_nested_p1_priors,ExactNestedP1Priors
    return exact_nested_p1_priors,ExactNestedP1Priors


def fixture(case,dtype=torch.float64):
    if case=="quad":
        return torch.tensor([[[[0.,0.],[1.,.1]],[[-.1,1.],[1.1,1.2]]]],dtype=dtype)
    if case=="near":
        return torch.tensor([[[[0.,0.],[1.,0.]],[[0.,1.],[1.,.0012]]]],dtype=dtype)
    yy,xx=torch.meshgrid(torch.arange(5,dtype=torch.float32)/4,torch.arange(5,dtype=torch.float32)/4,indexing="ij")
    reference=torch.stack((xx,yy),-1)[None].to(dtype)
    if case=="affine":
        return reference@torch.tensor([[1.2,.15],[-.1,.9]],dtype=dtype).T+torch.tensor([.03,-.02],dtype=dtype)
    generator=torch.Generator().manual_seed(37)
    out=reference.repeat(2,1,1,1)
    out[:,1:-1,1:-1]+=.018*torch.randn(2,3,3,2,generator=generator,dtype=dtype)
    return out


@pytest.mark.parametrize("case",["quad","affine","random","near"])
@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("factor",[1,2,4])
def test_exact_fine_functional_values_and_full_y_gradients(case,diagonal,factor,record_property):
    api,cls=prior_api();vertices=fixture(case).requires_grad_()
    side=factor*(vertices.shape[1]-1)+1
    fine=refine_p1_vertices(vertices,factor,diagonal)
    full=(strain_penalty(fine),corner_symmetric_dirichlet(fine))
    reduced=api(vertices,side,diagonal)
    module=cls(vertices.shape[1],side,diagonal=diagonal,dtype=vertices.dtype)
    cached=module(vertices)
    for name,expected,actual,m in zip(("strain","shape"),full,reduced,cached):
        torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-12)
        torch.testing.assert_close(actual,m,rtol=0,atol=0)
        fg,=torch.autograd.grad(expected,vertices,retain_graph=True)
        rg,=torch.autograd.grad(actual,vertices,retain_graph=True)
        if case=="near":
            absolute=float((rg-fg).abs().max())
            relative=float(torch.linalg.vector_norm(rg-fg)/torch.linalg.vector_norm(fg))
            record_property(f"near_{name}_gradient_maximum_absolute_rounding_error",absolute)
            record_property(f"near_{name}_gradient_relative_l2_error",relative)
            full_entry=float(fg[0,0,0,1]);reduced_entry=float(rg[0,0,0,1])
            record_property(f"near_{name}_small_corner_y_full_gradient",full_entry)
            record_property(f"near_{name}_small_corner_y_reduced_gradient",reduced_entry)
            record_property(f"near_{name}_small_corner_y_relative_error",
                abs(full_entry-reduced_entry)/max(abs(full_entry),torch.finfo(fg.dtype).tiny))
            # Explicit fine backward cancels ~1e9 contributions into ~1-sized
            # entries: an elementwise tiny-relative tolerance is not justified.
            # Keep/report the absolute error; test the complete derivative norm.
            assert relative<2e-10
        else:
            torch.testing.assert_close(rg,fg,rtol=2e-10,atol=2e-10)


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("case",["random","near"])
def test_directional_fd_and_first_order_all_boundary_entries(diagonal,case):
    api,_=prior_api();vertices=fixture(case).requires_grad_()
    fine_side=4*(vertices.shape[1]-1)+1
    strain,shape=api(vertices,fine_side,diagonal)
    gradient,=torch.autograd.grad(3*strain+.0001*shape,vertices)
    tangent=torch.randn(vertices.shape,generator=torch.Generator().manual_seed(9),dtype=vertices.dtype)
    h=1e-8 if case=="near" else 1e-6
    plus=api(vertices.detach()+h*tangent,fine_side,diagonal)
    minus=api(vertices.detach()-h*tangent,fine_side,diagonal)
    fd=(3*(plus[0]-minus[0])+.0001*(plus[1]-minus[1]))/(2*h)
    torch.testing.assert_close(fd,(gradient*tangent).sum(),rtol=2e-6 if case=="near" else 2e-8,atol=2e-9)
    assert (gradient[:,0]!=0).any() and (gradient[:,:,-1]!=0).any()


@pytest.mark.parametrize("case",["random","near"])
@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_float32_rounding_discrepancy_is_measured_not_bitwise_claim(case,diagonal,record_property):
    api,_=prior_api();vertices=fixture(case,torch.float32).requires_grad_()
    side=4*(vertices.shape[1]-1)+1
    fine=refine_p1_vertices(vertices,4,diagonal)
    full=(strain_penalty(fine),corner_symmetric_dirichlet(fine));small=api(vertices,side,diagonal)
    for name,expected,actual in zip(("strain","shape"),full,small):
        fg,=torch.autograd.grad(expected,vertices,retain_graph=True)
        rg,=torch.autograd.grad(actual,vertices,retain_graph=True)
        value_error=float((actual-expected).abs()/expected.abs().clamp_min(torch.finfo(vertices.dtype).tiny))
        grad_error=float(torch.linalg.vector_norm(rg-fg)/torch.linalg.vector_norm(fg).clamp_min(torch.finfo(vertices.dtype).tiny))
        record_property(f"{name}_relative_value_error",value_error)
        record_property(f"{name}_relative_gradient_error",grad_error)
        print(dict(case=case,diagonal=diagonal,prior=name,relative_value_error=value_error,relative_gradient_error=grad_error))
        assert torch.isfinite(actual) and torch.isfinite(rg).all() and torch.isfinite(fg).all()
        if case=="random":
            assert value_error<1e-4 and grad_error<1e-4
        # Thin f32 materialized cells amplify rounding; no equivalence tolerance
        # is silently widened to pretend the rounded function is unchanged.


@pytest.mark.parametrize("case",["nondyadic_coarse","nondyadic_fine","smaller","diagonal","non_square","batch","dtype","buffer_dtype","trainable_reference"])
def test_guards(case):
    api,cls=prior_api();vertices=fixture("random");side=17;diagonal="ac"
    if case=="nondyadic_coarse":vertices=torch.zeros(1,4,4,2,dtype=torch.float64)
    elif case=="nondyadic_fine":side=13
    elif case=="smaller":side=3
    elif case=="diagonal":diagonal="wrong"
    elif case=="non_square":vertices=vertices[:,:,:-1]
    elif case=="batch":vertices=vertices[:0]
    elif case=="dtype":vertices=vertices.half()
    with pytest.raises(ValueError):
        if case=="buffer_dtype":cls(5,17,dtype=torch.float32)(vertices)
        elif case=="trainable_reference":
            module=cls(5,17);module.source_reference.requires_grad_();module(vertices)
        else:api(vertices,side,diagonal)
