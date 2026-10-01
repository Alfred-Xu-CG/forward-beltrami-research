"""Same fine functional via coarse P1 queries, independently finite-differenced."""
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_nested_refinement import FrozenNestedP1Refinement
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools.coordinated_real_case import Evidence,optimize
from test_coordinated_nested_application import configuration


def fixture(diagonal,loss,image_dtype=torch.float64):
    torch.manual_seed(391)
    axis=torch.arange(5,dtype=torch.float64)/4
    yy,xx=torch.meshgrid(axis,axis,indexing="ij")
    vertices=torch.stack((xx,yy),-1)[None]
    vertices[:,1:-1,1:-1]+=.017*torch.randn(1,3,3,2,dtype=torch.float64)
    # Pixel centers and points include original diagonals, edges and vertices.
    source=torch.tensor([[.25,.25],[0.,0.],[1.,1.],[.4375,.5625],[.6,.7]],dtype=vertices.dtype)
    matches=ImageCorrespondences(source,source.clone(),torch.ones(5,dtype=vertices.dtype),
        pixel_scale=16.,robust_scale=3.)
    fixed=torch.rand(1,1,16,16,dtype=image_dtype)
    moving=torch.rand_like(fixed)
    matrix=torch.tensor([[1.13,.09],[-.08,.92]],dtype=vertices.dtype)
    offset=torch.tensor([-.045173,.064831],dtype=vertices.dtype)
    evidence=Evidence(fixed,moving,matrix,offset,loss,3.,1.,.0001,
        interpolation="p1_"+diagonal,matches=matches,match_weight=.1)
    evidence.prepare_fixed_p1_sampling(17,17,dtype=vertices.dtype,device="cpu")
    return vertices,evidence,FrozenNestedP1Refinement(5,17,diagonal=diagonal)


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("loss",["mind","local_ncc"])
def test_complete_values_all_y_derivatives_and_directional_fd(diagonal,loss):
    vertices,evidence,refine=fixture(diagonal,loss)
    reduced=evidence.coarse_nested_evidence(5,17,dtype=vertices.dtype,device="cpu")
    assert reduced.fixed_feature is evidence.fixed_feature and reduced.moving_feature is evidence.moving_feature
    assert reduced.matches is evidence.matches and reduced.mask is evidence.mask
    assert evidence.nested_priors is None and evidence.fixed_p1_evaluator.rows==17
    vertices.requires_grad_()
    old,op=evidence(refine(vertices));new,np=reduced(vertices)
    for key in op:
        torch.testing.assert_close(np[key],op[key],rtol=2e-10,atol=1e-12)
    og,=torch.autograd.grad(old,vertices);ng,=torch.autograd.grad(new,vertices)
    torch.testing.assert_close(ng,og,rtol=2e-9,atol=2e-11)
    tangent=torch.randn_like(vertices) # Includes boundary derivatives.
    eps=1e-7
    finite=(reduced(vertices.detach()+eps*tangent)[0]-reduced(vertices.detach()-eps*tangent)[0])/(2*eps)
    torch.testing.assert_close(finite,(ng*tangent).sum(),rtol=2e-5,atol=2e-7)
    assert float(np["outside_fraction"])>0


def test_bilinear_knot_keeps_same_selected_subgradient_not_central_fd():
    vertices,evidence,refine=fixture("ac","local_ncc")
    # This rational affine puts THREE boundary-query components on exact image
    # knots. Central FD averages branches; grid_sample AD selects one branch.
    evidence.offset=torch.tensor([-.045,.065],dtype=vertices.dtype)
    reduced=evidence.coarse_nested_evidence(5,17,dtype=vertices.dtype,device="cpu")
    vertices.requires_grad_();old,_=evidence(refine(vertices));new,_=reduced(vertices)
    og,=torch.autograd.grad(old,vertices);ng,=torch.autograd.grad(new,vertices)
    torch.testing.assert_close(ng,og,rtol=2e-9,atol=2e-11)
    coords=(reduced.fixed_p1_evaluator(vertices)@evidence.matrix.T+evidence.offset)*16-.5
    assert int(((coords-coords.round()).abs()<1e-12).sum())==3
    tangent=torch.randn_like(vertices);h=1e-8
    central=(reduced(vertices.detach()+h*tangent)[0]-reduced(vertices.detach()-h*tangent)[0])/(2*h)
    assert abs(float(central-(ng*tangent).sum()))>1e-3


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_mixed_image_precision_records_floating_rounding(diagonal,record_property):
    vertices,evidence,refine=fixture(diagonal,"mind",torch.float32)
    reduced=evidence.coarse_nested_evidence(5,17,dtype=vertices.dtype,device="cpu")
    vertices.requires_grad_();a,_=evidence(refine(vertices));b,_=reduced(vertices)
    ga,=torch.autograd.grad(a,vertices);gb,=torch.autograd.grad(b,vertices)
    error=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(ga))
    record_property("mixed_full_gradient_relative_error",error)
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8)
    assert error<2e-5


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_tiny_actual_pipeline_keeps_fine_checks_and_original_acceptance(tmp_path,diagonal):
    args=configuration(tmp_path,nested_evaluation="coarse_exact",interpolation="p1_"+diagonal)
    report=optimize(args)
    assert report["gradient_steps"]==8 and report["failed_trials"]==0
    assert report["saved_binary_certificate"]["valid"] and report["control_vertices"]==81
    assert report["nested_evidence_cache_bytes"]>0
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    assert all(t["margin"]>0 for t in report["trace"])
    assert all("rounded_full_stage_fallback" in s for s in report["stages"])


def test_reject_q1_or_nonnested_reduced_evaluation(tmp_path):
    vertices,evidence,_=fixture("ac","mind")
    evidence.interpolation="q1"
    with pytest.raises(ValueError):evidence.coarse_nested_evidence(5,17,dtype=vertices.dtype,device="cpu")
    with pytest.raises(ValueError):optimize(configuration(tmp_path,nested_evaluation="coarse_exact",control_hierarchy="fixed"))


def test_original_stage_objective_rejects_spurious_reduced_winner(tmp_path,monkeypatch):
    """Deliberate disagreement fixture for acceptance, NOT real image evidence."""
    from tools.digital_q1_dhr_distill import identity_vertices
    def objective(self,vertices):
        if self.nested_priors is not None:
            total=-vertices[:,1:-1,1:-1].sum()
        else:
            total=(vertices-identity_vertices(9,device="cpu").double()).square().sum()
        zero=total*0
        return total,dict(image=total,strain=zero,oob=zero,outside_fraction=zero)
    monkeypatch.setattr(Evidence,"__call__",objective)
    report=optimize(configuration(tmp_path,nested_evaluation="coarse_exact",inner_steps=1))
    assert all(s["rounded_full_stage_fallback"] for s in report["stages"])
    assert all(s["accepted_total"]==s["anchor_total"]==0 for s in report["stages"])
    assert report["final"]["total"]==0 and report["selected_stage"] is None
    assert report["saved_binary_certificate"]["valid"]
