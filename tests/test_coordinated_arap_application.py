"""A prior ablation must retain evidence, geometry and acceptance conventions."""
import pytest
import torch

from tools.coordinated_real_case import Evidence,optimize
from tools.digital_mind_safe_optimize import strain_penalty
from test_coordinated_nested_application import configuration
from test_coordinated_reduced_evidence import fixture


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_only_prior_changes_complete_functional_and_full_vertex_derivative(diagonal):
    from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
    vertices,old,refine=fixture(diagonal,"mind")
    import copy
    new=copy.copy(old);new.strain_model="p1_arap"
    vertices=refine(vertices).detach().requires_grad_()
    original,original_parts=old(vertices);changed,changed_parts=new(vertices)
    for key in ("image","oob","shape","match","outside_fraction"):
        torch.testing.assert_close(changed_parts[key],original_parts[key],rtol=0,atol=0)
    explicit=p1_arap_energy(vertices,diagonal=diagonal)-strain_penalty(vertices)
    torch.testing.assert_close(changed-original,3*explicit,rtol=1e-12,atol=1e-14)
    delta_grad,=torch.autograd.grad(changed-original,vertices,retain_graph=True)
    prior_grad,=torch.autograd.grad(3*explicit,vertices)
    torch.testing.assert_close(delta_grad,prior_grad,rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize("method",["radial","analytic","f1","f2"])
def test_actual_tiny_arap_pipeline_all_geometry_methods(tmp_path,method):
    args=configuration(tmp_path,strain_model="p1_arap",control_hierarchy="fixed",
        method=method,geometry_backend="stage_cache" if method in ("radial","analytic") else "existing",
        patch_cells=4,f2_accepted_gain=1.,inner_steps=3)
    report=optimize(args)
    assert report["strain_model"]=="p1_arap" and report["failed_trials"]==0
    assert report["saved_binary_certificate"]["valid"]
    assert report["gradient_steps"]==(12 if method in ("radial","analytic") else 6)
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    assert all(t["margin"]>0 for t in report["trace"])


@pytest.mark.parametrize("changes",[dict(strain_model="wrong"),dict(interpolation="q1")])
def test_incompatible_prior_configuration_rejected(tmp_path,changes):
    values=dict(strain_model="p1_arap");values.update(changes)
    with pytest.raises(ValueError):optimize(configuration(tmp_path,**values))


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_reduced_evidence_uses_current_arap_not_membrane(diagonal):
    vertices,evidence,refine=fixture(diagonal,"mind");evidence.strain_model="p1_arap"
    reduced=evidence.coarse_nested_evidence(5,17,dtype=torch.float64,device="cpu")
    assert reduced.nested_priors.strain_model=="p1_arap"
    vertices.requires_grad_()
    fine,fine_parts=evidence(refine(vertices));coarse,coarse_parts=reduced(vertices)
    for name in fine_parts:
        torch.testing.assert_close(fine_parts[name],coarse_parts[name],rtol=2e-10,atol=1e-12)
    torch.testing.assert_close(fine,coarse,rtol=2e-10,atol=1e-12)
    fg,=torch.autograd.grad(fine,vertices);cg,=torch.autograd.grad(coarse,vertices)
    torch.testing.assert_close(fg,cg,rtol=2e-9,atol=2e-11)


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("method",["radial","analytic"])
def test_actual_nested_arap_pipeline_keeps_fine_validation_and_acceptance(tmp_path,diagonal,method):
    report=optimize(configuration(tmp_path,strain_model="p1_arap",nested_evaluation="coarse_exact",
        interpolation="p1_"+diagonal,method=method))
    assert report["strain_model"]=="p1_arap"
    assert report["gradient_steps"]==8 and report["failed_trials"]==0
    assert report["saved_binary_certificate"]["valid"]
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    assert all(t["margin"]>0 for t in report["trace"])
    assert all("rounded_full_stage_fallback" in s for s in report["stages"])
