"""Algebra/gradient and application checks, not anatomical efficacy evidence."""
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_multiscale_evidence import SimultaneousImageEvidence
from tools import coordinated_real_case as app
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from test_coordinated_shared_affine_evidence import identity,configuration


def pyramid():
    gen=torch.Generator().manual_seed(4821)
    fixed=torch.rand(1,1,31,31,generator=gen,dtype=torch.float64)*.8+.1
    moving=fixed.roll(2,-1)
    mask=torch.rand(1,1,31,31,generator=gen,dtype=torch.float64)
    a=torch.tensor([[1.07,.03],[-.02,.98]],dtype=torch.float64)
    b=torch.tensor([.015,-.023],dtype=torch.float64)
    p=torch.tensor([[.237,.349],[.724,.671]],dtype=torch.float64)
    matches=ImageCorrespondences(p,p+.015,torch.ones(2,dtype=torch.float64))
    result={}
    for side in (11,19,31):
        resize=lambda x:F.interpolate(x,(side,side),mode="area")
        result[side]=app.Evidence(resize(fixed),resize(moving),a,b,"mind",3.,1.,1e-4,
            fixed_mask=resize(mask),interpolation="p1_ac",matches=matches,match_weight=.1,
            strain_model="p1_arap",mind_frame="shared_affine")
        result[side].prepare_fixed_p1_sampling(7,9,dtype=torch.float64,device="cpu")
    return result


def deformed():
    v=identity(7,9)
    v[:,1:-1,1:-1]+=torch.tensor([.0143,-.0091],dtype=v.dtype)
    return v.requires_grad_()


def test_non_aligned_fractional_masks_weighted_values_and_full_map_gradients():
    objects=pyramid();v=deformed();combined=SimultaneousImageEvidence(objects)
    actual,parts=combined(v)
    # Independent sum of COMPLETE single-raster functionals with all nonimage
    # contributions removed, then add one finest set back explicitly.
    images=[]
    for obj in objects.values():
        _,p=obj(v);images.append(p["image"])
    _,fine=objects[31](v)
    expected=sum(images)/3+3*fine["strain"]+1e-4*fine["shape"]+.1*fine["match"]+fine["oob"]
    ga,=torch.autograd.grad(actual,v,retain_graph=True)
    ge,=torch.autograd.grad(expected,v)
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(ga,ge,rtol=2e-13,atol=2e-13)
    for key in ("strain","shape","match","oob","outside_fraction"):
        assert torch.equal(parts[key],fine[key])
    assert len({float(e.denominator) for e in objects.values()})==3


def test_zero_coarse_weights_reproduce_reference_fine_objective_and_gradient_bitwise():
    objects=pyramid();v=deformed()
    combined=SimultaneousImageEvidence(objects,{11:0.,19:0.,31:1.})
    actual,parts=combined(v);expected,fine=objects[31](v)
    ga,=torch.autograd.grad(actual,v,retain_graph=True)
    ge,=torch.autograd.grad(expected,v)
    assert torch.equal(actual,expected) and torch.equal(ga,ge)
    assert parts.keys()==fine.keys() and all(torch.equal(parts[k],fine[k]) for k in parts)


def test_no_coarse_priors_or_matches_are_called(monkeypatch):
    objects=pyramid()
    # __call__ is the only full-objective path; lower rasters may only use
    # image_terms. Replacing their prior machinery must therefore have no effect.
    def forbidden(*args,**kwargs):pytest.fail("duplicated prior/point evaluation")
    for side in (11,19):
        objects[side].joint_p1_priors=forbidden
        objects[side].matches=forbidden
    result,parts=SimultaneousImageEvidence(objects)(deformed())
    assert torch.isfinite(result) and set(("image_11","image_19","image_31"))<=parts.keys()


def test_directional_finite_difference_on_non_aligned_queries():
    obj=SimultaneousImageEvidence(pyramid())
    # The symmetric constant-translation fixture lands on bilinear/L1 knots;
    # perturb it deterministically before a two-sided derivative comparison.
    v=(deformed().detach()+torch.randn(1,7,9,2,
        generator=torch.Generator().manual_seed(721),dtype=torch.float64)*.001).requires_grad_()
    value,_=obj(v);g,=torch.autograd.grad(value,v)
    d=torch.randn(v.shape,generator=torch.Generator().manual_seed(82),dtype=v.dtype)*.1
    eps=1e-7
    fd=(obj(v+eps*d)[0]-obj(v-eps*d)[0])/(2*eps)
    torch.testing.assert_close((g*d).sum(),fd,rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize("method",["analytic","f2"])
def test_tiny_whole_optimizer_all_stage_objective_budgets_and_topology(tmp_path,method):
    cfg=configuration(tmp_path,method);cfg.mind_frame="shared_affine"
    cfg.image_objective="simultaneous_multiscale"
    report=app.optimize(cfg)
    assert report["gradient_steps"]==4 and report["evaluations"]==8
    assert report["objective_evaluations"]==18 and report["failed_trials"]==0
    assert report["saved_binary_certificate"]["valid"]
    assert report["image_objective_scales"]==[8,16]
    assert report["image_objective_weights"]=={8:.5,16:.5}
    assert all("image_8" in t and "image_16" in t for t in report["trace"])
    totals=[s["accepted_full_total"] for s in report["stages"]]
    assert all(b<=a+1e-14 for a,b in zip([report["initial"]["total"]]+totals,totals))
    assert all(s["accepted_total"]==s["accepted_full_total"] for s in report["stages"])


@pytest.mark.parametrize("weights",[{11:-.1,19:.1,31:1.},{11:.2,19:.2,31:.2},{11:0.,19:1.}])
def test_bad_weights_rejected(weights):
    with pytest.raises(ValueError):SimultaneousImageEvidence(pyramid(),weights)
