import importlib.util

import pytest
import torch


@pytest.mark.parametrize("side,nested,coefficient_level",[(3,True,3),(5,True,5),(9,True,9),(9,False,5)])
def test_nested_profiler_preserves_full_fine_evidence_gradient_and_counts(side,nested,coefficient_level):
    name="tools.coordinated_profile_nested_trial"
    assert importlib.util.find_spec(name) is not None,"nested trial profiler not implemented"
    from tools.coordinated_profile_nested_trial import diagnostic_anchor,profile_nested_trial
    from tools.coordinated_real_case import Evidence
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    generator=torch.Generator().manual_seed(71)
    image=torch.rand(1,1,16,16,generator=generator)
    points=torch.rand(12,2,generator=generator,dtype=torch.float64)*.8+.1
    matches=ImageCorrespondences(points,points,torch.ones(12,dtype=torch.float64),pixel_scale=16,robust_scale=8)
    evidence=Evidence(image,image.clone(),torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        "mind",3.,1.,.0001,interpolation="p1_ac",matches=matches,match_weight=.1)
    evidence.prepare_fixed_p1_sampling(9,9,dtype=torch.float64,device="cpu")
    anchor=diagnostic_anchor(side,device="cpu")
    result=profile_nested_trial(anchor,evidence,fine_side=9,nested=nested,
        coefficient_level=coefficient_level,warmup=3,repeats=10)
    assert result["fine_side"]==9 and result["control_side"]==side
    assert result["coefficient_parameters"]==(coefficient_level-2)**2
    assert result["finite_gradient"] and result["fresh_actual_normalized_qmin"]>.001
    assert result["chain_rule_gradient_max_difference"]<1e-12
    assert result["chain_rule_gradient_relative_l2_error"]<1e-10
    assert result["fixed_boundary_equal"] and result["materializer_constant_bytes"]>=0
    assert len(result["full_forward_seconds"])==len(result["full_vjp_seconds"])==10
    assert set(result["objective_parts"])=={"image","strain","shape","match","oob","outside_fraction"}
    assert (result["fine_margin_check_enabled"] is True)==nested
    assert result["anchor_source"]=="synthetic legal smooth residual, NOT optimizer trajectory"
    with pytest.raises(ValueError,match="constant"):
        profile_nested_trial(anchor.requires_grad_(),evidence,fine_side=9,nested=nested)


def test_per_term_fine_vertex_vjps_match_declared_weighted_complete_objective():
    from tools.coordinated_profile_nested_trial import diagnostic_anchor,profile_evidence_terms
    from tools.coordinated_real_case import Evidence
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    generator=torch.Generator().manual_seed(9)
    image=torch.rand(1,1,16,16,generator=generator)
    points=torch.rand(12,2,generator=generator,dtype=torch.float64)*.8+.1
    matches=ImageCorrespondences(points,points,torch.ones(12,dtype=torch.float64),pixel_scale=16,robust_scale=8)
    evidence=Evidence(image,image.clone(),torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        "mind",3.,1.,.0001,interpolation="p1_ac",matches=matches,match_weight=.1)
    evidence.prepare_fixed_p1_sampling(9,9,dtype=torch.float64,device="cpu")
    result=profile_evidence_terms(diagnostic_anchor(9,device="cpu"),evidence,warmup=3,repeats=10)
    assert result["gradient_variable"]=="fine_vertices"
    assert set(result["terms"])=={"image","strain","shape","match","oob"}
    assert result["weighted_sum_gradient_relative_l2_error"]<1e-10
    assert result["weighted_sum_gradient_maximum_absolute_error"]<1e-12
    assert result["weights"]==dict(image=1.,strain=3.,shape=.0001,match=.1,oob=1.)
    for term in result["terms"].values():
        assert term["finite_gradient"] and len(term["vjp_seconds"])==10
