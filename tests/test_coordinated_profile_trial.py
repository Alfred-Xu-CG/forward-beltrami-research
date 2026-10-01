import importlib.util

import torch


def test_fixed_anchor_profile_uses_real_evidence_and_chain_rule():
    name="tools.coordinated_profile_trial"
    assert importlib.util.find_spec(name) is not None,"trial profiler not implemented"
    from tools.coordinated_profile_trial import profile_one_trial
    from tools.coordinated_real_case import Evidence
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    axis=torch.linspace(0,1,9,dtype=torch.float64)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    anchor=torch.stack((x+.01*torch.sin(torch.pi*x).square()*torch.sin(torch.pi*y).square(),y),-1)[None]
    generator=torch.Generator().manual_seed(35)
    fixed=torch.rand(1,1,16,16,generator=generator)
    points=torch.rand(12,2,generator=generator,dtype=torch.float64)*.8+.1
    matches=ImageCorrespondences(points,points,torch.ones(12,dtype=torch.float64),pixel_scale=16,robust_scale=8)
    evidence=Evidence(fixed,fixed.clone(),torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        "mind",3.,1.,.0001,interpolation="p1_ac",matches=matches,match_weight=.1)
    evidence.prepare_fixed_p1_sampling(9,9,dtype=anchor.dtype,device=anchor.device)
    result=profile_one_trial(anchor,evidence,level=5,amplitude=.002,warmup=10,repeats=10)
    assert result["coefficient_parameters"]==9
    assert result["anchor_requires_grad"] is False
    assert result["chain_rule_gradient_max_difference"]<1e-12
    assert result["finite_gradient"] is True
    assert result["actual_normalized_margin"]>0
    assert len(result["decoder_forward_seconds"])==10
    assert set(result["objective_parts"])=={"image","strain","shape","oob","outside_fraction","match"}
    try:
        profile_one_trial(anchor.requires_grad_(),evidence,level=5)
    except ValueError as error:
        assert "fixed" in str(error)
    else:
        raise AssertionError("anchor gradient silently dropped")
