"""Safe proposal acceptance is separate from discrete feature searching."""
import pytest
import torch

from tools.coordinated_capture_prefix import apply_capture_proposal
from tools.coordinated_geometry_pilot import reference_grid
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


def inputs():
    reference = reference_grid(9)
    proposal = torch.zeros_like(reference)
    proposal[:, 1:-1, 1:-1] = torch.tensor([.01, -.02], dtype=reference.dtype)
    return reference, proposal


def test_two_axes_refresh_geometry_and_keep_strict_boundary():
    reference, proposal = inputs()
    target = reference + proposal
    seen = []
    def evidence(candidate):
        seen.append(candidate.clone())
        value = (candidate-target).square().sum()
        return value, {"image": value}
    initial = float(evidence(reference)[0]); seen.clear()
    result, report = apply_capture_proposal(reference, reference, proposal, evidence, anchor_total=initial)
    assert report["accepted_axes"] == report["objective_evaluations"] == 2
    assert report["failed_attempts"] == 0
    assert report["accepted_total"] < initial
    assert not torch.equal(seen[0][..., 0], reference[..., 0])
    assert torch.equal(seen[1][..., 0], seen[0][..., 0])
    torch.testing.assert_close(result, target, rtol=0, atol=1e-15)
    for side in (result[:, 0]-reference[:, 0], result[:, -1]-reference[:, -1],
                 result[:, :, 0]-reference[:, :, 0], result[:, :, -1]-reference[:, :, -1]):
        assert not bool(side.any())
    assert float((q1_corner_determinants(result)/q1_corner_determinants(reference)).min()) > .001


def test_objective_worse_rejects_both_without_ladder_or_trial_compounding():
    reference, proposal = inputs()
    seen = []
    def evidence(candidate):
        seen.append(candidate.clone())
        value = (candidate-reference).square().sum()
        return value, {"image": value}
    result, report = apply_capture_proposal(reference, reference, proposal, evidence, anchor_total=0.)
    assert report["accepted_axes"] == 0 and report["geometry_attempts"] == 2
    assert torch.equal(result, reference)
    assert torch.equal(seen[1][..., 0], reference[..., 0])


def test_zero_tie_is_not_an_accepted_update():
    reference, proposal = inputs(); proposal.zero_()
    def evidence(candidate):return torch.tensor(1.), {"image": torch.tensor(1.)}
    result, report = apply_capture_proposal(reference, reference, proposal, evidence, anchor_total=1.)
    assert torch.equal(result, reference) and report["accepted_axes"] == 0


def test_nonfinite_objective_preserves_valid_anchor_and_records_failure():
    reference, proposal = inputs()
    def evidence(candidate):return torch.tensor(float("nan")), {}
    result, report = apply_capture_proposal(reference, reference, proposal, evidence, anchor_total=1.)
    assert torch.equal(result, reference) and report["failed_attempts"] == 2
    assert report["objective_evaluations"] == 2
    assert all("nonfinite" in row["error"] for row in report["attempts"])


def test_invalid_coarse_boundary_refused():
    reference, proposal = inputs(); proposal[:, 0, 1, 0] = .01
    with pytest.raises(ValueError, match="boundary"):
        apply_capture_proposal(reference, reference, proposal, lambda _: None, anchor_total=1.)


def test_zero_prefix_retains_whole_cached_optimizer_map_and_counts_extra_calls(tmp_path, monkeypatch):
    import argparse
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from test_coordinated_lung_all20 import assets
    from tools.coordinated_lung_all20 import make_configuration
    from tools import coordinated_real_case as app
    from qcopt.neural_bijection.dense import coordinated_discrete_capture as capture
    source=assets(tmp_path)
    fixed=source.canvas/"cc10_fixed512.png"; moving=source.canvas/"cc10_moving512.png"
    for path in (fixed,moving):
        with Image.open(path) as image: resized=image.resize((128,128))
        resized.save(path)
    config=make_configuration(dict(name="fixture",fixed=fixed,moving=moving,
        affine=source.affines_from/"he_to_cc10_affine.npz"),"analytic",
        argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    config.grid_side=33; config.levels=[17,33]; config.image_side=128; config.image_levels=[32,128]
    config.matches=None; config.match_weight=0.
    calls=[]
    def zero(f,m,mask,A,b,*,geometry_dtype):
        assert f.shape[-2:]==m.shape[-2:]==mask.shape[-2:]==(128,128)
        calls.append(True)
        return SimpleNamespace(proposal=torch.zeros(1,33,33,2,dtype=geometry_dtype),
                               diagnostics={"fixture":"zero proposal, no search claim"})
    monkeypatch.setattr(capture,"discrete_capture_proposal",zero)
    reports=[]; maps=[]
    for mode in ("none","mind_discrete"):
        config.capture_prefix=mode; config.output=tmp_path/(mode+".npz")
        report=app.optimize(config); reports.append(report)
        with np.load(config.output) as saved: maps.append(saved["vertices"].copy())
    assert len(calls)==1 and np.array_equal(maps[0],maps[1])
    assert reports[0]["final"]==reports[1]["final"]
    assert all(r["geometry_backend"]=="stage_cache" for r in reports)
    assert all(r["gradient_steps"]==4 and r["failed_trials"]==0 for r in reports)
    assert reports[1]["objective_evaluations"]==reports[0]["objective_evaluations"]+2
    assert reports[1]["capture_record"]["accepted_axes"]==0
    assert reports[1]["capture_record"]["complete_prefix_seconds"]>=reports[1]["capture_record"]["search_seconds"]>=0
    assert reports[1]["capture_record"]["failed_attempts"]==0
    assert reports[1]["saved_binary_certificate"]["valid"]


@pytest.mark.parametrize("change",[
    {"capture_prefix":"invalid"},{"method":"f2"},{"interpolation":"q1"},
    {"image_weight":0.},{"oob_weight":0.},{"mind_order":"after_warp"},
    {"image_levels":[32,64,64,256,512]},
])
def test_unsupported_capture_configuration_rejected_before_loading(tmp_path,change):
    import argparse
    from tools.coordinated_lung_all20 import make_configuration
    from tools.coordinated_real_case import optimize
    config=make_configuration(dict(name="missing",fixed=tmp_path/"missing.png",
        moving=tmp_path/"missing.png",affine=tmp_path/"missing.npz"),"analytic",
        argparse.Namespace(output=tmp_path,device="cpu",threads=1))
    # Avoid earlier unrelated cache/ARAP guards so this fixture actually reaches
    # the capture guard; production experiments still use their original recipe.
    config.geometry_backend="existing"; config.strain_model="displacement_gradient"
    config.capture_prefix="mind_discrete"
    for key,value in change.items():setattr(config,key,value)
    with pytest.raises(ValueError,match="capture"):
        optimize(config)
