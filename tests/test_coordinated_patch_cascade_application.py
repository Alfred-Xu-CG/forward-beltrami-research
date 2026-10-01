"""Dense-only support changes must preserve the complete instance objective."""
from dataclasses import replace

import pytest
import torch

from tools.coordinated_real_case import optimize
from test_coordinated_nested_application import configuration


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("backend",["ordinary","manual"])
def test_fine_only_cascade_counts_all_passes_and_retains_fine_acceptance(tmp_path,mode,diagonal,backend):
    report=optimize(configuration(tmp_path,method=mode,fine_patch_cells=2,
        fine_patch_backend=backend,strain_model="p1_arap",nested_evaluation="coarse_exact",interpolation="p1_"+diagonal))
    assert report["gradient_steps"]==8 and report["failed_trials"]==0
    assert report["evaluations"]==12 and report["coordinated_substep_evaluations"]==30
    assert report["fine_patch_cells"]==2 and report["saved_binary_certificate"]["valid"]
    assert report["fine_patch_backend"]==backend
    assert ("first-order" in report["geometry_backend_scope"])==(backend=="manual")
    for trial in report["trace"]:
        assert trial["margin"]>0
        if trial["level"]==9:
            assert trial["geometry_pass_count"]==4 and trial["intermediate_margin"]>0
        else:
            assert "geometry_pass_count" not in trial
    assert all(stage["accepted_total"]<=stage["anchor_total"] for stage in report["stages"])


@pytest.mark.parametrize("patch",[-1,True,1,3,8])
def test_bad_fine_patch_geometry_rejected(tmp_path,patch):
    with pytest.raises(ValueError,match="fine_patch_cells"):
        optimize(configuration(tmp_path,fine_patch_cells=patch))


@pytest.mark.parametrize("backend",["unsupported",None,False])
def test_bad_backend_rejected(tmp_path,backend):
    with pytest.raises(ValueError,match="fine_patch_backend"):
        optimize(configuration(tmp_path,fine_patch_cells=2,fine_patch_backend=backend))


def test_manual_backend_without_fine_patches_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError,match="requires active"):
        optimize(configuration(tmp_path,fine_patch_backend="manual"))


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_ordinary_manual_actual_pipeline_agree(tmp_path,mode):
    import numpy as np
    reports=[];maps=[]
    for backend in ("ordinary","manual"):
        args=configuration(tmp_path,method=mode,fine_patch_cells=2,fine_patch_backend=backend,
            strain_model="p1_arap",nested_evaluation="coarse_exact",output=tmp_path/(backend+".npz"))
        reports.append(optimize(args))
        with np.load(args.output) as data:
            maps.append(data["vertices"].copy())
    np.testing.assert_allclose(maps[0],maps[1],rtol=1e-12,atol=1e-12)
    assert reports[0]["final"]["total"]==pytest.approx(reports[1]["final"]["total"],abs=1e-12)
    assert reports[0]["gradient_steps"]==reports[1]["gradient_steps"]
    assert reports[0]["selected_stage"]==reports[1]["selected_stage"]


@pytest.mark.parametrize("changes",[dict(method="f1"),dict(coordinate_mode="joint")])
def test_unsupported_other_decoders_rejected(tmp_path,changes):
    with pytest.raises(ValueError,match="fine_patch_cells"):
        optimize(configuration(tmp_path,fine_patch_cells=2,geometry_backend="existing",**changes))


@pytest.mark.parametrize("bad_margin",[0.,float("nan"),float("inf")])
def test_invalid_intermediate_pass_is_rejected_even_with_valid_final_map(tmp_path,monkeypatch,bad_margin):
    from qcopt.neural_bijection.dense.coordinated_patch_cascade import CoordinatedPatchCascade
    original=CoordinatedPatchCascade.forward
    def corrupt_margin(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        margins=result.pass_margin_min.clone()
        margins[:,1]=bad_margin
        assert bool((result.normalized_margin_min>0).all())
        return replace(result,pass_margin_min=margins)
    monkeypatch.setattr(CoordinatedPatchCascade,"forward",corrupt_margin)
    report=optimize(configuration(tmp_path,fine_patch_cells=2,strain_model="p1_arap",
        nested_evaluation="coarse_exact"))
    assert report["failed_trials"]==2 and report["gradient_steps"]==4
    assert report["coordinated_substep_evaluations"]==14
    assert all(trial["intermediate_margin"]<=0 or not trial["intermediate_margins_finite"] for trial in report["failures"])
    assert report["saved_binary_certificate"]["valid"]
