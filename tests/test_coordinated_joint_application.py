"""Joint coordinates are a stage organization, not an image-driven guarantee."""
import numpy as np
import pytest

from tools.coordinated_real_case import optimize
from test_coordinated_nested_application import configuration


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("backend",["ordinary","cached_manual"])
@pytest.mark.parametrize("hierarchy",["fixed","nested_p1"])
def test_actual_joint_pipeline_keeps_budget_both_substep_checks_and_output(tmp_path,mode,backend,hierarchy):
    args=configuration(tmp_path,coordinate_mode="joint",joint_backend=backend,
        geometry_backend="existing",control_hierarchy=hierarchy,inner_steps=4,method=mode)
    report=optimize(args)
    assert report["gradient_steps"]==8 and report["evaluations"]==10
    assert report["coordinated_substep_evaluations"]==20
    assert report["extra_joint_diagnostic_passes"]==(10 if backend=="cached_manual" else 0)
    assert len(report["stages"])==2 and all(s["direction"]=="joint_xy" for s in report["stages"])
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    assert all(t["intermediate_margin"]>0 and t["margin"]>0 for t in report["trace"])
    assert report["failed_trials"]==0 and report["saved_binary_certificate"]["valid"]
    with np.load(args.output) as data:assert data["vertices"].shape==(1,9,9,2)


def test_joint_nested_reduced_evidence_retains_original_acceptance(tmp_path):
    args=configuration(tmp_path,coordinate_mode="joint",joint_backend="cached_manual",
        geometry_backend="existing",nested_evaluation="coarse_exact",inner_steps=4)
    report=optimize(args)
    assert report["gradient_steps"]==8 and report["saved_binary_certificate"]["valid"]
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    assert all("rounded_full_stage_fallback" in s for s in report["stages"])


@pytest.mark.parametrize("changes",[dict(method="f1"),dict(method="regional_analytic"),
    dict(geometry_backend="stage_cache"),dict(joint_backend="wrong")])
def test_joint_rejects_incompatible_flags_before_running(tmp_path,changes):
    values=dict(coordinate_mode="joint",geometry_backend="existing");values.update(changes)
    with pytest.raises(ValueError):optimize(configuration(tmp_path,**values))


def test_valid_final_map_does_not_hide_bad_intermediate_margin(tmp_path,monkeypatch):
    """Injected diagnostic failure: isolates the application guard, not geometry."""
    from qcopt.neural_bijection.dense.coordinated_joint_stage import FrozenAnchorJointCoordinatedUpdate
    original=FrozenAnchorJointCoordinatedUpdate.__call__
    def bad_intermediate(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        assert (result.normalized_margin_min>0).all()
        result.substep_margin_min[:,0]=0
        return result
    monkeypatch.setattr(FrozenAnchorJointCoordinatedUpdate,"__call__",bad_intermediate)
    report=optimize(configuration(tmp_path,coordinate_mode="joint",geometry_backend="existing",inner_steps=4))
    assert report["gradient_steps"]==0 and report["failed_trials"]==2
    assert report["median_vjp_seconds"] is None
    assert all(f["intermediate_margin"]==0 and f["margin"]>0 for f in report["failures"])
    assert report["selected_stage"] is None and report["saved_binary_certificate"]["valid"]
    assert all(s["accepted_total"]==s["anchor_total"] for s in report["stages"])
