"""Descriptor order is shared at all stages, never a change of geometry."""
import pytest

from tools.coordinated_real_case import optimize
from test_coordinated_nested_application import configuration


@pytest.mark.parametrize("method",["radial","analytic","f1","f2"])
def test_after_warp_runs_actual_safe_pipeline_with_common_prior(tmp_path,method):
    args=configuration(tmp_path,mind_order="after_warp",strain_model="p1_arap",control_hierarchy="fixed",
        method=method,geometry_backend="stage_cache" if method in ("radial","analytic") else "existing",
        patch_cells=4,f2_accepted_gain=1.,inner_steps=3)
    report=optimize(args)
    assert report["mind_order"]=="after_warp" and report["strain_model"]=="p1_arap"
    assert report["image_levels"]==[8,16] and report["failed_trials"]==0
    assert report["saved_binary_certificate"]["valid"]
    assert report["gradient_steps"]==(12 if method in ("radial","analytic") else 6)
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])


@pytest.mark.parametrize("changes",[dict(mind_order="wrong"),dict(mind_order="after_warp",loss="local_ncc")])
def test_descriptor_order_bad_config_rejected(tmp_path,changes):
    with pytest.raises(ValueError):optimize(configuration(tmp_path,**changes))
