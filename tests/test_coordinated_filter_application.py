"""Filter raw coefficients, retain original map/acceptance and full gradients."""
import numpy as np
import pytest

from tools.coordinated_real_case import optimize
from test_coordinated_nested_application import configuration


@pytest.mark.parametrize("coordinate_mode",["alternating","joint"])
@pytest.mark.parametrize("steps",[0,4])
def test_filter_pipeline_scales_only_requested_levels_and_keeps_topology(tmp_path,coordinate_mode,steps):
    args=configuration(tmp_path,coordinate_mode=coordinate_mode,geometry_backend="existing",
        proposal_filter_steps=steps,proposal_filter_min_level=9,inner_steps=3,control_hierarchy="fixed")
    report=optimize(args)
    assert report["gradient_steps"]==(6 if coordinate_mode=="joint" else 12)
    assert report["failed_trials"]==0 and report["saved_binary_certificate"]["valid"]
    assert report["proposal_filter_steps"]==steps
    for row in report["trace"]:
        assert row["proposal_filter_steps"]==(steps if row["level"]>=9 else 0)
        assert row["filtered_coefficient_rms"]<=row["raw_coefficient_rms"]+1e-14
        if row["proposal_filter_steps"]==0:
            assert row["filtered_coefficient_rms"]==row["raw_coefficient_rms"]
        assert row["candidate_displacement_rms"]>=0
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    with np.load(args.output) as data:
        vertices,reference=data["vertices"],data["boundary_reference"]
        np.testing.assert_array_equal(vertices[:,[0,-1]],reference[:,[0,-1]])
        np.testing.assert_array_equal(vertices[:,:,[0,-1]],reference[:,:,[0,-1]])


@pytest.mark.parametrize("changes",[dict(proposal_filter_steps=-1),dict(proposal_filter_steps=True),
    dict(proposal_filter_steps=1.5),dict(proposal_filter_min_level=2),dict(method="f1"),dict(method="regional_analytic")])
def test_bad_filter_configuration_is_rejected(tmp_path,changes):
    values=dict(proposal_filter_steps=4);values.update(changes)
    with pytest.raises(ValueError):optimize(configuration(tmp_path,**values))
