"""Explicit joint export routing; original frozen affine rule remains intact."""
import json
import numpy as np
import pytest

from tools.coordinated_miit_score import _analytic_map
from tools.coordinated_lung_all20_score import _safe_map


def save_fixture(tmp_path,*,joint=True,forged_original=False,forged_combined=False):
    from tools.coordinated_joint_pose import combined_affine
    import torch
    a=np.array([[.9,.04],[-.03,.92]],dtype=np.float32)
    b=np.array([.03,.04],dtype=np.float32)
    y,x=np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3),indexing="ij")
    identity=np.stack((x,y),-1)[None]
    physical=np.array([.02,-.01,.1,.02,.03,-.01]) if joint else np.zeros(6)
    out,offset=combined_affine(torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64),torch.tensor(physical))
    out=out.numpy();offset=offset.numpy()
    if forged_combined:out=out*1.1  # Positive orientation, but inconsistent with the stored pose.
    np.savez(tmp_path/"map.npz",vertices=identity,boundary_reference=identity,
        post_affine_matrix=out,post_affine_offset=offset,
        original_post_affine_matrix=a+(.01 if forged_original else 0),original_post_affine_offset=b,
        pose_parameters=physical,interpolation=np.array("p1_ac"))
    config=dict(interpolation="p1_ac",minimum_jacobian=.001)
    record=dict(output="map.npz",report="map.json")
    if joint:config["pose_mode"]=record["pose_mode"]="joint_positive_affine"
    (tmp_path/"map.json").write_text(json.dumps(dict(configuration=config,landmarks_used=False)),encoding="utf-8")
    return record,a,b


def test_joint_accepts_valid_changed_affine_only_explicitly(tmp_path):
    record,a,b=save_fixture(tmp_path)
    vertices,metadata=_analytic_map(record,tmp_path,a,b)
    assert np.isfinite(vertices).all() and metadata["joint_export_validation"]["valid"]
    with pytest.raises(ValueError,match="common affine"):_safe_map(record,tmp_path,a,b)
    implicit={k:v for k,v in record.items() if k!="pose_mode"}
    with pytest.raises(ValueError,match="common affine"):_analytic_map(implicit,tmp_path,a,b)


@pytest.mark.parametrize("forgery",["original","combined"])
def test_joint_rejects_forged_affines(tmp_path,forgery):
    record,a,b=save_fixture(tmp_path,forged_original=forgery=="original",forged_combined=forgery=="combined")
    with pytest.raises(ValueError,match="original common affine" if forgery=="original" else "validation failed"):
        _analytic_map(record,tmp_path,a,b)


def test_default_frozen_map_matches_unchanged_safe_reader(tmp_path):
    record,a,b=save_fixture(tmp_path,joint=False)
    old,oldmeta=_safe_map(record,tmp_path,a,b)
    actual,metadata=_analytic_map(record,tmp_path,a,b)
    np.testing.assert_array_equal(actual,old)
    assert metadata==oldmeta
