"""Independent quarter-turn frame/ranking tests; no installed matcher invoked."""
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import pytest

from tools import coordinated_rotation_initializer as initializer


@pytest.mark.parametrize("shape",[(7,9),(16,16),(512,512)])
@pytest.mark.parametrize("k",range(4))
def test_inverse_matches_literal_rot90_pixel_indices_and_center_units(shape,k):
    h,w=shape
    original=torch.arange(h*w).reshape(h,w)
    rotated=torch.rot90(original,k,(-2,-1))
    xy=np.array([[0,0],[rotated.shape[1]-1,rotated.shape[0]-1],
        [rotated.shape[1]//2,rotated.shape[0]//2]],dtype=np.float64)
    recovered=initializer.unrotate_points(xy,k,width=w,height=h)
    for (xr,yr),(x,y) in zip(xy,recovered,strict=True):
        assert rotated[int(yr),int(xr)]==original[int(y),int(x)]
    if h==w:
        unit=(xy+.5)/w;u,v=unit.T
        expected=unit if k==0 else np.column_stack((1-v,u)) if k==1 else 1-unit if k==2 else np.column_stack((v,1-u))
        np.testing.assert_allclose((recovered+.5)/w,expected,rtol=0,atol=1e-16)


def candidate(k,n,bins,rmse,status="ok"):
    return dict(quarter_turns=k,ransac_inliers=n,occupied_quadrants_4x4=bins,inlier_fit_rmse_512px=rmse,status=status)


def test_exact_lexicographic_support_fit_and_k_ranking():
    c=initializer.select_candidate
    assert c([candidate(0,8,16,0.),candidate(1,9,1,2.)])["quarter_turns"]==1
    assert c([candidate(0,9,1,0.),candidate(1,9,2,2.)])["quarter_turns"]==1
    assert c([candidate(0,9,2,2.),candidate(1,9,2,1.)])["quarter_turns"]==1
    assert c([candidate(3,9,2,1.),candidate(0,9,2,1.)])["quarter_turns"]==0
    assert c([candidate(0,100,16,0.,"failed")]) is None


def paths(tmp_path,side=16):
    raster=np.arange(side*side,dtype=np.uint8).reshape(side,side)
    for role in ("fixed","moving"):Image.fromarray(raster).save(tmp_path/(role+".png"))
    return [("fixture",tmp_path/"fixed.png",tmp_path/"moving.png",tmp_path/"affine.npz")]


def factory_fixture(events,counts=(5,9,8,4)):
    def factory(config,device):
        events.append("model")
        assert config==dict(superpoint=dict(nms_radius=4,keypoint_threshold=.005,max_keypoints=3000),
            superglue=dict(weights="outdoor",sinkhorn_iterations=30,match_threshold=.3))
        fixed_seen=None;moving_seen=None;k=0
        def model(images):
            nonlocal k,fixed_seen,moving_seen
            events.append(k)
            if k==0:fixed_seen=images["image0"].clone();moving_seen=images["image1"].clone()
            assert torch.equal(images["image0"],fixed_seen)
            assert torch.equal(images["image1"],torch.rot90(moving_seen,k,(-2,-1)))
            p0=np.array([[x,y] for x in (1.,6.,12.) for y in (1.,6.,12.)])
            # Original moving pixel target is a true positive90-degree similarity.
            p1=p0@np.array([[0.,-.8],[.8,0.]]).T+np.array([14.,1.])
            x,y=p1.T;side=images["image0"].shape[-1]
            p1rot=p1 if k==0 else np.column_stack((y,side-1-x)) if k==1 else side-1-p1 if k==2 else np.column_stack((side-1-y,x))
            matches=np.arange(9,dtype=np.int64);matches[counts[k]:]=-1
            k+=1
            return dict(keypoints0=torch.tensor(p0)[None],keypoints1=torch.tensor(p1rot)[None],matches0=torch.tensor(matches)[None])
        return model
    return factory


def test_four_calls_one_model_original_frame_fit_and_standard_provenance(tmp_path):
    events=[];pair=paths(tmp_path)
    report=initializer.extract(pair,tmp_path/"report.json",device_name="cpu",
        matcher_factory=factory_fixture(events),production=False)
    assert events==["model",0,1,2,3]
    row=report["rows"][0]
    assert row["status"]=="ok" and row["selected_quarter_turns"]==1
    assert len(row["candidates"])==4 and row["matcher_calls"]==4
    assert [c["raw_matches"] for c in row["candidates"]]==[5,9,8,4]
    expected=np.array([[0.,-.8],[.8,0.]])
    # Pixel fit t=M*x+[14,1]; center-normalized offset is (b+.5-M*.5)/16.
    offset=(np.array([14.,1.])+.5-expected@np.array([.5,.5]))/16
    with np.load(pair[0][3]) as archive:
        assert str(archive["affine_source"])=="direct_image_superglue"
        assert str(archive["initializer"])=="quarter_turns"
        assert archive["post_affine_matrix"].dtype==archive["post_affine_offset"].dtype==np.float32
        np.testing.assert_allclose(archive["post_affine_matrix"],expected,atol=2e-7,rtol=0)
        np.testing.assert_allclose(archive["post_affine_offset"],offset,atol=2e-7,rtol=0)
        assert np.linalg.det(archive["post_affine_matrix"].astype(np.float64))>0
    assert report["landmarks_initial_DHR_affine_full_DHR_field_loaded"] is False
    assert "no downstream objective" in report["selection"]
    assert json.loads((tmp_path/"report.json").read_text())["rows"][0]["selected_quarter_turns"]==1


def test_no_eligible_candidate_no_identity_fallback_or_affine_export(tmp_path):
    events=[];pair=paths(tmp_path)
    row=initializer.extract(pair,tmp_path/"report.json",device_name="cpu",production=False,
        matcher_factory=factory_fixture(events,(3,4,5,6)))["rows"][0]
    assert events==["model",0,1,2,3] and row["status"]=="no_eligible_candidate"
    assert row["selected_quarter_turns"] is None and len(row["candidates"])==4
    assert not pair[0][3].exists()


@pytest.mark.parametrize("kind",["negative","nonfinite","degenerate"])
def test_invalid_similarity_candidates_retained_and_never_exported(tmp_path,monkeypatch,kind):
    pair=paths(tmp_path);events=[]
    if kind=="negative":monkeypatch.setattr(initializer,"similarity_from_inlier_units",lambda *a:(np.diag([1.,-1.]),np.zeros(2)))
    elif kind=="nonfinite":monkeypatch.setattr(initializer,"similarity_from_inlier_units",lambda *a:(np.eye(2),np.array([np.inf,0.])))
    else:
        def bad(*a):raise ValueError("degenerate source point set")
        monkeypatch.setattr(initializer,"similarity_from_inlier_units",bad)
    row=initializer.extract(pair,tmp_path/"report.json",device_name="cpu",production=False,
        matcher_factory=factory_fixture(events,(9,9,9,9)))["rows"][0]
    assert row["status"]=="no_eligible_candidate" and not pair[0][3].exists()
    assert all(c["status"]=="failed" and "error" in c for c in row["candidates"])


def test_production_square512_and_no_threshold_sweep_or_overwrite(tmp_path):
    pair=paths(tmp_path)
    factory=lambda *a:pytest.fail("model must not start")
    with pytest.raises(ValueError,match="square512"):
        initializer.extract(pair,tmp_path/"report.json",device_name="cpu",matcher_factory=factory)
    with pytest.raises(ValueError,match="threshold"):
        initializer.extract(pair,tmp_path/"report.json",device_name="cpu",match_threshold=.2,matcher_factory=factory)
    with pytest.raises(ValueError,match="new output"):
        initializer.extract(pair,pair[0][3],device_name="cpu",matcher_factory=factory)
    pair[0][3].write_text("keep")
    with pytest.raises(ValueError):initializer.extract(pair,tmp_path/"report.json",device_name="cpu",matcher_factory=factory)
    assert pair[0][3].read_text()=="keep"


@pytest.mark.parametrize("k",[-1,4,True,1.5])
def test_quarter_turn_type_guard(k):
    with pytest.raises(ValueError):initializer.unrotate_points(np.zeros((1,2)),k,width=512,height=512)
