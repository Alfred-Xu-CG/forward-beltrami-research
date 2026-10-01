import numpy as np
import pytest

from tools.coordinated_image_matches import raw_correspondences


def test_raw_matches_preserve_nonrigid_motion_and_half_pixel_frame():
    source=np.array([[0.,0.],[1.,3.],[6.,5.]])
    target=np.array([[1.,7.],[5.,2.]])
    report=raw_correspondences(source,target,np.array([1,-1,0]),np.array([.9,.1,.7]),height=8,width=8)
    np.testing.assert_allclose(report["source_points_unit"],(source[[0,2]]+.5)/8)
    np.testing.assert_allclose(report["target_points_unit"],(target[[1,0]]+.5)/8)
    assert report["confidence"]==[.9,.7] and report["raw_matches"]==2
    assert not report["global_geometric_ransac_used"]


@pytest.mark.parametrize("matches,scores",[(np.array([2]),np.array([.7])),
    (np.array([-2]),np.array([.7])),(np.array([.5]),np.array([.7])),
    (np.array([0]),np.array([np.nan])),(np.array([0]),np.array([1.2]))])
def test_invalid_raw_match_inputs_rejected(matches,scores):
    with pytest.raises(ValueError):
        raw_correspondences(np.zeros((1,2)),np.zeros((1,2)),matches,scores,height=8,width=8)


def test_no_matches_is_reported_not_fabricated():
    report=raw_correspondences(np.zeros((2,2)),np.empty((0,2)),np.array([-1,-1]),np.zeros(2),height=8,width=8)
    assert report["raw_matches"]==0 and report["source_points_unit"]==[]
    assert report["status"]=="insufficient_matches"
