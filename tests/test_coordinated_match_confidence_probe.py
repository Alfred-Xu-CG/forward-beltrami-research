import numpy as np
import pytest
from tools.coordinated_match_confidence_probe import footprint, compare_targets,unbatch_vertices,cohort_summary,summarize


def unit(x,y):return (np.array([x,y])*8+.5)/512


def test_saved_batch_is_removed_before_literal_p1():
    yy,xx=np.meshgrid(np.arange(257)/256,np.arange(257)/256,indexing='ij')
    vertices=np.stack((xx,yy),-1)[None]
    unbatched=unbatch_vertices(vertices)
    assert unbatched.shape==(257,257,2) and np.shares_memory(unbatched,vertices)
    np.testing.assert_array_equal(unbatched[17,29],[29/256,17/256])
    with pytest.raises(ValueError):unbatch_vertices(vertices[0])
    with pytest.raises(ValueError):unbatch_vertices(vertices.astype(np.float32))


def test_exact_centers_and_closed_hull():
    ids,w=footprint(unit(63,63));assert ids.tolist()==[4030,4031,4094,4095]
    np.testing.assert_array_equal(w,[0,0,0,1])
    assert footprint(unit(-1e-8,3)) is None
    assert footprint(unit(63+1e-8,3)) is None
    with pytest.raises(ValueError):footprint([float('nan'),0])


def test_bilinear_rows_targets_strict_ranks_and_maximum():
    # Coarse side64 but only distinct first2 source rows: no identity target oracle.
    C=np.zeros((4096,4096),np.float32);C[0,0]=.8;C[1,0]=.4;C[:,1]=.2
    result=compare_targets(C,unit(.5,0),unit(.5,0),unit(1,0))
    assert result['supported']
    assert result['bilinear']['true_score']==pytest.approx(.4)
    assert result['bilinear']['predicted_score']==pytest.approx(.2)
    assert result['bilinear']['true_rank']==2 and result['bilinear']['predicted_rank']==2
    assert result['footprint_max']['true_rank']==1
    assert result['bilinear']['ordering']=='true_above_prediction'


def test_unsupported_retained_not_clamped_and_ties():
    C=np.zeros((4096,4096),np.float32)
    result=compare_targets(C,unit(0,0),unit(64,0),unit(1,0))
    assert not result['supported'] and result['support']==dict(source=True,true_target=False,prediction=True)
    assert result['bilinear'] is None
    tie=compare_targets(C,unit(0,0),unit(2,0),unit(1,0))
    assert tie['bilinear']['ordering']=='tie' and tie['bilinear']['true_zero']
    assert tie['bilinear']['true_rank']==tie['bilinear']['predicted_rank']==1


def test_missing_and_zero_support_keep_cohort_denominators():
    record=dict(supported=False,support=dict(source=False,true_target=True,prediction=True),bilinear=None,footprint_max=None)
    rows=[dict(name='miit_2_to_3',summary=summarize([record]))]
    summary=cohort_summary(rows,{'miit_2_to_3':1,'miit_7_to_8':3})['miit']
    assert summary['case_denominator']==2 and summary['computed_cases']==1
    assert summary['denominator']==4 and summary['computed_case_ID_count']==1
    assert summary['bilinear_mean_case_rank_advantage'] is None
