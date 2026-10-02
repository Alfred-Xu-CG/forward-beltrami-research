import pytest
import torch
from tools.coordinated_mind_frame_probe import channel_permutation,probe,run


@pytest.mark.parametrize("k,expected",[(0,[0,1,2,3,4,5,6,7]),
    (1,[2,3,1,0,6,4,7,5]),(2,[1,0,3,2,7,6,5,4]),(3,[3,2,0,1,5,7,4,6])])
def test_literal_quarter_turn_offset_indices(k,expected):
    assert channel_permutation(k)==expected


def test_exact_covariance_and_directional_noninvariance():
    g=torch.Generator().manual_seed(33)
    image=torch.rand((1,1,23,23),generator=g,dtype=torch.float64)
    rows=probe(image)
    assert rows[0]['unpermuted']['maximum_absolute']==0
    for row in rows[1:]:
        assert row['permuted']['maximum_absolute']<3e-15
        assert row['unpermuted']['mean_absolute']>.01


def test_artifact_is_diagnostic_not_optimizer_and_preserves_existing(tmp_path):
    out=tmp_path/'probe.json';result=run(out)
    assert result['annotations_read'] is False and result['optimizer_executed'] is False
    assert len(result['rows'][0]['results'])==4
    with pytest.raises(FileExistsError):run(out)


@pytest.mark.parametrize('bad',[True,-1,4,1.0])
def test_bad_turns_rejected(bad):
    with pytest.raises(ValueError):channel_permutation(bad)
