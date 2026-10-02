import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_miit_frame_pilot as pilot
from test_coordinated_miit_transfer import assets,invoke,optimizer


def source(tmp_path):
    args=assets(tmp_path);invoke(args);return args.output/'predictions.json'


def test_only_frame_and_output_change(tmp_path):
    path=source(tmp_path);manifest=json.loads(path.read_text())
    original=json.loads((path.parent/manifest['rows'][0]['methods']['analytic']['report']).read_text())
    untouched=copy.deepcopy(original)
    new=pilot.configuration(original,tmp_path/'new.npz')
    assert original==untouched
    normalized={key:str(value) if isinstance(value,Path) else value for key,value in vars(new).items()}
    assert normalized.pop('mind_frame')=='shared_affine'
    normalized['output']=original['configuration']['output']
    assert normalized==original['configuration']
    original['configuration']['mind_frame']='shared_affine'
    with pytest.raises(ValueError):pilot.configuration(original,tmp_path/'bad.npz')


def test_six_attempts_share_original_inputs_and_reuse_native_reference(tmp_path,monkeypatch):
    path=source(tmp_path);calls=[]
    def mock(config):
        calls.append(config)
        r=optimizer(config);r['mind_frame']='shared_affine';return r
    monkeypatch.setattr(pilot.application,'optimize',mock)
    output=tmp_path/'pilot';r=pilot.run(path,output)
    assert len(calls)==6 and r['prediction_complete'] and r['annotations_read'] is False
    assert r['label_informed_development'] is True
    assert all(c.mind_frame=='shared_affine' for c in calls)
    assert all(row['status']=='ok' for row in r['rows'])
    for row in r['rows']:
        assert (output/row['fixed']).resolve().parent==path.parent
        assert 'NOT rerun' in row['methods']['dhr']['pilot_reference']
        assert (output/row['methods']['dhr']['field']).resolve().exists()
    with pytest.raises(FileExistsError):pilot.run(path,output)


def test_failure_retained_and_other_attempts_not_discarded(tmp_path,monkeypatch):
    path=source(tmp_path);calls=[]
    def mock(config):
        calls.append(config)
        if len(calls)==1:raise RuntimeError('deliberate failure')
        r=optimizer(config);r['mind_frame']='shared_affine';return r
    monkeypatch.setattr(pilot.application,'optimize',mock)
    r=pilot.run(path,tmp_path/'pilot')
    assert len(calls)==6 and r['prediction_complete']
    assert r['rows'][0]['status']=='partial_failure'
    assert r['rows'][0]['methods']['analytic']['status']=='failed'
    assert all(row['status']=='ok' for row in r['rows'][1:])


@pytest.mark.parametrize('corruption',['nonfinite_diagnostic','zero_geometry','wrong_f2_reserve','wrong_dispatch'])
def test_corruption_retained_without_aborting_other_attempts(tmp_path,monkeypatch,corruption):
    import numpy as np
    path=source(tmp_path);calls=[]
    def mock(config):
        calls.append(config);r=optimizer(config);r['mind_frame']='shared_affine'
        target=2 if corruption=='wrong_f2_reserve' else 1
        if len(calls)==target:
            if corruption=='nonfinite_diagnostic':r['final']['total']=float('nan')
            elif corruption=='wrong_f2_reserve':r['configuration']['f2_floor_safety_fraction']=1.
            elif corruption=='wrong_dispatch':r['mind_frame']='original'
            else:
                with np.load(config.output) as z:data={k:z[k] for k in z.files}
                data['vertices']=np.zeros_like(data['vertices']);np.savez(config.output,**data)
        return r
    monkeypatch.setattr(pilot.application,'optimize',mock)
    r=pilot.run(path,tmp_path/'pilot')
    assert len(calls)==6 and r['prediction_complete']
    method='f2' if corruption=='wrong_f2_reserve' else 'analytic'
    assert r['rows'][0]['methods'][method]['status']=='failed'
    assert all(row['status']=='ok' for row in r['rows'][1:])
    if corruption=='nonfinite_diagnostic':assert r['rows'][0]['methods'][method]['final']['total'] is None
