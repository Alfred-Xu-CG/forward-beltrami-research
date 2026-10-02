import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_coupled_pilot as pilot
from test_coordinated_stiffness_pilot import source
from test_coordinated_miit_transfer import optimizer


def test_configuration_only_adds_one_seed_to_original300(tmp_path):
    path=source(tmp_path);row=json.loads(path.read_text())['rows'][0]
    old=json.loads((path.parent/row['methods']['analytic']['report']).read_text());untouched=copy.deepcopy(old)
    cfg=pilot.configuration(old,tmp_path/'new.npz')
    assert old==untouched and cfg.seed_initializer=='coupled_mind'
    assert cfg.inner_steps==30 and cfg.inner_steps_by_level==[30]*5
    values={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}
    for key in ('output','inner_steps_by_level','mind_frame','image_objective','seed_initializer'):values.pop(key,None)
    expected={k:v for k,v in old['configuration'].items() if k not in ('output','inner_steps_by_level','mind_frame','image_objective','seed_initializer')}
    assert values==expected
    old['configuration']['seed_initializer']='coupled_mind'
    with pytest.raises(ValueError):pilot.configuration(old,tmp_path/'invalid.npz')


def test_exact_three_image_only_calls_failures_retained_and_references_unchanged(tmp_path,monkeypatch):
    path=source(tmp_path);calls=[]
    def mock(cfg):
        calls.append(cfg)
        if len(calls)==1:raise RuntimeError('deliberate constructor failure')
        result=optimizer(cfg)
        result.update(seed_initializer='coupled_mind',seed_objective_evaluations=1)
        return result
    monkeypatch.setattr(pilot,'optimize',mock)
    original_open=Path.open
    def guarded(path,*args,**kwargs):
        if path.suffix=='.csv':pytest.fail('correspondence labels must remain unopened')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    report=pilot.run(path,tmp_path/'out')
    assert len(calls)==3 and all(c.seed_initializer=='coupled_mind' and c.inner_steps==30 for c in calls)
    assert report['prediction_complete'] and report['annotations_read'] is False
    assert report['rows'][0]['methods']['analytic']['status']=='failed'
    assert all(row['methods']['analytic']['gradient_steps']==300 for row in report['rows'][1:])
    for row in report['rows']:
        assert 'NOT rerun' in row['methods']['f2']['pilot_reference']
        assert (tmp_path/'out'/row['methods']['f2']['output']).resolve().exists()
    with pytest.raises(FileExistsError):pilot.run(path,tmp_path/'out')
