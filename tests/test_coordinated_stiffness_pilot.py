import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_stiffness_pilot as pilot
from test_coordinated_miit_transfer import assets, invoke, optimizer


def source(tmp_path):
    args=assets(tmp_path); invoke(args)
    manifest=json.loads((args.output/'predictions.json').read_text())
    # Reuse tiny prepared assets, but exercise the strict production config
    # through a mocked optimizer; do not relax the production pilot guard.
    for row in manifest['rows']:
        path=args.output/row['methods']['analytic']['report']
        value=json.loads(path.read_text())
        value['configuration'].update(grid_side=257,image_side=512,levels=[17,33,65,129,257],
            image_levels=[32,64,128,256,512],inner_steps=30)
        path.write_text(json.dumps(value))
    return args.output/'predictions.json'


def test_configuration_changes_only_declared_budget_and_frame(tmp_path):
    path=source(tmp_path); rows=json.loads(path.read_text())['rows']
    old=json.loads((path.parent/rows[0]['methods']['analytic']['report']).read_text())
    untouched=copy.deepcopy(old)
    cfg=pilot.configuration(old,tmp_path/'new.npz','adam900')
    assert old==untouched and cfg.inner_steps==90 and cfg.inner_steps_by_level==[90]*5
    assert cfg.mind_frame=='shared_affine' and cfg.image_objective=='continuation'
    values={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}
    for key in ('output','inner_steps','inner_steps_by_level','mind_frame','image_objective'):
        values.pop(key,None)
    expected={k:v for k,v in old['configuration'].items()
              if k not in ('output','inner_steps','inner_steps_by_level','mind_frame','image_objective')}
    assert values==expected
    cfg=pilot.configuration(old,tmp_path/'stiff.npz','stiffness300')
    assert cfg.inner_steps==30 and cfg.inner_steps_by_level==[30]*5
    old['configuration']['method']='f2'
    with pytest.raises(ValueError): pilot.configuration(old,tmp_path/'bad.npz','adam900')


def test_three_adam_attempts_only_archived_f2_and_failures_retained(tmp_path,monkeypatch):
    path=source(tmp_path); calls=[]
    def mock(cfg):
        calls.append(cfg)
        if len(calls)==1: raise RuntimeError('deliberate failure')
        result=optimizer(cfg); result['mind_frame']='shared_affine'
        return result
    monkeypatch.setattr(pilot,'optimize',mock)
    report=pilot.run(path,tmp_path/'out','adam900')
    assert len(calls)==3 and all(c.method=='analytic' and c.inner_steps==90 for c in calls)
    assert report['prediction_complete'] and report['annotations_read'] is False
    assert report['rows'][0]['methods']['analytic']['status']=='failed'
    assert all(row['methods']['analytic']['gradient_steps']==900 for row in report['rows'][1:])
    for row in report['rows']:
        assert 'NOT rerun' in row['methods']['f2']['pilot_reference']
        assert (tmp_path/'out'/row['methods']['f2']['output']).resolve().exists()
    with pytest.raises(FileExistsError): pilot.run(path,tmp_path/'out','adam900')
