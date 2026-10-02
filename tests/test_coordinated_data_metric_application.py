import copy
import json
from pathlib import Path
import numpy as np
import pytest
from tests.test_coordinated_shared_affine_evidence import configuration
from tests.test_coordinated_data_metric import Tick
from tools import coordinated_data_metric_application as app
from tools import coordinated_real_case as original


def test_common_raw_eighth_stage_and_two_timed_complete_call_counts(tmp_path,monkeypatch):
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    args.levels=[3,3,5,5,args.grid_side];args.image_levels=[8,8,8,8,args.image_side]
    full=[];baseline=original.optimize(args,lambda y,s,t:full.append(y.clone()))
    ordinary_path=args.output
    monkeypatch.setattr(app,'_validate_configuration',lambda cfg:None)
    args.output=tmp_path/'common_prefix_best.npz'
    prefix=app.extract_prefix(args)
    assert prefix['gradient_steps']==8 and prefix['accepted_stages']==8
    with np.load(prefix['prefix_start']) as saved:np.testing.assert_array_equal(saved['vertices'],full[7].numpy())
    prefix_report=json.loads(Path(prefix['prefix_report']).read_text())
    assert prefix_report['stages']==json.loads(json.dumps(baseline['stages'][:8]))
    assert prefix_report['final']['total']==min([baseline['initial']['total']]+[s['accepted_full_total'] for s in baseline['stages'][:8]])
    previous=original.Evidence.__call__;calls=[]
    def counted(self,Y):calls.append(self.fixed.shape[-1]);return previous(self,Y)
    monkeypatch.setattr(original.Evidence,'__call__',counted)
    for name in ('optimize_data_metric_fiber','optimize_timed_adam_fiber'):
        fn=getattr(app,name)
        monkeypatch.setattr(app,name,lambda *a,_fn=fn,**kw:_fn(*a,clock=Tick(),**kw))
    reports=[]
    for arm in ('adam','data_metric'):
        calls.clear();args.output=tmp_path/(arm+'.npz')
        report=app.optimize_timed_final(args,prefix_best=prefix['prefix_best'],prefix_start=prefix['prefix_start'],arm=arm)
        reports.append(report)
        assert report['prefix_gradient_steps']==8
        assert report['gradient_steps']==8+report['suffix_gradient_steps']
        assert report['suffix_gradient_steps']==sum(s['counts']['gradient_steps'] for s in report['stages'])
        assert report['suffix_objective_evaluations']==len(calls)
        assert report['terminal_budget']['stage_records']==report['stages'] and len(report['stages'])==2
        assert report['saved_binary_certificate']['valid']
        assert report['final']['total']==min([report['prefix_best_total']]+[s['accepted_full_total'] for s in report['stages']])
        assert report['landmarks_used'] is False
        assert report['query_count']==args.image_side**2 and report['control_vertices']==args.grid_side**2
        assert all(s['elapsed_seconds']>0 and s['time_overrun_seconds']>=0 for s in report['stages'])
    assert reports[0]['initial']==reports[1]['initial']
    assert reports[0]['prefix_best_total']==reports[1]['prefix_best_total']
    with pytest.raises(ValueError,match='changed prefix configuration'):
        bad=copy.deepcopy(args);bad.output=tmp_path/'wrong_config.npz';bad.shape_weight=.1
        app.optimize_timed_final(bad,prefix_best=prefix['prefix_best'],prefix_start=prefix['prefix_start'],arm='adam')


def test_production_frozen_configuration_rejects_changes(tmp_path):
    source=Path('outputs/coordinated_instance_registration/match_fusion_all50_t23/fusion/predictions.json')
    if not source.exists():pytest.skip('archived development config unavailable')
    import argparse
    values=json.loads(source.read_text())['rows'][0]['configuration']
    args=argparse.Namespace(**values);app._validate_configuration(args)
    for key,value in [('learning_rate',.1),('match_weight',.1),('image_side',1024),('geometry_backend','existing')]:
        bad=copy.deepcopy(args);setattr(bad,key,value)
        with pytest.raises(ValueError):app._validate_configuration(bad)
