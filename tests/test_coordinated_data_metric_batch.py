"""Timed suffix bookkeeping and shared-prefix pairing; no image optimization."""
import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_data_metric_batch as batch
from tests.test_coordinated_match_fusion import fixtures,original


def timed_report(arm='adam'):
    stages=[dict(elapsed_seconds=2.01,time_overrun_seconds=.01,budget_elapsed=True,stop_reason='time_budget',counts={'gradient_steps':6}),
        dict(elapsed_seconds=.2,time_overrun_seconds=0.,budget_elapsed=False,stop_reason='line_search_exhausted',counts={'gradient_steps':6})]
    return dict(prefix_gradient_steps=240,suffix_gradient_steps=12,gradient_steps=252,failed_trials=1,
        terminal_budget=dict(protocol='timed_final',arm=arm,seconds_per_axis=2.,stage_records=stages),stages=stages)


def test_timed_accounting_is_not_a_false300_or_convergence_claim():
    report=timed_report();batch.validate_timed_budget(report,arm='adam')
    for field,value in [('gradient_steps',300),('prefix_gradient_steps',239),('suffix_gradient_steps',True)]:
        with pytest.raises(ValueError):batch.validate_timed_budget({**report,field:value})
    broken=copy.deepcopy(report);broken['stages'][0]['time_overrun_seconds']=0.
    with pytest.raises(ValueError):batch.validate_timed_budget(broken)
    with pytest.raises(ValueError):batch.validate_timed_budget(report,arm='data_metric')


def test_same_frozen_evidence_configuration(tmp_path):
    source=original(tmp_path);source['match_weight']=.2
    cfg=batch.configuration(source,tmp_path/'new.npz')
    assert cfg.image_side==512 and cfg.grid_side==257 and cfg.inner_steps_by_level==[30]*5
    assert cfg.match_weight==.2 and cfg.match_robust_scale==8 and cfg.matches==Path(source['matches'])


def test_shared_prefix_alternating_order_and_failures_before_all50_scoring(tmp_path):
    cases,_=fixtures(tmp_path)
    for case in cases:case['original_configuration']['match_weight']=.2
    output=tmp_path/'run';prefixes=[];calls=[]
    def extract(cfg):
        name=cfg.output.name.removesuffix('_prefix_best.npz');prefixes.append(name)
        if name==cases[0]['name']:raise ValueError('explicit prefix failure')
        return dict(prefix_best=cfg.output,prefix_start=cfg.output.with_name(name+'_suffix_start.npz'),
            prefix_report=cfg.output.with_suffix('.json'))
    def optimize(cfg,*,prefix_best,prefix_start,arm,seconds_per_axis):
        for item in batch.ARMS:assert not (output/item/'miit_predictions.json').exists()
        name=cfg.output.name.removesuffix('_analytic.npz')
        assert prefix_best.name==name+'_prefix_best.npz' and prefix_start.name==name+'_suffix_start.npz'
        assert seconds_per_axis==2.
        calls.append((name,arm,str(prefix_start)))
        if name==cases[-1]['name'] and arm=='data_metric':raise ValueError('explicit last suffix failure')
        return timed_report(arm)
    result=batch.run('unused',output,source_loader=lambda _:copy.deepcopy(cases),prefix_extractor=extract,
        suffix_optimizer=optimize,prefix_validator=lambda *_:None,output_validator=lambda *_:.2)
    assert len(prefixes)==25 and len(calls)==48 and result['prediction_complete']
    assert result['arms']==dict(adam=dict(successful=24,failed=1),data_metric=dict(successful=23,failed=2))
    for i,case in enumerate(cases[1:],1):
        pair=[r for r in calls if r[0]==case['name']]
        assert [r[1] for r in pair]==list(reversed(batch.ARMS) if i%2 else batch.ARMS)
        assert pair[0][2]==pair[1][2]
    for arm in batch.ARMS:
        manifest=json.loads((output/arm/'existing_predictions.json').read_text())
        assert manifest['all50_terminal'] and manifest['budget_mode']=='timed_final'
