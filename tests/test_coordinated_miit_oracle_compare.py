import argparse
import pytest
from tools import coordinated_miit_oracle_compare as module


def arguments(tmp_path):
    return argparse.Namespace(output=tmp_path/'comparison',predictions=tmp_path/'original.json',
        source_data=tmp_path/'labels',device='cpu',threads=2)


def result(method='analytic'):
    return dict(label_oracle=True,landmarks_used=True,budget_complete=True,
        counts=dict(gradient_steps=300,decoder_trials=310),saved_binary_certificate=dict(valid=True),
        configuration=dict(method=method),capacity_witness_success=True,fit_seconds=.1,end_to_end_seconds=.2,peak_allocated_bytes=None,
        final_geometry=dict(valid=True),end_errors=dict(canvas_pixels=dict(mean=.1,p90=.2,maximum=.3,per_label={})),
        end_original_e1=dict(total=.5))


def test_fixed_warmup_abba_same_source_and_fresh_outputs(tmp_path,monkeypatch):
    args=arguments(tmp_path);calls=[]
    def fake(config):calls.append(config);return result(config.method)
    monkeypatch.setattr(module.witness,'run',fake)
    r=module.run(args)
    assert [c.method for c in calls]==['analytic','f2','analytic','f2','f2','analytic']
    assert all(c.predictions==args.predictions and c.source_data==args.source_data for c in calls)
    assert len({c.output for c in calls})==6
    assert r['comparison_complete'] and r['landmarks_used']
    assert all(row['status']=='ok' for row in r['rows'])
    with pytest.raises(FileExistsError):module.run(args)


def test_one_failure_retained_no_replacement_or_later_omission(tmp_path,monkeypatch):
    calls=[]
    def fake(config):
        calls.append(config)
        if len(calls)==3:raise RuntimeError('declared failed attempt')
        return result(config.method)
    monkeypatch.setattr(module.witness,'run',fake)
    r=module.run(arguments(tmp_path))
    assert len(calls)==6 and r['comparison_complete']
    assert r['rows'][2]['status']=='failed'
    assert all(row['status']=='ok' for row in r['rows'][3:])
