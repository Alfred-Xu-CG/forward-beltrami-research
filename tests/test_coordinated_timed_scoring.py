"""Timed-only accounting checks and unchanged legacy scoring routes."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from tests.test_coordinated_data_metric_batch import timed_report
from tests.test_coordinated_dhr_existing_score import _complete_fixture
from tests.test_coordinated_miit_score import manifest as miit_manifest
from tools import coordinated_existing_shared_frame_score as existing
from tools import coordinated_miit_score as miit


def _map(path):
    y,x=np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3),indexing='ij')
    identity=np.stack((x,y),-1)[None]
    np.savez(path,vertices=identity,boundary_reference=identity,post_affine_matrix=np.eye(2),
        post_affine_offset=np.zeros(2),interpolation=np.asarray('p1_ac'))


def _existing(tmp_path):
    path,root,value=_complete_fixture(tmp_path)
    _map(tmp_path/'map.npz')
    report=timed_report('adam');report.update(configuration=dict(interpolation='p1_ac'),landmarks_used=False)
    (tmp_path/'timed.json').write_text(json.dumps(report))
    value.update(cohort_size=22,budget_mode='timed_final',all50_terminal=True,seconds_per_axis=2.,timed_arm='adam')
    for row in value['rows']:
        row.update(output='map.npz',report='timed.json',configuration=dict(mind_frame='shared_affine'),
            **{key:copy.deepcopy(report[key]) for key in ('gradient_steps','prefix_gradient_steps',
                'suffix_gradient_steps','terminal_budget','failed_trials')})
    path.write_text(json.dumps(value))
    return path,root,value,report


def test_existing_timed_variable_budget_and_early_failures_keep_legal_maps(tmp_path):
    path,root,value,report=_existing(tmp_path)
    result=existing.score(path,root,budget_mode='timed_final')
    assert report['gradient_steps']==252 and report['failed_trials']==1
    assert result['scored_pairs']==22 and result['expected_gradient_steps'] is None
    assert result['budget_mode']=='timed_final' and result['equal_specimen_canvas']['mean_pair_mean']<1e-12
    assert [row['scored_landmarks'] for row in result['rows']]==[80]*20+[77,69]
    # Unchanged legacy default does not silently accept a252-gradient control.
    legacy=existing.score(path,root)
    assert legacy['scored_pairs']==0 and legacy['expected_gradient_steps']==300
    # Explicit old fixed-gradient override remains available, without timing metadata.
    value.pop('budget_mode');value.pop('all50_terminal');value.pop('seconds_per_axis');value.pop('timed_arm')
    for row in value['rows']:
        row['gradient_steps']=250;row['failed_trials']=0
        for key in ('report','prefix_gradient_steps','suffix_gradient_steps','terminal_budget'):row.pop(key)
    path.write_text(json.dumps(value))
    legacy=existing.score(path,root,expected_gradient_steps=250)
    assert legacy['scored_pairs']==22 and legacy['expected_gradient_steps']==250


def test_existing_timed_report_manifest_mismatch_retained_as_failure(tmp_path):
    path,root,value,report=_existing(tmp_path)
    value['rows'][0]['gradient_steps']+=1;path.write_text(json.dumps(value))
    result=existing.score(path,root,budget_mode='timed_final')
    assert result['scored_pairs']==21 and result['rows'][0]['status']=='failed'
    assert 'accounting mismatch' in result['rows'][0]['error']
    assert result['lung_all20']['all_directions'] is None


@pytest.mark.parametrize('field,value',[('all50_terminal',False),('seconds_per_axis',1.),('budget_mode','fixed_gradients')])
def test_existing_timed_requires_complete_paired_protocol_before_labels(tmp_path,monkeypatch,field,value):
    path,root,manifest,report=_existing(tmp_path)
    manifest[field]=value;path.write_text(json.dumps(manifest))
    monkeypatch.setattr(existing,'_case_data',lambda *a:pytest.fail('labels read before timing protocol check'))
    with pytest.raises(ValueError,match='paired timed-final'):
        existing.score(path,root,budget_mode='timed_final')


def test_miit_timed_reader_variable_budget_match_and_legacy_same_map(tmp_path):
    _map(tmp_path/'map.npz')
    report=timed_report('data_metric');report.update(configuration=dict(interpolation='p1_ac'),landmarks_used=False)
    path=tmp_path/'map.json';path.write_text(json.dumps(report))
    record=dict(output='map.npz',report='map.json',arm='data_metric',
        **{key:copy.deepcopy(report[key]) for key in ('gradient_steps','prefix_gradient_steps',
            'suffix_gradient_steps','terminal_budget','failed_trials')})
    vertices,metadata=miit._analytic_map(record,tmp_path,np.eye(2),np.zeros(2))
    assert metadata['gradient_steps']==252 and metadata['failed_trials']==1
    altered=copy.deepcopy(record);altered['suffix_gradient_steps']+=1
    with pytest.raises(ValueError,match='accounting mismatch'):
        miit._analytic_map(altered,tmp_path,np.eye(2),np.zeros(2))
    altered=copy.deepcopy(record);altered['arm']='adam'
    with pytest.raises(ValueError,match='accounting'):
        miit._analytic_map(altered,tmp_path,np.eye(2),np.zeros(2))
    legacy=dict(output='map.npz',report='map.json')
    old,oldmeta=miit._safe_map(legacy,tmp_path,np.eye(2),np.zeros(2))
    actual,actualmeta=miit._analytic_map(legacy,tmp_path,np.eye(2),np.zeros(2))
    np.testing.assert_array_equal(actual,old);np.testing.assert_array_equal(vertices,old)
    assert actualmeta==oldmeta


@pytest.mark.parametrize('change',['not_all50','missing_budget','wrong_arm'])
def test_miit_declared_timed_manifest_never_falls_through_legacy(tmp_path,change):
    value=miit_manifest();value.update(budget_mode='timed_final',all50_terminal=True,seconds_per_axis=2.,timed_arm='adam')
    for row in value['rows']:
        row['methods']['analytic'].update(status='ok',arm='adam',terminal_budget=timed_report('adam')['terminal_budget'])
    if change=='not_all50':value['all50_terminal']=False
    elif change=='missing_budget':value['rows'][0]['methods']['analytic'].pop('terminal_budget')
    else:value['rows'][0]['methods']['analytic']['arm']='data_metric'
    path=tmp_path/'predictions.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):miit.load_manifest(path)


def test_miit_complete_timed_and_legacy_manifests_both_supported(tmp_path):
    value=miit_manifest();path=tmp_path/'predictions.json'
    path.write_text(json.dumps(value));legacy,_=miit.load_manifest(path)
    assert legacy==value
    value.update(budget_mode='timed_final',all50_terminal=True,seconds_per_axis=2.,timed_arm='data_metric')
    # Failed attempts need no fabricated timed report; denominators are retained.
    path.write_text(json.dumps(value));timed,_=miit.load_manifest(path)
    assert timed==value
    for row in value['rows']:
        row['methods']['analytic'].update(status='ok',arm='data_metric',terminal_budget=timed_report('data_metric')['terminal_budget'])
    path.write_text(json.dumps(value));timed,_=miit.load_manifest(path)
    assert timed==value
