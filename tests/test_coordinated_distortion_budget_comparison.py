"""Synthetic read-only comparison checks; no actual new labels or scores."""
import copy
import importlib.util
from pathlib import Path
import pytest

from tools.coordinated_dhr_existing_inputs import existing_rows
from tests.test_coordinated_distortion_budget_batch import report_fixture

SOURCE=Path(__file__).resolve().parents[1]/'outputs/coordinated_instance_registration/check_sources/distortion_budget_comparison_t28.py'
spec=importlib.util.spec_from_file_location('budget_comparison_test',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'BASE',tmp_path)
    directories=dict(fusion300=tmp_path/'match_fusion_all50_t23/fusion',
        frozen_suffix=tmp_path/'refreshed_match_all50_t27/frozen_suffix',distortion_cap=tmp_path/'cap')
    names=['miit_2_to_3','miit_7_to_8','miit_10_to_11']+[r['name'] for r in existing_rows(Path('unused'))]
    files={};manifests={};calls=[]
    for index,(method,directory) in enumerate(directories.items()):
        rows=[]
        for name in names:
            incoming='/remote/fusion/'+name+'_analytic.npz'
            value=report_fixture();value.update(initial_map=dict(path=incoming),name=name,status='ok',
                incumbent=incoming,output=name+'_analytic.npz',report=name+'_analytic.json',
                complete_call_seconds=4.+index,peak_allocated_bytes=100*(index+1))
            value['initial'].update(image=1.5,oob=.2,shape=100.,match=1.45)
            value['final'].update(image=1.4,oob=.2,shape=100.,match=1.45)
            if method!='distortion_cap':
                value['gradient_steps']=300
                for state in ('initial','final'):value[state]['total']=value[state]['original_total']
            rows.append(copy.deepcopy(value));files[directory/value['report']]=copy.deepcopy(value)
        manifest=dict(rows=rows,prediction_complete=True,all50_terminal=True,all25_terminal=True,
            annotations_read=False,budget_mode='distortion_cap',source_fusion_predictions='/remote/fusion/predictions.json')
        manifests[method]=manifest;files[directory/'predictions.json']=manifest
        scored=lambda:dict(status='ok',metrics=dict(canvas_pixels=dict(mean=4.-index,p90=8.-index,maximum=12.-index,
            per_label={'a':3.,'b':4.})))
        files[directory/'miit_scores.json']=dict(rows=[dict(name=n,methods=dict(analytic=scored())) for n in names[:3]])
        files[directory/'existing_scores.json']=dict(rows=[dict(name=n,**scored()) for n in names[3:]])
    def read(path):
        calls.append(Path(path));return files[Path(path)]
    monkeypatch.setattr(module,'read',read)
    return directories,manifests,files,calls


def test_all25_groups_regressions_costs_and_own_objective_fields(tmp_path,monkeypatch):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    value=module.compare(directories['distortion_cap'])
    assert len(value['rows'])==25 and len(value['cohorts'])==4
    assert value['regressions']['cap_minus_frozen_suffix']['mean']['improved']==25
    assert value['cohorts']['lung_all20']['methods']['distortion_cap']['pair_denominator']==20
    for method in module.METHODS:assert value['costs'][method]['peak_allocated_bytes']['total'] is None
    assert value['costs']['distortion_cap']['recorded_incumbent_plus_suffix_calls']['total']==250
    diagnostics=value['rows'][0]['objectives']
    assert diagnostics['distortion_cap']['final_D']==pytest.approx(1.9)
    assert diagnostics['distortion_cap']['original_E']==2.47
    assert diagnostics['frozen_suffix']['original_E']==2.47
    assert diagnostics['distortion_cap']['gradient_steps']==20
    assert diagnostics['frozen_suffix']['gradient_steps']==300
    assert value['failed_attempts']==[]


@pytest.mark.parametrize('failure',['prediction_pending','missing_row','protocol_incomplete','bad_stage_budget'])
def test_all_completion_and_actual_budget_checks_precede_scores(tmp_path,monkeypatch,failure):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch);cap=manifests['distortion_cap']
    if failure=='prediction_pending':cap['rows'][7]['status']='pending'
    elif failure=='missing_row':cap['rows'].pop()
    elif failure=='protocol_incomplete':cap['all25_terminal']=False
    else:files[directories['distortion_cap']/cap['rows'][4]['report']]['stages'][1]['distortion_budget']=.21
    with pytest.raises(ValueError):module.compare(directories['distortion_cap'])
    assert not any(p.name.endswith('_scores.json') for p in calls)


def test_failed_case_keeps_denominators_cost_and_undefined_aggregate(tmp_path,monkeypatch):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    cap=manifests['distortion_cap'];cap['rows'][3].update(status='failed',error='numeric failure')
    files[directories['distortion_cap']/'existing_scores.json']['rows'][0].update(status='failed',error='no map')
    value=module.compare(directories['distortion_cap'])
    group=value['cohorts']['lung_all20']['methods']['distortion_cap']
    assert group['pair_denominator']==20 and group['scored_pairs']==19 and group['failure_count']==1
    assert group['all_directions_canvas_pixels'] is None
    assert value['equal_specimen_deltas']['cap_minus_frozen_suffix'] is None
    assert value['regressions']['cap_minus_frozen_suffix']['mean']['compared']==24
    assert len(value['failed_attempts'])==1
    assert value['costs']['distortion_cap']['complete_calls']['total']==150
    assert value['costs']['distortion_cap']['recorded_incumbent_plus_suffix_calls']['total']==250


@pytest.mark.parametrize('field',['missing','none'])
def test_unknown_suffix_cost_not_zero_or_exception(tmp_path,monkeypatch,field):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    row=manifests['distortion_cap']['rows'][3]
    if field=='missing':row.pop('complete_call_seconds')
    else:row['complete_call_seconds']=None
    value=module.compare(directories['distortion_cap'])
    costs=value['costs']['distortion_cap']['recorded_incumbent_plus_suffix_calls']
    assert costs['denominator']==25 and costs['recorded_count']==24 and costs['missing_count']==1
    assert costs['total']==240


@pytest.mark.parametrize('damage',['row_mismatch','both_substituted','actual_cap_substituted','actual_frozen_substituted'])
def test_same_saved_incumbent_means_actual_paths_and_original_fusion_output(tmp_path,monkeypatch,damage):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    cap=manifests['distortion_cap']['rows'][0];frozen=manifests['frozen_suffix']['rows'][0]
    report=files[directories['distortion_cap']/cap['report']]
    if damage=='row_mismatch':cap['incumbent']='/other.npz'
    elif damage=='both_substituted':
        cap['incumbent']=frozen['incumbent']='/other.npz'
        cap['initial_map']['path']=frozen['initial_map']['path']=report['initial_map']['path']='/other.npz'
    elif damage=='actual_cap_substituted':cap['initial_map']['path']=report['initial_map']['path']='/other.npz'
    else:frozen['initial_map']['path']='/other.npz'
    with pytest.raises(ValueError):module.compare(directories['distortion_cap'])
    assert not any(p.name.endswith('_scores.json') for p in calls)


def test_data_parts_not_catastrophic_originalE_minus_threeR():
    parts=dict(image=.3,oob=.2,shape=100.,match=.7,strain=1e30,original_total=3*1e30+.65)
    assert module.data_value(parts)==pytest.approx(.65)
    assert parts['original_total']-3*parts['strain']==0.
    parts['strain']=2e30
    assert module.data_value(parts)==pytest.approx(.65)


@pytest.mark.parametrize('damage',['failed_scored_ok','pending_score','different_label_ids'])
def test_invalid_score_status_or_labels_rejected(tmp_path,monkeypatch,damage):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    scored=files[directories['distortion_cap']/'existing_scores.json']['rows'][0]
    if damage=='failed_scored_ok':manifests['distortion_cap']['rows'][3]['status']='failed'
    elif damage=='pending_score':scored['status']='pending'
    else:scored['metrics']['canvas_pixels']['per_label']={'other':2.}
    with pytest.raises(ValueError):module.compare(directories['distortion_cap'])


def test_unknown_incumbent_cost_also_stays_unknown(tmp_path,monkeypatch):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    manifests['fusion300']['rows'][0]['complete_call_seconds']=None
    value=module.compare(directories['distortion_cap'])
    for method in ('frozen_suffix','distortion_cap'):
        assert value['costs'][method]['recorded_incumbent_plus_suffix_calls']['missing_count']==1


def test_manifest_cannot_substitute_different_objective_parts_from_actual_report(tmp_path,monkeypatch):
    directories,manifests,files,calls=fixture(tmp_path,monkeypatch)
    manifests['distortion_cap']['rows'][0]['final']['image']+=.1
    with pytest.raises(ValueError):module.compare(directories['distortion_cap'])
    assert not any(p.name.endswith('_scores.json') for p in calls)
