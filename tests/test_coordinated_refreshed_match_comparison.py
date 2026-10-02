import importlib.util
from pathlib import Path
import pytest
from tools.coordinated_dhr_existing_inputs import existing_rows

PATH=Path(__file__).resolve().parents[1]/'outputs/coordinated_instance_registration/check_sources/refreshed_match_comparison_t27.py'
spec=importlib.util.spec_from_file_location('refresh_comparison_test',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    names=['miit_2_to_3','miit_7_to_8','miit_10_to_11']+[r['name'] for r in existing_rows(Path('unused'))]
    outer=dict(prediction_complete=True,extraction_complete=True,annotations_read=False,attempt_denominator=50,
        source_fusion_predictions='/remote/fusion/predictions.json',extractions=[dict(name=n,status='ok',complete_call_seconds=2.,
        extraction=dict(peak_allocated_bytes=200)) for n in names],model_setup_status='ok',model_setup_complete_seconds=3.,
        model_setup=dict(setup_peak_allocated_bytes=300))
    manifests={}
    for method in module.METHODS:
        rows=[]
        for n in names:
            incumbent='/remote/fusion/'+n+'_analytic.npz'
            rows.append(dict(name=n,status='ok',output=incumbent,report=n+'_analytic.json',incumbent=incumbent,
                initial_map=dict(path=incumbent),gradient_steps=300,new_suffix_gradient_steps=300,
                failed_trials=0,initial=dict(total=1.),final=dict(total=.9),peak_allocated_bytes=100,
                complete_call_seconds=4.,actual_minimum_corner_ratio=.2))
        manifests[method]=dict(rows=rows,prediction_complete=True,all50_terminal=True,annotations_read=False)
    scores={}
    for i,method in enumerate(module.METHODS):
        rec=lambda:dict(status='ok',metrics=dict(canvas_pixels=dict(mean=4-i,p90=8-i,maximum=12-i)))
        scores[method]=(dict(rows=[dict(name=n,methods=dict(analytic=rec())) for n in names[:3]]),
            dict(rows=[dict(name=n,**rec()) for n in names[3:]]))
    return outer,manifests,scores


def test_all25_fourcohorts_costs_objectives_and_regressions():
    outer,manifests,scores=fixture()
    result=module.compare(outer,manifests,scores,{})
    assert len(result['rows'])==25 and len(result['cohorts'])==4
    assert result['cohorts']['miit']['deltas']['refresh_minus_frozen']['mean_pair_mean']==-1
    assert result['regressions']['refresh_minus_frozen']['mean']['improved_count']==25
    assert result['costs']['incumbent300']['peak_allocated_bytes']['total'] is None
    assert result['costs']['refreshed_suffix']['recorded_incumbent_plus_suffix_calls']['total']==200
    assert result['refresh_costs']['setup_complete_seconds']==3
    assert result['refresh_costs']['extraction_complete_calls']['total']==50
    assert result['costs']['refreshed_suffix']['recorded_incumbent_suffix_refresh_calls']['total']==250
    assert result['objectives'][0]['methods']['refreshed_suffix']['own_final_minus_initial']==pytest.approx(-.1)
    assert result['historical_preparation']['unknown_initializer_and_sg_seconds'] is None


def test_failed_extraction_keeps_frozen_control_and_denominator():
    outer,manifests,scores=fixture();i=3
    outer['extractions'][i].update(status='failed',error='injected')
    manifests['refreshed_suffix']['rows'][i].update(status='failed',error='no refresh')
    scores['refreshed_suffix'][1]['rows'][0].update(status='failed',error='no map')
    result=module.compare(outer,manifests,scores,{})
    assert result['extraction_denominator']==25 and result['extraction_failed']==1
    assert result['cohorts']['lung_all20']['methods']['refreshed_suffix']['pair_denominator']==20
    assert result['cohorts']['lung_all20']['methods']['refreshed_suffix']['all_directions_canvas_pixels'] is None
    assert result['cohorts']['lung_all20']['methods']['frozen_suffix']['scored_pairs']==20
    assert result['equal_specimen_deltas']['refresh_minus_frozen'] is None


@pytest.mark.parametrize('damage',['pending','missing','wrong_incumbent','extraction_success_fabricated'])
def test_reject_incomplete_and_wrong_initial_map(damage):
    outer,manifests,scores=fixture()
    if damage=='pending':outer['extraction_complete']=False
    if damage=='missing':manifests['frozen_suffix']['rows'].pop()
    if damage=='wrong_incumbent':manifests['refreshed_suffix']['rows'][0]['initial_map']['path']='/other.npz'
    if damage=='extraction_success_fabricated':outer['extractions'][0]['status']='failed'
    with pytest.raises(ValueError):module.compare(outer,manifests,scores,{})


def test_no_scores_opened_until_all_terminal(tmp_path,monkeypatch):
    outer,manifests,_=fixture();outer['prediction_complete']=False
    calls=[]
    def read(path):
        calls.append(str(path))
        assert str(path).endswith('predictions.json')
        return outer if Path(path).parent==tmp_path else manifests['fusion300' if 'fusion' in str(path) else Path(path).parent.name]
    monkeypatch.setattr(module,'read',read)
    with pytest.raises(ValueError):module.build(tmp_path,fusion_directory=tmp_path/'fusion')
    assert not any('scores' in Path(p).name for p in calls)


def test_adverse_tail_is_reported_despite_mean_gain():
    outer,manifests,scores=fixture()
    scores['refreshed_suffix'][1]['rows'][20]['metrics']['canvas_pixels']['maximum']=30
    result=module.compare(outer,manifests,scores,{})
    adverse=result['regressions']['refresh_minus_frozen']['maximum']
    assert adverse['regressed_count']==1 and adverse['regressed_pairs'][0]['name']=='histo'
    assert result['cohorts']['histo']['deltas']['refresh_minus_frozen']['worst_pair_maximum']==19
