import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_distortion_budget_batch as batch
from tests.test_coordinated_match_fusion import fixtures


def report_fixture():
    stages=[]
    for level,side in zip((17,33,65,129,257),(32,64,128,256,512)):
        for direction in ([1.,0.],[0.,1.]):
            stages.append(dict(level=level,image_side=side,direction=direction,maximum_gradients=30,
                numerical_failure=False,stop_reason='finite_non_descent',distortion_budget=.2,
                anchor_distortion=.2,accepted_distortion=.19,actual_distortion=.19,
                counts=dict(gradient_steps=2,data_vjps=2,distortion_vjps=2,numerical_failures=0)))
    return dict(budget_mode='distortion_cap',optimizer='incumbent_ARAP_budget_spectral_QCQP',
        numerical_failure=False,schedule_complete=True,maximum_gradient_steps=300,distortion_budget=.2,
        stages=stages,gradient_steps=20,counts=dict(gradient_steps=20,data_vjps=20,distortion_vjps=20,numerical_failures=0),
        initial=dict(total=2.,strain=.2,original_total=2.6),final=dict(total=1.9,strain=.19,original_total=2.47))


def test_maximum_not_fabricated_completed_budget_and_actual_fixed_cap():
    value=report_fixture()
    assert batch.validate_budget(value)==.2
    for change in ('inflated_cap','fake_gradients','nan_final','missing_stage','numerical_failure'):
        bad=copy.deepcopy(value)
        if change=='inflated_cap':bad['stages'][1]['distortion_budget']=.21
        elif change=='fake_gradients':bad['gradient_steps']=300
        elif change=='nan_final':bad['final']['strain']=float('nan')
        elif change=='missing_stage':bad['stages'].pop()
        else:bad['numerical_failure']=True
        with pytest.raises(ValueError):batch.validate_budget(bad)


def test_same_incumbent_control_and_complete_failed_denominator(tmp_path):
    cases,_=fixtures(tmp_path)
    for case in cases:case['original_configuration']['match_weight']=.2
    control_rows=[]
    for case in cases:
        cfg=batch.configuration(case['original_configuration'],tmp_path/'old.npz')
        control_rows.append(dict(name=case['name'],status='ok',gradient_steps=300,
            incumbent=str(Path(case['source_report']).with_suffix('.npz')),
            configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
    control=dict(prediction_complete=True,all50_terminal=True,annotations_read=False,arm='frozen_suffix',rows=control_rows)
    path=tmp_path/'frozen.json';path.write_text(json.dumps(control))
    calls=[]
    def optimize(cfg,*,initial_map):
        calls.append((cfg,initial_map))
        result=report_fixture()
        result['initial_map']=dict(path=str(initial_map))
        if len(calls)==4:result['numerical_failure']=True
        return result
    output=tmp_path/'budget'
    result=batch.run(tmp_path/'unused.json',path,output,source_loader=lambda _:cases,
        optimizer=optimize,output_validator=lambda *a:.1)
    assert len(calls)==25 and result['prediction_complete'] and result['all25_terminal']
    assert sum(r['status']=='failed' for r in result['rows'])==1 and result['rows'][3]['numerical_failure']
    for cohort in ('miit','existing'):
        manifest=json.loads((output/(cohort+'_predictions.json')).read_text())
        assert manifest['budget_mode']=='distortion_cap' and manifest['maximum_gradient_steps']==300
    bad=copy.deepcopy(control);bad['rows'][0]['configuration']['matches']='changed.json'
    with pytest.raises(ValueError,match='changed incumbent'):batch.check_control(cases,bad)

