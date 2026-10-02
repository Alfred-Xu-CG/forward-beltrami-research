"""Independent reverse role/inverse-preparation and declared-denominator checks.

This precheck opens no landmark CSV and launches no inference or optimizer.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'outputs/coordinated_instance_registration'
MIIT=Path('D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')
NAMES=('miit_2_to_3','miit_7_to_8','miit_10_to_11')
REVERSES=('miit_3_to_2','miit_8_to_7','miit_11_to_10')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def original_cases():
    rows=read(BASE/'match_fusion_all50_t23/fusion/predictions.json')['rows'][:3]
    cases=[dict(name=row['name'],cohort=row['cohort'],original_configuration=copy.deepcopy(row['configuration'])) for row in rows]
    for row in cases:
        cfg=row['original_configuration']
        for key in ('fixed','moving','affine','matches'):
            cfg[key]=str(BASE/'miit_three_rotations_t153'/Path(cfg[key]).name)
    return cases


def explicit_inverse(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    determinant=a[0,0]*a[1,1]-a[0,1]*a[1,0]
    assert determinant>0
    ai=np.array([[a[1,1],-a[0,1]],[-a[1,0],a[0,0]]])/determinant
    return ai,-ai@b


def inspect_prepared(row,case,directory):
    name=row['name'];forward=case['name'];cfg=case['original_configuration']
    _,oldmoving,_,oldfixed=forward.split('_')
    assert name==f'miit_{oldfixed}_to_{oldmoving}'
    assert row['moving_section']==int(oldfixed) and row['fixed_section']==int(oldmoving)
    assert row['map_direction']=='fixed_canvas_to_moving_canvas' and row['source_forward_case']==forward
    before=read(BASE/'miit_three_rotations_t153'/(forward+'_layout.json'))
    layout=read(directory/row['layout'])
    assert layout['side']==512 and layout['role_swap_of']==forward
    for role,old in (('fixed','moving'),('moving','fixed')):
        assert (directory/row[role]).read_bytes()==Path(cfg[old]).read_bytes()
        assert row['accepted512_layouts'][role]==layout[role]
        for key,value in before[old].items():
            if key not in ('source','canvas_png'):assert layout[role][key]==value
        section=oldmoving if role=='fixed' else oldfixed
        assert Path(row['source_'+role])==MIIT/section/'images/image.tif'
        assert layout[role]['source']==row['source_'+role]
        assert np.array_equal(np.array(layout[role]['resized_wh'])/layout[role]['original_wh'],layout[role]['effective_original_to_canvas_scale_xy'])
    with np.load(cfg['affine']) as original:a=original['post_affine_matrix'].astype(float);b=original['post_affine_offset'].astype(float)
    ai,bi=explicit_inverse(a,b)
    with np.load(directory/row['affine']) as saved:
        ar=saved['post_affine_matrix'];br=saved['post_affine_offset']
        assert ar.dtype==br.dtype==np.float32
        assert np.array_equal(ar,ai.astype(np.float32)) and np.array_equal(br,bi.astype(np.float32))
    assert np.array_equal(ar,row['shared_affine_matrix']) and np.array_equal(br,row['shared_affine_offset'])
    audit=row['inverse_audit'];assert audit['stored_dtype']=='float32' and audit['optimized_map_inverted'] is False
    assert np.max(abs(ai-audit['inverse_matrix_float64']))<3e-16 and np.max(abs(bi-audit['inverse_offset_float64']))<3e-16
    q=np.array([[0,0],[1,0],[1,1],[0,1]],float);ar=ar.astype(float);br=br.astype(float)
    reverse_after=float(np.linalg.norm(((q@a.T+b)@ar.T+br-q)*512,axis=1).max())
    forward_after=float(np.linalg.norm(((q@ar.T+br)@a.T+b-q)*512,axis=1).max())
    assert abs(reverse_after-audit['reverse_after_forward_max_corner_canvas_pixels'])<1e-12
    assert abs(forward_after-audit['forward_after_reverse_max_corner_canvas_pixels'])<1e-12
    assert audit['stored_determinant']==np.linalg.det(ar)>0
    expected=copy.deepcopy(cfg);expected.update(fixed=str(directory/row['fixed']),moving=str(directory/row['moving']),affine=str(directory/row['affine']),match_weight=.1)
    assert row['original_configuration']==expected
    return dict(name=name,stored_inverse_determinant=audit['stored_determinant'],
        inverse_roundtrip_corner_canvas_pixels=dict(reverse_after_forward=reverse_after,forward_after_reverse=forward_after),
        exact_original_canvas_bytes_and_layout_roles_swapped=True,only_supplied_affine_inverted=True)


def preparation():
    from tools import coordinated_miit_reverse as module
    checked=[]
    with tempfile.TemporaryDirectory(prefix='independent_reverse_') as temp:
        directory=Path(temp)
        for case in original_cases():
            row=module.prepare(case,MIIT,directory)
            checked.append(inspect_prepared(row,case,directory))
    return checked


def failure_denominator():
    from tools import coordinated_miit_reverse as module
    cases=original_cases()
    for row in cases:row['original_configuration']['device']='cpu'
    events=[]
    with tempfile.TemporaryDirectory(prefix='independent_reverse_failure_') as temp:
        directory=Path(temp)/'run'
        def sg(args):
            pending=read(directory/'predictions.json')
            assert len(pending['rows'])==3 and pending['attempt_denominator']==9 and not pending['prediction_complete']
            assert all(r['methods'][m]['status']=='pending' for r in pending['rows'] for m in module.METHODS)
            events.append('sg');raise ArithmeticError('independent injected SG failure')
        class Model:
            setup_report={}
            def __init__(self,*args,**kwargs):events.append('setup')
            def extract(self,*args,**kwargs):events.append('ma');raise ArithmeticError('independent injected MA failure')
            def close(self):events.append('close')
        def no_optimizer(*args,**kwargs):
            events.append('unexpected_optimizer')
            raise AssertionError('optimizer may not run after unavailable machine evidence')
        def native(*args,**kwargs):events.append('native');raise ArithmeticError('independent injected native failure')
        kwargs=dict(fusion_predictions='unused',source_data=MIIT,output=directory,source_root='unused',checkpoint='unused',
            cases=cases,sg_extractor=sg,model_factory=Model,optimizer=no_optimizer,native_runner=native)
        result=module.run(**kwargs)
        assert events==['sg']*3+['setup']+['ma']*3+['close']+['native']*3
        assert result['prediction_complete'] and result['all9_terminal'] and result['annotations_read'] is False
        assert result['successful_calls']==0 and result['failed_calls']==result['attempt_denominator']==9
        assert [r['name'] for r in result['rows']]==list(REVERSES)
        assert all(r['methods'][m]['status']=='failed' and r['methods'][m].get('error') for r in result['rows'] for m in module.METHODS)
        before=(directory/'predictions.json').read_bytes()
        try:module.run(**kwargs)
        except FileExistsError:pass
        else:raise AssertionError('existing output was not rejected')
        assert (directory/'predictions.json').read_bytes()==before
    return dict(all9_pending_before_first_inference=True,all9_failures_retained=True,
        no_archive_or_affine_fallback=True,existing_output_rejected_before_writes=True)


def synthetic_scoring():
    """Exercise actual reversed unequal frames using synthetic coordinates only."""
    from tools import coordinated_miit_reverse as module
    from tools import coordinated_miit_reverse_score as scorer
    with tempfile.TemporaryDirectory(prefix='independent_reverse_score_') as temp:
        directory=Path(temp);rows=[];points={};expected={};nominal=[str(i) for i in range(124)]
        for case in original_cases():
            row=module.prepare(case,MIIT,directory);rows.append(row)
            row['methods']={method:dict(status='failed',error='synthetic failed output retained') for method in module.METHODS}
            layout=read(directory/row['layout']);a=np.array(row['shared_affine_matrix']);b=np.array(row['shared_affine_offset'])
            q=np.column_stack((np.linspace(.45,.55,124),np.linspace(.44,.56,124)))
            fixed=(q*512-layout['fixed']['padding_xy'])/layout['fixed']['effective_original_to_canvas_scale_xy']-.5
            target=((q@a.T+b)*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5
            offset=np.array([.37,-.21]);target+=offset
            fixed[0]=np.inf;target[1]=np.inf
            points[row['fixed_section']]={label:xy for label,xy in zip(nominal,fixed)}
            points[row['moving_section']]={label:xy for label,xy in zip(nominal,target)}
            expected[row['name']]=dict(native_moving_pixels=float(np.linalg.norm(offset)),
                canvas_pixels=float(np.linalg.norm(offset*np.array(layout['moving']['effective_original_to_canvas_scale_xy']))))
        manifest=dict(prediction_complete=True,all9_terminal=True,annotations_read=False,attempt_denominator=9,
            cohort_size=3,scope='synthetic coordinates only',rows=rows)
        module.save(directory/'predictions.json',manifest)
        def synthetic(path,*args,**kwargs):return points[int(Path(path).stem)]
        with patch.object(scorer,'read_points',side_effect=synthetic):
            result=scorer.score(directory,MIIT)
        ids=sorted(set(nominal)-{'0','1'})
        for row in result['rows']:
            assert row['available_pair_labels']==ids and row['scored_landmarks']==122 and row['nominal_landmarks']==124
            assert row['unavailable_fixed_labels']==['0'] and row['unavailable_moving_labels']==['1']
            for unit,wanted in expected[row['name']].items():
                metrics=row['methods']['common_affine']['metrics'][unit]
                assert list(metrics['per_label'])==ids
                assert max(abs(value-wanted) for value in metrics['per_label'].values())<2e-12
            assert all(row['methods'][m]['status']=='failed' for m in module.METHODS)
        for method in module.METHODS:
            assert result['aggregate'][method]==dict(pair_denominator=3,scored_pairs=0,failed_pairs=3,all_three=None,successful_only=None)
        # Independently make a purported completed batch contain a pending ninth
        # attempt, and assert rejection before the coordinate-provider is called.
        manifest['rows'][-1]['methods']['native_shared']['status']='pending';module.save(directory/'predictions.json',manifest)
        with patch.object(scorer,'read_points',side_effect=AssertionError('coordinate access too early')) as csv_provider:
            try:scorer.score(directory,MIIT)
            except ValueError:pass
            else:raise AssertionError('pending ninth attempt accepted')
            csv_provider.assert_not_called()
    return dict(synthetic_coordinates_only=True,reversed_unequal_native_and_canvas_frames_checked=True,
        all124_nominal_and122_synthetic_available_IDs_preserved=True,nine_method_failures_retained=True,
        pending_ninth_attempt_rejected_before_coordinates=True)


if __name__=='__main__':
    print(json.dumps(dict(label_free_actual_preparation=preparation(),failure_denominator=failure_denominator(),
        synthetic_scoring=synthetic_scoring()),indent=2))
