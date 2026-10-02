"""Independent saved cap-run geometry/objective and recorded-work audit.

Only saved incoming/final maps are independently reevaluated. Unsaved trial
geometry, gradients and spectral vectors are not reconstructed from summaries.
Original CSV access requires --scores AND a complete all25 terminal manifest.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_data_metric_postrun_20261002 import (
    BASE,ARCHIVE,read,load_map,objective_data,literal_objective,label_check)

COUNT_KEYS=('gradient_steps','data_vjps','distortion_vjps','objective_evaluations',
    'trial_attempts','trial_evaluations','accepted_steps','backtracks','rejected_trials',
    'dual_evaluations','dual_bracket_probes','dual_bisections','numerical_failures')
LEGAL_STOPS={'gradient_budget','line_search_exhausted','zero_data_gradient',
    'zero_ellipsoid','zero_computed_direction','finite_nondescent_direction'}


def close(a,b,tolerance=2e-15):
    assert np.isfinite(a) and np.isfinite(b) and abs(a-b)<=tolerance,(a,b,abs(a-b))


def direct_D(parts):
    return parts['image']+parts['oob']+1e-4*parts['shape']+.2*parts['match']


def audit_recorded_trace(report):
    """Arithmetic/status conservation only: no unsaved trial-map claims."""
    B=report['distortion_budget'];stages=report['stages'];trace=report['trace']
    assert np.isfinite(B) and B>=0 and report['initial']['strain']==B
    assert report['budget_mode']=='distortion_cap' and report['maximum_gradient_steps']==300
    assert report['optimizer']=='incumbent_ARAP_budget_spectral_QCQP'
    assert report['schedule_complete'] and not report['numerical_failure'] and len(stages)==10
    assert report['control_vertices']==257**2 and report['query_count']==512**2 and not report['landmarks_used']
    assert report['output_selection']=='best_full_data_subject_to_incumbent_ARAP_budget'
    totals=Counter();stops=Counter();dual_active=0;maximum_mu=0.;min_recorded_slack=float('inf')
    for index,stage in enumerate(stages):
        records=[r for r in trace if r['stage']==index]
        level=(17,33,65,129,257)[index//2];side=(32,64,128,256,512)[index//2]
        axis=([1.,0.],[0.,1.])[index%2]
        assert stage['level']==level and stage['image_side']==side and stage['direction']==axis
        assert stage['maximum_gradients']==30 and stage['physical_rms_step']==.004*16/(level-1)
        assert not stage['numerical_failure'] and stage['distortion_budget']==B
        assert stage['stop_reason'] in LEGAL_STOPS
        assert stage['minimum_contracted_slack']>0
        assert [r['step'] for r in records]==list(range(len(records))) and 1<=len(records)<=30
        assert all(r['level']==level and r['image_side']==side and r['direction']==axis for r in records)
        counts=Counter({k:0 for k in COUNT_KEYS});counts.update(gradient_steps=len(records),data_vjps=len(records),distortion_vjps=len(records))
        counts['objective_evaluations']=len(records)
        value=stage['anchor_total'];R=stage['anchor_distortion']
        if index:assert R==stages[index-1]['accepted_distortion']
        else:assert R==B
        if index%2:assert value==stages[index-1]['accepted_total']
        for step,row in enumerate(records):
            assert row['objective']==value and row['distortion']==R and 0<=R<=B
            assert row['distortion_budget']==B and row['remaining_budget']==B-R
            close(direct_D(row['parts']),value)
            assert row['parts']['strain']==R
            close(row['parts']['original_total'],value+3*R)
            dual=row['dual'];assert not dual['numerical_failure'] and dual['approximate_root'] and not dual['exact_kkt_claim']
            for key in ('dual_evaluations','dual_bracket_probes','dual_bisections'):counts[key]+=dual[key]
            assert 0<=dual['dual_bracket_probes']<=32 and 0<=dual['dual_bisections']<=32
            if dual['stop_reason'] in ('zero_data_gradient','zero_ellipsoid'):
                assert dual['dual_evaluations']==0 and not row['trials'] and not row['accepted']
                assert step==len(records)-1 and stage['stop_reason']==dual['stop_reason']
                continue
            assert dual['dual_evaluations']==1+dual['dual_bracket_probes']+dual['dual_bisections']
            assert dual['constraint_residual']<=0 and np.isfinite(dual['constraint_residual'])
            assert dual['tau']>0 and np.isfinite(dual['tau'])
            assert dual['multiplier']==dual['bracket_upper']>=dual['bracket_lower']>=0
            assert np.isfinite(dual['multiplier'])
            maximum_mu=max(maximum_mu,dual['multiplier'])
            if dual['cap_active']:
                dual_active+=1;assert dual['dual_bracket_probes']>=1
                assert dual['multiplier']<=3*2**(dual['dual_bracket_probes']-1)
                initial_width=3*2**max(dual['dual_bracket_probes']-2,0)
                assert dual['bracket_upper']-dual['bracket_lower']<=initial_width/2**dual['dual_bisections']+1e-12
            else:
                assert dual['multiplier']==0 and dual['dual_bracket_probes']==dual['dual_bisections']==0
                close(dual['proposal_physical_rms'],stage['physical_rms_step'],2e-14)
            if dual['stop_reason'] is not None:
                assert dual['stop_reason'] in ('zero_computed_direction','finite_nondescent_direction')
                assert not row['trials'] and not row['accepted'] and step==len(records)-1
                assert stage['stop_reason']==dual['stop_reason'];continue
            slope=row['directional_derivative'];assert slope==dual['directional_derivative']<0 and np.isfinite(slope)
            assert row['minimum_contracted_slack']>0
            min_recorded_slack=min(min_recorded_slack,row['minimum_contracted_slack'])
            amax=row['alpha_max']
            assert row['initial_alpha']==.99*min(1.,float('inf') if amax is None else amax)
            assert amax is None or amax>0
            trials=row['trials'];assert 1<=len(trials)<=13
            counts['trial_attempts']+=len(trials);counts['backtracks']+=len(trials)-1
            assert sum(t['accepted'] for t in trials)==int(row['accepted'])
            assert not any(t['accepted'] for t in trials[:-1])
            for j,trial in enumerate(trials):
                assert trial['alpha']==row['initial_alpha']/2**j and 0<trial['alpha']<=.99
                close(trial['armijo_rhs'],value+1e-4*trial['alpha']*slope,0.)
                assert 'failure' not in trial
                if trial['geometry_valid']:
                    counts['trial_evaluations']+=1;counts['objective_evaluations']+=1
                    d=trial['objective'];rr=trial['distortion'];assert np.isfinite(d) and np.isfinite(rr)
                    assert trial['budget_valid']==(0<=rr<=B)
                    assert trial['armijo_valid']==(d<=trial['armijo_rhs'])
                    assert trial['strict_descent']==(d<value)
                    expected=trial['budget_valid'] and trial['armijo_valid'] and trial['strict_descent']
                    assert trial['accepted']==expected
                else:
                    assert trial['objective'] is None and trial['distortion'] is None and not trial['accepted']
                if trial['accepted']:
                    counts['accepted_steps']+=1
                    assert row['accepted_alpha']==trial['alpha'] and row['accepted_objective']==trial['objective']
                    assert row['accepted_distortion']==trial['distortion']
                    close(row['accepted_displacement_rms'],trial['alpha']*dual['proposal_physical_rms'],1e-15)
                else:counts['rejected_trials']+=1
            if row['accepted']:
                value=row['accepted_objective'];R=row['accepted_distortion']
            else:
                assert step==len(records)-1 and len(trials)==13 and stage['stop_reason']=='line_search_exhausted'
        assert value==stage['accepted_total'] and R==stage['accepted_distortion']==stage['actual_distortion'] and 0<=R<=B
        close(stage['accepted_original_full_total'],stage['accepted_full_total']+3*R)
        assert dict(counts)==stage['counts'],(index,counts,stage['counts'])
        if stage['stop_reason']=='gradient_budget':assert len(records)==30 and records[-1]['accepted']
        totals.update(counts);stops[stage['stop_reason']]+=1
    assert sum(totals[k] for k in ('accepted_steps','rejected_trials'))==totals['trial_attempts']
    assert dict(totals)==report['counts'] and totals['gradient_steps']==report['gradient_steps']==len(trace)
    assert report['evaluations']==totals['trial_evaluations']
    assert report['wrapper_objective_evaluations']==12 and report['objective_evaluations']==totals['objective_evaluations']+12
    values=[report['initial']['total']]+[s['accepted_full_total'] for s in stages]
    choice=int(np.argmin(values));assert report['selected_stage']==(None if choice==0 else choice-1)
    assert report['final']['total']==values[choice]
    selected_R=B if choice==0 else stages[choice-1]['actual_distortion']
    assert report['final']['strain']==selected_R<=B
    close(report['final']['original_total'],report['final']['total']+3*selected_R)
    return dict(counts=dict(totals),stage_stops=dict(stops),active_dual_directions=dual_active,
        maximum_multiplier=maximum_mu,minimum_recorded_contracted_slack=min_recorded_slack,
        scope='record arithmetic and actual reported acceptances; unsaved trial maps and spectral vectors not independently reevaluated')


def audit_objective(expected,record,errors):
    expected=dict(expected)
    expected['original_total']=expected.pop('total')
    expected['total']=direct_D(expected)
    for key,value in expected.items():
        error=abs(value-record[key]);errors[key]=max(errors.get(key,0.),error)
        tolerance=3e-7 if key in ('image','total','original_total') else (5e-8 if key=='outside_fraction' else 2e-10)
        assert error<tolerance,(key,error,value,record[key])
    return expected


def audit_case(directory,name,row):
    report=read(directory/row['report']);old=read(ARCHIVE/(name+'_analytic.json'))
    incoming,a,b,q0=load_map(ARCHIVE/(name+'_analytic.npz'))
    output,_,_,qmin=load_map(directory/row['output'],a,b)
    assert {k for k in old['configuration'] if old['configuration'][k]!=report['configuration'][k]}=={'output'}
    assert Path(report['initial_map']['path']).name==name+'_analytic.npz'
    assert '/match_fusion_all50_t23/fusion/' in report['initial_map']['path'].replace('\\','/')
    close(report['initial_map']['minimum_corner_ratio'],q0,1e-13)
    assert report['initial_map']['saved_binary_certificate']['valid']
    if 'incumbent' in row:assert row['incumbent']==report['initial_map']['path']
    if 'actual_minimum_corner_ratio' in row:close(qmin,row['actual_minimum_corner_ratio'],1e-13)
    for key in ('gradient_steps','counts','maximum_gradient_steps','numerical_failure','schedule_complete',
        'objective_evaluations','evaluations','initial','final','distortion_budget','budget_mode','stages',
        'initial_map','selected_stage','output_selection','saved_binary_certificate','peak_allocated_bytes',
        'loading_seconds','feature_seconds','optimize_seconds','serialization_seconds','certification_seconds',
        'end_to_end_seconds','control_vertices','query_count','optimizer'):
        if key in row:assert row[key]==report[key],key
    trace=audit_recorded_trace(report)
    evidence=objective_data(name,old['configuration'],a,b);errors={}
    before=literal_objective(incoming,evidence);after=literal_objective(output,evidence)
    original_before=audit_objective(before,report['initial'],errors)
    original_after=audit_objective(after,report['final'],errors)
    for key in ('image','strain','shape','oob','outside_fraction','match'):assert report['initial'][key]==old['final'][key]
    assert report['initial']['original_total']==old['final']['total']
    assert original_after['strain']<=original_before['strain']
    assert report['final']['strain']<=report['distortion_budget']==report['initial']['strain']
    if report['selected_stage'] is None:assert np.array_equal(output,incoming)
    assert 'per original area-reduced raw raster scale' in report['image_preprocessing']['moving_descriptor_prewarp_scope']
    assert set(report['mind_frame_by_resolution'])=={'32','64','128','256','512'}
    assert sum(report[k] for k in ('loading_seconds','feature_seconds','optimize_seconds','serialization_seconds','certification_seconds'))<=report['end_to_end_seconds']
    return dict(name=name,minimum_corner_ratio=qmin,initial=report['initial'],final=report['final'],
        selected_stage=report['selected_stage'],independent_initial_R=original_before['strain'],
        independent_final_R=original_after['strain'],trace=trace,maximum_objective_component_errors=errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--scores',action='store_true')
    parser.add_argument('--directory',type=Path,default=BASE/'distortion_budget_all25_t28');args=parser.parse_args()
    if args.smoke:
        assert not args.scores
        directory=BASE/'distortion_budget_smoke_t28'
        result=audit_case(directory,'miit_2_to_3',dict(report='miit_2_to_3.json',output='miit_2_to_3.npz'))
        print(json.dumps(dict(scope='label-free saved smoke only',result=result),indent=2));return
    directory=args.directory;manifest=read(directory/'predictions.json')
    original=read(ARCHIVE/'predictions.json')
    assert manifest['prediction_complete'] and manifest['all25_terminal'] and not manifest['annotations_read']
    assert manifest['budget_mode']=='distortion_cap' and manifest['cohort_size']==25 and manifest['specimen_count']==4
    assert [r['name'] for r in manifest['rows']]==[r['name'] for r in original['rows']]
    assert len(manifest['rows'])==25 and all(r['status'] in ('ok','failed') for r in manifest['rows'])
    checked=[];failures=[];labels=0;errors={};score_errors={}
    for row in manifest['rows']:
        if row['status']=='failed':
            failures.append(dict(name=row['name'],error=row.get('error'),numerical_failure=row.get('numerical_failure'),
                scope='retained in full25 denominator; not recast as successful registration'))
            continue
        result=audit_case(directory,row['name'],row)
        for key,error in result['maximum_objective_component_errors'].items():errors[key]=max(errors.get(key,0.),error)
        if args.scores:
            scored=label_check(directory,row['name'],row,'');result['scores']=scored['metrics'];labels+=scored['landmarks']
            for key,error in scored['maximum_errors'].items():score_errors[key]=max(score_errors.get(key,0.),error)
        checked.append(result);print(json.dumps(dict(checked=row['name'],gradients=result['trace']['counts']['gradient_steps'],objective_error=errors['total'])),flush=True)
    if args.scores and not failures:assert labels==2074
    counts=Counter();stops=Counter()
    for r in checked:counts.update(r['trace']['counts']);stops.update(r['trace']['stage_stops'])
    result=dict(all25_terminal_denominator_preserved=True,successful_maps_independently_recomputed=len(checked),
        failures=failures,actual_trial_record_scope='record consistency only; no saved intermediate maps or spectral gradient vectors',
        saved_endpoint_R_D_E_and_binary_geometry_independently_recomputed=True,original_CSV_scores_checked=args.scores,
        total_original_CSV_errors=labels,maximum_original_CSV_score_errors=score_errors,
        maximum_objective_component_errors=errors,minimum_saved_corner_ratio=min(r['minimum_corner_ratio'] for r in checked),
        total_actual_counts=dict(counts),stage_stop_counts=dict(stops),rows=checked)
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
