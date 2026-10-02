"""Independent current-fusion A/F2 saved endpoint audit, no inference or labels before50 terminal."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_data_metric_postrun_20261002 import (
    BASE,ARCHIVE,IDENTITY,read,load_map,objective_data,literal_objective,compare_objective)
from independent_stain_proxy_probe_20261002 import literal_p1,original_landmark_case


def check_trace(saved,method):
    assert saved['gradient_steps']==300 and saved['evaluations']==310 and saved['objective_evaluations']==332
    assert saved['failed_trials']==0 and not saved['failures'] and not saved['landmarks_used']
    assert saved['output_selection']=='best_full' and saved['inner_steps_by_level']==[30]*5
    assert saved['control_vertices']==257**2 and saved['query_count']==512**2
    assert len(saved['stages'])==10 and len(saved['trace'])==310
    selected=[saved['initial']['total']]+[s['accepted_full_total'] for s in saved['stages']]
    assert saved['final']['total']==min(selected)
    index=saved['selected_stage'];assert saved['final']['total']==selected[0 if index is None else index+1]
    repeated=0.
    for i,stage in enumerate(saved['stages']):
        j=i//2 if method=='analytic' else i%5
        cycle=0 if method=='analytic' else i//5
        level=[17,33,65,129,257][j];raster=[32,64,128,256,512][j]
        direction=[[1.,0.],[0.,1.]][i%2] if method=='analytic' else None
        records=saved['trace'][31*i:31*(i+1)]
        expected=dict(cycle=cycle,level=level,control_side=257,image_side=raster,direction=direction)
        assert all(stage[k]==v for k,v in expected.items())
        assert stage['inner_steps']==30 and stage['physical_lr']==.004*16/(level-1)
        assert [r['step'] for r in records]==list(range(31))
        assert all(all(r[k]==v for k,v in expected.items()) for r in records)
        repeated=max(repeated,abs(stage['anchor_total']-records[0]['total']))
        assert abs(stage['accepted_total']-min([stage['anchor_total']]+[r['total'] for r in records]))<1e-12
    assert repeated<1e-12
    return repeated


def specimen(name):
    return 'MIIT' if name.startswith('miit_') else 'HistoReg' if name=='histo' else 'Kidney' if name=='rat_kidney' else 'Lung'


def score_agreement(directory,result):
    miit={r['name']:r for r in read(directory/'miit_scores.json')['rows']}
    existing={m:{r['name']:r for r in read(directory/m/'existing_scores.json')['rows']} for m in ('analytic','f2')}
    maximum={'canvas_pixels':0.,'native_moving_pixels':0.};count=0
    for row in result['rows']:
        name=row['name']
        for method,own in row['arms'].items():
            if own['status']!='ok':continue
            source=miit[name] if name.startswith('miit_') else existing[method][name]
            reported=source['methods'][method]['metrics'] if name.startswith('miit_') else source['metrics']
            assert source['available_pair_labels']==list(own['per_label'])
            for units,key in (('canvas_pixels','per_label'),('native_moving_pixels','native_per_label')):
                assert list(reported[units]['per_label'])==list(own[key])
                delta=max(abs(value-reported[units]['per_label'][label]) for label,value in own[key].items())
                maximum[units]=max(maximum[units],delta);assert delta<1e-9
            for metric in ('mean','p90','maximum'):
                assert abs(own[metric]-reported['canvas_pixels'][metric])<1e-10
            count+=len(own['per_label'])
    assert count==result['original_CSV_errors_recomputed']
    result['production_score_agreement']=dict(original_CSV_errors=count,maximum_errors=maximum)
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['production_score_agreement'],indent=2))


def audit(directory):
    outer=read(directory/'comparison.json')
    arms={m:read(directory/m/'predictions.json') for m in ('analytic','f2')}
    assert outer['attempt_denominator']==50 and outer['all50_terminal'] and outer['prediction_complete'] and not outer['annotations_read']
    assert all(a['all50_terminal'] and a['prediction_complete'] and not a['annotations_read'] and len(a['rows'])==25 for a in arms.values())
    assert all(r['status'] in ('ok','failed') for a in arms.values() for r in a['rows'])
    oldrows=read(ARCHIVE/'predictions.json')['rows'];names=[r['name'] for r in oldrows]
    assert all([r['name'] for r in a['rows']]==names for a in arms.values())
    schedule=[dict(case_index=i,name=name,method=m) for i,name in enumerate(names)
              for m in (('analytic','f2') if i%2==0 else ('f2','analytic'))]
    assert outer['planned_calls']==schedule and len(outer['runs'])==50
    for call,plan in zip(outer['runs'],schedule,strict=True):
        assert all(call[k]==v for k,v in plan.items())
        row=arms[plan['method']]['rows'][plan['case_index']]
        assert call['status']==row['status'] and call['complete_call_seconds']==row['complete_call_seconds']
    failures=[];rows=[];maxima={};labels=0;minimum=1.;times=0.;repeated=0.
    miit_adapter=read(directory/'analytic/miit_predictions.json')
    assert miit_adapter['actual_methods']==['analytic','f2'] and miit_adapter['archived_dhr_reference_only']
    for adapter in miit_adapter['rows']:
        for m in ('analytic','f2'):
            target=adapter['methods'][m]
            assert 'pilot_reference' not in target and target['configuration']['method']==m
            assert Path(target['output']).name==adapter['name']+'_'+m+'.npz'
        assert 'unchanged archived' in adapter['methods']['dhr']['pilot_reference']
    for i,name in enumerate(names):
        original=oldrows[i]['configuration'];_,a,b,_=load_map(ARCHIVE/Path(oldrows[i]['output']).name)
        evidence=objective_data(name,original,a,b);initial=literal_objective(IDENTITY,evidence)
        layout,points,ids=original_landmark_case(name)
        unit=lambda xy,l:((xy+.5)*l['effective_original_to_canvas_scale_xy']+l['padding_xy'])/512
        query=unit(points['fixed'],layout['fixed']);truth=unit(points['moving'],layout['moving'])
        case=dict(name=name,specimen=specimen(name),landmark_count=len(ids),arms={})
        initial_reports=[]
        for method in ('analytic','f2'):
            row=arms[method]['rows'][i];cfg=row['configuration'];folder=directory/method
            assert row['original_configuration']==original
            assert set(cfg)==set(original)
            changes={k for k in original if cfg[k]!=original[k]}
            assert changes==({'output'} if method=='analytic' else {'method','cycles','f2_floor_safety_fraction','geometry_backend','output'})
            assert cfg['joint_prior_backend']=='eager' and cfg['match_p1_sampling']=='existing'
            assert cfg['patch_cells']==8 and cfg['f2_accepted_gain']==1 and cfg['minimum_jacobian']==.001
            assert cfg['geometry_backend']==('stage_cache' if method=='analytic' else 'existing')
            assert cfg['cycles']==(1 if method=='analytic' else 2)
            if method=='f2':assert cfg['f2_floor_safety_fraction']==.95
            times+=row['complete_call_seconds']
            if row['status']!='ok':
                failure={k:row.get(k) for k in ('error','gradient_steps','evaluations','objective_evaluations','failed_trials','failures','saved_binary_certificate','complete_call_seconds')}
                failure.update(name=name,method=method);failures.append(failure)
                case['arms'][method]=dict(status='failed',**failure)
                if (folder/row['output']).is_file():
                    _,_,_,floor=load_map(folder/row['output'],a,b)
                    failure['partial_saved_minimum_corner_ratio']=floor
                continue
            saved=read(folder/row['report']);assert saved['configuration']==cfg
            repeated=max(repeated,check_trace(saved,method));initial_reports.append(saved['initial'])
            for key in ('initial','final','gradient_steps','evaluations','objective_evaluations','failed_trials','saved_binary_certificate','landmarks_used'):
                assert saved[key]==row[key]
            v,_,_,floor=load_map(folder/row['output'],a,b);minimum=min(minimum,floor)
            assert abs(floor-row['actual_minimum_corner_ratio'])<1e-13
            compare_objective(initial,saved['initial'],maxima)
            compare_objective(literal_objective(v,evidence),saved['final'],maxima)
            world=literal_p1(v@a.T+b,query)
            errors=np.linalg.norm((world-truth)*512,axis=-1);labels+=len(ids)
            native=np.linalg.norm((world*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5-points['moving'],axis=-1)
            memory=row['whole_call_memory'];assert memory['peak_reserved_bytes']>=memory['peak_allocated_bytes']>=saved['peak_allocated_bytes']
            assert memory['captured_segments']>=2
            case['arms'][method]=dict(status='ok',minimum_corner_ratio=floor,initial=saved['initial'],final=saved['final'],
                selected_stage=saved['selected_stage'],mean=float(errors.mean()),p90=float(np.percentile(errors,90)),maximum=float(errors.max()),
                per_label=dict(zip(ids,errors.tolist())),native_per_label=dict(zip(ids,native.tolist())),
                complete_call_seconds=row['complete_call_seconds'],whole_call_memory=memory)
        if len(initial_reports)==2:
            assert initial_reports[0]==initial_reports[1]
            aa,ff=case['arms']['analytic'],case['arms']['f2']
            case['analytic_minus_f2']={k:aa[k]-ff[k] for k in ('mean','p90','maximum')}
            case['analytic_minus_f2']['final_total']=aa['final']['total']-ff['final']['total']
            case['f2_over_analytic_call_ratio']=ff['complete_call_seconds']/aa['complete_call_seconds']
        rows.append(case)
        print(json.dumps(dict(checked=name,status={m:r['status'] for m,r in case['arms'].items()},maximum_objective_error=maxima.get('total'))),flush=True)
    assert outer['successful_calls']==50-len(failures) and outer['elapsed_seconds']>=times
    if not failures:assert labels==4148
    summary={}
    for group in ('MIIT','Lung','HistoReg','Kidney'):
        chosen=[c for c in rows if c['specimen']==group]
        summary[group]={}
        for method in ('analytic','f2'):
            values=[c['arms'][method] for c in chosen if c['arms'][method]['status']=='ok']
            summary[group][method]=dict(attempts=len(chosen),successful=len(values),failed=len(chosen)-len(values),
                mean=float(np.mean([r['mean'] for r in values])) if values else None,
                mean_pair_p90=float(np.mean([r['p90'] for r in values])) if values else None,
                worst_maximum=max((r['maximum'] for r in values),default=None),
                mean_complete_call_seconds=float(np.mean([r['complete_call_seconds'] for r in values])) if values else None)
    complete=[r for r in rows if 'analytic_minus_f2' in r]
    result=dict(independent_saved_output_check_passed=True,all50_successful=not failures,failures=failures,
        original_CSV_errors_recomputed=labels,minimum_corner_ratio=minimum,maximum_objective_component_errors=maxima,
        maximum_stage_repeated_objective_difference=repeated,application_call_seconds_sum=times,
        whole_batch_seconds=outer['elapsed_seconds'],summary=summary,
        paired_successful_cases=len(complete),analytic_strictly_better_counts={k:sum(r['analytic_minus_f2'][k]<0 for r in complete) for k in ('mean','p90','maximum','final_total')},
        median_paired_f2_over_analytic_call_ratio=statistics.median(r['f2_over_analytic_call_ratio'] for r in complete) if complete else None,
        scope='Actual saved geometry/literal full objective/original CSV endpoint audit; one paired observation per case, not a repeated time-to-accuracy curve.',rows=rows)
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--scores-only',action='store_true');args=parser.parse_args()
    if args.scores_only:score_agreement(args.directory,read(args.directory/'independent_check.json'))
    else:audit(args.directory)
