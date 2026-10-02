"""Independent saved-map/full-objective/CSV and recorded timing arithmetic audit.

No compiler, optimization or inference; CSVs only after all56 slots terminal.
GPU gradients/timers are not replayed: their source and recorded tests are checked.
"""
import argparse
import itertools
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

CASES=('miit_2_to_3','he_to_ki67','histo','rat_kidney')


def check_schedule(saved):
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
        records=saved['trace'][31*i:31*(i+1)]
        level=[17,33,65,129,257][i//2];raster=[32,64,128,256,512][i//2]
        direction=[[1.,0.],[0.,1.]][i%2]
        assert stage['level']==level and stage['control_side']==257 and stage['image_side']==raster and stage['direction']==direction
        assert stage['inner_steps']==30 and stage['physical_lr']==.004*16/(level-1)
        assert [r['step'] for r in records]==list(range(31))
        assert all(r['level']==level and r['control_side']==257 and r['image_side']==raster and r['direction']==direction for r in records)
        # Fresh requires_grad=False/True compiled evaluations need not be bitwise.
        # This arithmetic check is separate from unchanged production probe bounds.
        repeated=max(repeated,abs(stage['anchor_total']-records[0]['total']))
        assert abs(stage['accepted_total']-min([stage['anchor_total']]+[r['total'] for r in records]))<1e-12
        if i%2:repeated=max(repeated,abs(stage['anchor_total']-saved['stages'][i-1]['accepted_total']))
    assert repeated<1e-12
    return repeated


def audit(directory):
    report=read(directory/'benchmark.json')
    assert report['attempt_denominator']==56 and report['all_attempts_terminal']
    assert not report['annotations_read'] and not report['matcher_executed']
    assert tuple(c['name'] for c in report['cases'])==CASES
    assert all(len(c['runs'])==14 for c in report['cases'])
    assert all(r['status'] in ('complete','failed_no_fallback','invalid_or_incomplete_attempt','not_run_after_failure')
               for c in report['cases'] for r in c['runs'])
    maxima={};rows=[];labels=0;times=0.;probes=0.;minimum=1.;failures=[];repeated_total_error=0.
    probe_maxima={kind:{field:0. for field in ('values','full_vertex_gradient')}
                  for kind in ('same_backend','cross_backend')}
    oldrows={r['name']:r for r in read(ARCHIVE/'predictions.json')['rows']}
    for case in report['cases']:
        name=case['name'];folder=directory/name;old=oldrows[name]
        assert case['configuration']==old['configuration']
        assert [r['backend'] for r in case['runs']]==['eager','inductor']+['eager','inductor','inductor','eager']*3
        anchor,a,b,_=load_map(ARCHIVE/Path(old['output']).name)
        evidence=objective_data(name,case['configuration'],a,b)
        initial=literal_objective(IDENTITY,evidence)
        # These original CSV reads occur only after the all56 terminal assertions.
        layout,points,ids=original_landmark_case(name)
        unit=lambda xy,l:((xy+.5)*l['effective_original_to_canvas_scale_xy']+l['padding_xy'])/512
        query=unit(points['fixed'],layout['fixed']);truth=unit(points['moving'],layout['moving'])
        case_rows=[];arrays={};errors={}
        for run in case['runs']:
            if run['status']!='complete':
                failures.append(dict(case=name,name=run['name'],status=run['status'],error=run.get('error')))
                continue
            saved=read(folder/Path(run['report']).name);cfg=saved['configuration']
            changed={k for k in case['configuration'] if cfg[k]!=case['configuration'][k]}
            assert changed <= {'output','joint_prior_backend','match_p1_sampling'}
            assert cfg['joint_prior_backend']==run['backend']
            assert cfg['match_p1_sampling']==('existing' if run['backend']=='eager' else 'frozen')
            repeated_total_error=max(repeated_total_error,check_schedule(saved))
            for key in ('initial','final','gradient_steps','evaluations','objective_evaluations','failed_trials','saved_binary_certificate'):
                assert saved[key]==run[key]
            assert len(run['factory_calls'])==(run['backend']=='inductor')
            assert len(run['point_preparation_calls'])==(run['backend']=='inductor')
            v,aa,bb,qmin=load_map(folder/Path(run['output']).name,a,b)
            assert abs(qmin-run['export_actual_minimum_corner_ratio'])<1e-13
            minimum=min(minimum,qmin);arrays[run['name']]=v
            compare_objective(initial,run['initial'],maxima)
            compare_objective(literal_objective(v,evidence),run['final'],maxima)
            world=literal_p1(v@a.T+b,query)
            error=np.linalg.norm((world-truth)*512,axis=-1);errors[run['name']]=error;labels+=len(ids)
            memory=run['whole_call_memory']
            assert memory['peak_reserved_bytes']>=memory['peak_allocated_bytes']>=run['peak_allocated_bytes']
            assert memory['captured_segments']>=2
            assert run['complete_call_seconds']>=run['optimizer_observer_seconds']>0
            times+=run['complete_call_seconds']
            case_rows.append(dict(name=run['name'],backend=run['backend'],phase=run['phase'],
                minimum_corner_ratio=qmin,mean=float(error.mean()),p90=float(np.percentile(error,90)),
                maximum=float(error.max()),per_label=dict(zip(ids,error.tolist())),
                whole_call_seconds=run['complete_call_seconds'],whole_call_memory=memory))
        if len(case_rows)==14:
            assert len(case['comparisons'])==91
            byname={r['name']:r for r in case['runs']}
            differences=[]
            for comparison in case['comparisons']:
                left,right=comparison['left'],comparison['right'];delta=arrays[left]-arrays[right]
                assert comparison['affine_boundary_interpolation_equal'] and comparison['counters_equal']
                assert comparison['map_max_abs_delta']==float(abs(delta).max())
                assert comparison['map_rms_delta']==float(np.sqrt(np.mean(delta*delta)))
                assert comparison['map_bitwise_equal']==bool(np.array_equal(arrays[left],arrays[right]))
                assert comparison['final_total_delta']==byname[right]['final']['total']-byname[left]['final']['total']
                differences.append(dict(left=left,right=right,same_backend=comparison['same_backend'],
                    both_warm=comparison['both_warm'],map_max_abs_delta=comparison['map_max_abs_delta'],
                    maximum_landmark_TRE_delta=float(abs(errors[left]-errors[right]).max()),
                    mean_TRE_delta=float(errors[right].mean()-errors[left].mean())))
            warm=[r for r in case['runs'] if r['phase']=='warm_ABBA']
            medians={arm:statistics.median(r['complete_call_seconds'] for r in warm if r['backend']==arm)
                     for arm in ('eager','inductor')}
            summary=case['warm_summary'];assert summary['complete_call_medians']==medians
            assert summary['baseline_over_candidate_median']==medians['eager']/medians['inductor']
            groups=[]
            for group in range(3):
                values={arm:statistics.mean(r['complete_call_seconds'] for r in warm
                            if r['group']==group and r['backend']==arm) for arm in medians}
                ratio=values['eager']/values['inductor'];groups.append(ratio)
                assert summary['paired_ABBA_groups'][group]['baseline_over_candidate']==ratio
            assert summary['median_paired_ABBA_ratio']==statistics.median(groups)
            equivalence=read(folder/'equivalence.json')
            assert equivalence['passed'] and equivalence['compiler_backend']=='inductor' and equivalence['production']
            assert equivalence['order']==['eager','compiled','compiled','eager']
            assert equivalence['tolerances']==dict(values=dict(rtol=1e-8,atol=1e-10),full_vertex_gradient=dict(rtol=1e-5,atol=1e-8))
            assert [(r['state'],r['image_side']) for r in equivalence['rows']]==[(s,n) for s in ('identity','saved_deformed') for n in (32,64,128,256,512)]
            for item in equivalence['rows']:
                values=np.array(item['values']);assert np.isfinite(values).all()
                assert item['passed'] and len(item['comparisons'])==6
                for key,comparison in item['comparisons'].items():
                    i,j=map(int,key.split('_vs_'));assert np.allclose(values[i],values[j],rtol=1e-8,atol=1e-10)
                    assert comparison['passed'] and comparison['values']['passed'] and comparison['full_vertex_gradient']['passed']
                    assert comparison['full_vertex_gradient']['finite']
                    kind='same_backend' if (i,j) in ((0,3),(1,2)) else 'cross_backend'
                    for field in ('values','full_vertex_gradient'):
                        probe_maxima[kind][field]=max(probe_maxima[kind][field],comparison[field]['maximum_absolute_error'])
            probes+=case['equivalence']['wrapper_seconds']
        else:
            differences=[];summary=None
        rows.append(dict(name=name,landmark_count=len(ids),runs=case_rows,
            independently_recomputed_pair_differences=differences,warm_summary=summary))
        print(json.dumps(dict(case=name,saved_maps_checked=len(case_rows),maximum_objective_error=maxima.get('total'))),flush=True)
    assert report['whole_workload_seconds']>=times+probes+report['cache_setup_seconds']
    result=dict(independent_saved_output_check_passed=True,all56_successful=not failures,failures=failures,
        minimum_corner_ratio=minimum,maximum_objective_component_errors=maxima,
        maximum_stage_repeated_objective_difference=repeated_total_error,
        maximum_recorded_probe_absolute_errors=probe_maxima,
        original_CSV_errors_recomputed=labels,application_call_seconds_sum=times,
        equivalence_wrapper_seconds_sum=probes,whole_workload_seconds=report['whole_workload_seconds'],
        scope='Actual saved geometry, independent literal full objectives and CSVs; recorded timers and GPU VJP comparisons audited, not independently replayed.',rows=rows)
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    audit(parser.parse_args().directory)
