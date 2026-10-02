"""Independent saved timed-suffix review. No optimization; labels follow all50.

Uses literal NumPy P1/descriptor/ARAP and original CSV oracles from prior checks.
"""
import argparse
import copy
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_joint_pose_probe_20261002 import pixels,corners,bilinear,descriptor,vector_p1,numpy_priors
from independent_stain_proxy_probe_20261002 import literal_p1,original_landmark_case
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
BASE=ROOT/'outputs/coordinated_instance_registration'
DATA=Path('D:/QC_optimization_data/digital_topology_wsi')
ARCHIVE=BASE/'match_fusion_all50_t23/fusion'
ARMS=('adam','data_metric')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
GRID=pixels(512)
YY,XX=np.meshgrid(np.arange(257)/256,np.arange(257)/256,indexing='ij')
IDENTITY=np.stack((XX,YY),-1)

def exactdet(a):return Fraction(float(a[0,0]))*Fraction(float(a[1,1]))-Fraction(float(a[0,1]))*Fraction(float(a[1,0]))

def load_map(path,a=None,b=None):
    with np.load(path,allow_pickle=False) as z:
        v=z['vertices'];reference=z['boundary_reference'];matrix=z['post_affine_matrix'];offset=z['post_affine_offset']
        assert str(z['interpolation'])=='p1_ac'
    assert v.shape==(1,257,257,2) and v.dtype==np.float64 and np.array_equal(reference,IDENTITY[None])
    assert np.array_equal(v[:,[0,-1]],reference[:,[0,-1]]) and np.array_equal(v[:,:,[0,-1]],reference[:,:,[0,-1]])
    assert exactdet(matrix)>0 and certify_q1_binary_map(path)['valid']
    if a is not None:assert np.array_equal(matrix,a) and np.array_equal(offset,b)
    minimum=float(corners(v[0]).min());assert minimum>.001
    return v[0],matrix.astype(float),offset.astype(float),minimum

def objective_data(name,cfg,a,b):
    folder=BASE/'miit_three_rotations_t153' if name.startswith('miit_') else DATA/('birl_anhir_dev/canvas' if name in ('histo','rat_kidney') else 'lung_lesion3_eval/canvas')
    images={r:1-np.asarray(Image.open(folder/Path(cfg[r]).name).convert('L'),dtype=np.float32)/255 for r in ('fixed','moving')}
    mask=images['fixed']>.04
    aligned=bilinear(images['moving'],((2*(GRID@a.T+b)-1).astype(np.float32).astype(float)+1)/2)[...,0]
    table=read(BASE/'match_fusion_all50_t23/fused_tables'/(name+'.json'))
    q=np.array(table['source_points_unit']);p=np.array(table['target_points_unit']);confidence=np.array(table['confidence'])
    eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1);weight=confidence*eligible;weight/=weight.sum()
    return dict(a=a,b=b,fixed_feature=descriptor(images['fixed']),moving_feature=descriptor(aligned),mask=mask,q=q,p=p,weight=weight)

def literal_objective(v,data):
    a,b=data['a'],data['b'];mask=data['mask'];mapped=vector_p1(v,GRID);world=mapped@a.T+b
    sampled=bilinear(data['moving_feature'],((2*mapped-1).astype(np.float32).astype(float)+1)/2)
    image=float((np.abs(data['fixed_feature']-sampled).mean(-1)*mask).sum()/mask.sum())
    excess=np.maximum(-world,0)+np.maximum(world-1,0);oob=float(((excess**2).sum(-1)*mask).sum()/mask.sum())
    outside=float((((world<0)|(world>1)).any(-1)*mask).sum()/mask.sum())
    z=((vector_p1(v,data['q'])-data['p'])@a.T)*64;point=float(((np.sqrt(1+(z*z).sum(-1))-1)*data['weight']).sum())
    strain,shape=numpy_priors(v)
    return dict(total=image+3*strain+1e-4*shape+.2*point+oob,image=image,strain=strain,shape=shape,match=point,oob=oob,outside_fraction=outside)

def compare_objective(expected,reported,maxima):
    for key,value in expected.items():
        error=abs(value-reported[key]);maxima[key]=max(maxima.get(key,0),error)
        tolerance=3e-7 if key in ('image','total') else (5e-8 if key=='outside_fraction' else 2e-10)
        assert error<tolerance,(key,error,value,reported[key])

def check_timed_trace(report,arm):
    assert report['prefix_gradient_steps']==240 and report['gradient_steps']==240+report['suffix_gradient_steps']
    assert report['query_count']==512**2 and report['control_vertices']==257**2 and not report['landmarks_used']
    stages=report['stages'];assert len(stages)==2 and report['terminal_budget']['stage_records']==stages
    assert report['terminal_budget']['protocol']=='timed_final' and report['terminal_budget']['seconds_per_axis']==2. and report['terminal_budget']['arm']==arm
    sums={key:sum(s['counts'][key] for s in stages) for key in stages[0]['counts']}
    assert sums==report['suffix_counts'] and sums['gradient_steps']==report['suffix_gradient_steps']
    assert report['suffix_objective_evaluations']==3+sums['objective_evaluations']
    assert report['objective_evaluations']==report['prefix_objective_evaluations']+report['suffix_objective_evaluations']
    assert report['failed_trials']==sums['failed_trials'] and report['evaluations']==248+sums['trial_evaluations']
    assert report['prefix_objective_evaluations']==266
    assert abs(report['suffix_start_total']-report['initial']['total'])<1e-15
    selected_values=[report['prefix_best_total']]+[s['accepted_full_total'] for s in stages]
    minimum=min(selected_values);assert abs(report['final']['total']-minimum)<1e-12
    selected=report['selected_stage'];assert selected in (None,0,1)
    assert abs((report['prefix_best_total'] if selected is None else stages[selected]['accepted_full_total'])-minimum)<1e-12
    statuses=Counter();pcg_reasons=Counter();maximum_recursive=0.;all_trials=0;accepted_trials=0
    for index,stage in enumerate(stages):
        assert stage['level']==257 and stage['image_side']==512 and stage['direction']==[[1.,0.],[0.,1.]][index]
        assert stage['budget_elapsed']==(stage['elapsed_seconds']>=2.) and abs(stage['time_overrun_seconds']-max(0,stage['elapsed_seconds']-2.))<1e-12
        if stage['stop_reason']=='time_budget':assert stage['budget_elapsed']
        statuses[stage['stop_reason']]+=1
        assert abs(stage['anchor_total']-(report['initial']['total'] if index==0 else stages[0]['accepted_total']))<1e-12
        assert stage['accepted_total']==stage['accepted_full_total'] and stage['accepted_total']<=stage['anchor_total']
        records=[r for r in report['trace'] if r['stage']==index];counts=stage['counts'];best=stage['anchor_total']
        assert sum(bool(r.get('accepted')) for r in records)==counts['accepted_steps']
        if arm=='adam':
            assert stage['physical_lr']==.00025 and 'diagnostic only' in stage['geometry_acceptance']
            assert all(counts[k]==0 for k in ('metric_refreshes','descriptor_forwards','descriptor_vjps','pcg_iterations','pcg_matvecs','preconditioner_calls','backtracks'))
            if stage['stop_reason']=='time_budget':assert len(records)==counts['gradient_steps']==counts['trial_evaluations'] and counts['objective_evaluations']==2+counts['gradient_steps']
            for record in records:
                if 'error' in record:continue
                assert record['accepted']==(record['total']<best)
                if record['accepted']:best=record['total']
            assert abs(best-stage['accepted_total'])<1e-12
        else:
            assert stage['physical_lr'] is None and stage['minimum_contracted_slack']>0 and 'strict fixed contracted' in stage['geometry_acceptance']
            assert counts['descriptor_vjps']<=8*counts['metric_refreshes'] and counts['descriptor_forwards']<=counts['metric_refreshes']
            successful_metrics=[r for r in records if 'pcg' in r]
            assert sum(r['pcg']['iterations'] for r in successful_metrics)==counts['pcg_iterations']
            assert sum(r['pcg']['matvecs'] for r in successful_metrics)==counts['pcg_matvecs']
            assert sum(r['pcg']['preconditioner_calls'] for r in successful_metrics)==counts['preconditioner_calls']
            assert sum(max(0,len(r.get('trials',[]))-1) for r in records)==counts['backtracks']
            if stage['stop_reason']=='time_budget':
                assert len(records)==counts['gradient_steps']==counts['metric_refreshes']==counts['descriptor_forwards']
                assert counts['descriptor_vjps']==8*counts['metric_refreshes']
            for record in records:
                if 'pcg' not in record:continue
                assert abs(record['total']-best)<1e-12
                solve=record['pcg'];pcg_reasons[solve['reason']]+=1
                assert solve['iterations']<=20 and solve['residual_kind']=='recursive Euclidean norm divided by initial norm'
                if solve['converged']:
                    assert solve['reason'] in ('relative_residual','zero_rhs') and 0<=solve['relative_residual']<=.1
                if solve['reason']=='iteration_cap':assert solve['iterations']==20 and not solve['converged']
                if solve['relative_residual'] is not None:maximum_recursive=max(maximum_recursive,solve['relative_residual'])
                assert np.isfinite(record['metric_gamma']) and record['metric_gamma']>=0
                trials=record.get('trials',[]);assert len(trials)<=13;all_trials+=len(trials)
                if trials:
                    assert record['directional_derivative']<0 and record['initial_alpha']>0
                    expected_alpha=1. if record['alpha_max'] is None else min(1.,.99*record['alpha_max'])
                    assert record['initial_alpha']==expected_alpha
                for j,trial in enumerate(trials):
                    assert trial['alpha']==record['initial_alpha']*.5**j
                    rhs=record['total']+1e-4*trial['alpha']*record['directional_derivative'];assert rhs==trial['armijo_rhs']
                    if trial['accepted']:
                        assert j==len(trials)-1 and trial['geometry_valid'] and trial['total']<record['total'] and trial['total']<=rhs
                        best=trial['total'];accepted_trials+=1
                assert bool(record.get('accepted'))==any(t['accepted'] for t in trials)
            assert abs(best-stage['accepted_total'])<1e-12
            finite_trial_calls=sum(t['total'] is not None for r in records for t in r.get('trials',[]))
            if counts['failed_trials']==0:
                assert finite_trial_calls==counts['trial_evaluations'] and counts['objective_evaluations']==1+counts['gradient_steps']+counts['trial_evaluations']
    return dict(stops=dict(statuses),pcg_reasons=dict(pcg_reasons),maximum_recursive_relative_residual=maximum_recursive,
        metric_trial_attempts=all_trials,accepted_metric_trials=accepted_trials)

def check_case(directory,name,prefix_record,arm_rows,*,smoke=False):
    common=directory if smoke else directory/'common'
    prefix=read(common/Path(prefix_record['prefix_report']).name)
    assert prefix['gradient_steps']==240 and prefix['objective_evaluations']==266 and prefix['evaluations']==248
    assert prefix['failed_trials']==0 and len(prefix['stages'])==8 and len(prefix['trace'])==248 and not prefix['landmarks_used']
    assert prefix['partial_prefix']['accepted_stages']==8 and not prefix['partial_prefix']['full_schedule_completed']
    assert prefix['final']['total']==min([prefix['initial']['total']]+[s['accepted_full_total'] for s in prefix['stages']])
    old=read(ARCHIVE/(name+'_analytic.json'))
    assert {k for k in old['configuration'] if old['configuration'][k]!=prefix['configuration'][k]}=={'output'}
    original,aa,bb,_=load_map(ARCHIVE/(name+'_analytic.npz'))
    raw,a,b,raw_min=load_map(common/Path(prefix_record['prefix_start']).name,aa,bb)
    best,_,_,best_min=load_map(common/Path(prefix_record['prefix_best']).name,a,b)
    data=objective_data(name,prefix['configuration'],a,b);maxima={}
    raw_E=literal_objective(raw,data);best_E=raw_E if np.array_equal(raw,best) else literal_objective(best,data)
    compare_objective(best_E,prefix['final'],maxima)
    assert abs(raw_E['total']-prefix['stages'][-1]['accepted_full_total'])<3e-7
    results={};first_report=None
    for arm,row in arm_rows.items():
        folder=directory if smoke else directory/arm;report=read(folder/row['report'])
        if first_report is None:first_report=report
        else:
            assert report['prefix_start']==first_report['prefix_start'] and report['prefix_best']==first_report['prefix_best']
            assert report['initial']==first_report['initial'] and report['prefix_best_total']==first_report['prefix_best_total']
        assert Path(report['prefix_start']).name==Path(prefix_record['prefix_start']).name and Path(report['prefix_best']).name==Path(prefix_record['prefix_best']).name
        assert report['image_match_evidence']==old['image_match_evidence']
        assert report['image_preprocessing']['original_moving_features_no_affine_prewarp'] is False
        assert report['image_preprocessing']['moving_descriptor_frame']=='shared_affine'
        assert {k for k in old['configuration'] if old['configuration'][k]!=report['configuration'][k]}=={'output'}
        check=check_timed_trace(report,arm)
        v,_,_,minimum=load_map(folder/row['output'],a,b)
        compare_objective(raw_E,report['initial'],maxima)
        assert abs(best_E['total']-report['prefix_best_total'])<3e-7
        final_E=literal_objective(v,data);compare_objective(final_E,report['final'],maxima)
        floor_checked=[];selected=report['selected_stage']
        if selected is None:assert np.array_equal(v,best)
        else:
            # Fixed-axis composition allows exact x-stage reconstruction if the
            # selected output is stage1: y updates never change its x coordinates.
            xmap=raw.copy();xmap[...,0]=v[...,0]
            if selected==0:assert np.array_equal(xmap,v)
            x_E=final_E if selected==0 else literal_objective(xmap,data)
            assert abs(x_E['total']-report['stages'][0]['accepted_total'])<3e-7
            xq=corners(xmap);rawq=corners(raw);slack=xq-(.001+.05*(rawq-.001))
            assert xq.min()>.001
            if arm=='data_metric':assert slack.min()>0
            assert abs(float(slack.min())-report['stages'][0]['minimum_contracted_slack'])<5e-13
            floor_checked.append(0)
            if selected==1:
                yslack=corners(v)-(.001+.05*(xq-.001))
                if arm=='data_metric':assert yslack.min()>0
                assert abs(float(yslack.min())-report['stages'][1]['minimum_contracted_slack'])<5e-13
                assert abs(report['stages'][1]['anchor_total']-x_E['total'])<3e-7
                floor_checked.append(1)
        results[arm]=dict(name=name,minimum_corner_ratio=minimum,selected_stage=selected,actual_stage_floors_reconstructed=floor_checked,
            initial=report['initial'],final=report['final'],prefix_best_total=report['prefix_best_total'],
            suffix_gradient_steps=report['suffix_gradient_steps'],suffix_counts=report['suffix_counts'],
            stages=report['stages'],**check)
    return results,maxima

def label_check(directory,name,row,arm):
    # Called ONLY after main has checked complete outer+both25case manifests.
    scores={r['name']:r for kind in ('miit','existing') for r in read(directory/arm/(kind+'_scores.json'))['rows']}
    score=scores[name];assert score['status']=='ok' if 'status' in score else True
    layout,points,ids=original_landmark_case(name);assert ids==score['available_pair_labels']
    v,a,b,_=load_map(directory/arm/row['output'])
    unit=lambda xy,l:((xy+.5)*l['effective_original_to_canvas_scale_xy']+l['padding_xy'])/512
    predicted=literal_p1(v@a.T+b,unit(points['fixed'],layout['fixed']))
    expected={'canvas_pixels':np.linalg.norm((predicted-unit(points['moving'],layout['moving']))*512,axis=1),
        'native_moving_pixels':np.linalg.norm((predicted*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5-points['moving'],axis=1)}
    metrics=(score['methods']['analytic']['metrics'] if name.startswith('miit_') else score['metrics']);errors={}
    for key,values in expected.items():
        errors[key]=max(abs(value-metrics[key]['per_label'][label]) for label,value in zip(ids,values))
        assert errors[key]<1e-9 and abs(values.mean()-metrics[key]['mean'])<1e-10 and abs(np.percentile(values,90)-metrics[key]['p90'])<1e-10
    return dict(landmarks=len(ids),maximum_errors=errors,metrics={k:metrics['canvas_pixels'][k] for k in ('mean','p90','maximum')})

def legacy_default_checks():
    """Read-only forward/VJP check of the actual omitted-field configurations."""
    import torch
    from tools.coordinated_data_metric_application import _build_evidence
    torch.set_num_threads(1);results=[]
    for name in ('histo','rat_kidney'):
        cfg=argparse.Namespace(**read(ARCHIVE/(name+'_analytic.json'))['configuration']);assert not hasattr(cfg,'mind_order')
        cfg.fixed=DATA/'birl_anhir_dev/canvas'/Path(cfg.fixed).name;cfg.moving=DATA/'birl_anhir_dev/canvas'/Path(cfg.moving).name
        cfg.matches=BASE/'match_fusion_all50_t23/fused_tables'/(name+'.json')
        with np.load(ARCHIVE/(name+'_analytic.npz')) as z:a=z['post_affine_matrix'];b=z['post_affine_offset']
        evidence,_,_=_build_evidence(cfg,a,b,torch.device('cpu'))
        explicit=copy.deepcopy(cfg);explicit.mind_order='transport';reference,_,_=_build_evidence(explicit,a,b,torch.device('cpu'))
        path=BASE/'data_metric_all50_t26/common'/(name+'_prefix_best_suffix_start.npz')
        with np.load(path) as z:vertices=torch.tensor(z['vertices'],requires_grad=True)
        actual=evidence(vertices)[0];expected=reference(vertices)[0]
        actual_gradient=torch.autograd.grad(actual,vertices)[0];expected_gradient=torch.autograd.grad(expected,vertices)[0]
        assert torch.equal(actual,expected) and torch.equal(actual_gradient,expected_gradient) and not hasattr(cfg,'mind_order')
        results.append(dict(name=name,actual257_forward_and_vertex_VJP_equal_explicit_transport_bitwise=True,configuration_not_mutated=True))
    return results

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--geometry-only',action='store_true');parser.add_argument('--legacy-defaults',action='store_true')
    parser.add_argument('--directory',type=Path,default=BASE/'data_metric_all50_t26');args=parser.parse_args()
    if args.legacy_defaults:print(json.dumps(legacy_default_checks(),indent=2));return
    if args.smoke:
        directory=BASE/'data_metric_smoke_t26';smoke=read(directory/'smoke.json')
        result,errors=check_case(directory,smoke['name'],smoke['prefix'],{a:dict(report=a+'.json',output=a+'.npz') for a in ARMS},smoke=True)
        print(json.dumps(dict(scope='label-free smoke only',errors=errors,rows=result),indent=2));return
    directory=args.directory;outer=read(directory/'predictions.json');manifests={a:read(directory/a/'predictions.json') for a in ARMS}
    assert outer['prediction_complete'] and not outer['annotations_read'] and outer['attempt_denominator']==50 and outer['prefix_denominator']==25
    assert len(outer['prefixes'])==25 and all(p['status']=='ok' for p in outer['prefixes'])
    failures=[]
    for a,m in manifests.items():
        assert m['prediction_complete'] and m['all50_terminal'] and not m['annotations_read'] and len(m['rows'])==25
        assert all(r['status'] in ('ok','failed') for r in m['rows'])
        failures.extend(dict(name=r['name'],arm=a,error=r.get('error')) for r in m['rows'] if r['status']=='failed')
        assert all((directory/a/r[k]).exists() for r in m['rows'] if r['status']=='ok' for k in ('report','output'))
    if failures and not args.geometry_only:raise ValueError('partial failed batch: no labels read; use geometry-only or the completed corrected rerun')
    rows={a:{r['name']:r for r in m['rows']} for a,m in manifests.items()};checked=[];errors={};labels=0;score_errors={}
    for index,prefix in enumerate(outer['prefixes']):
        name=prefix['name'];arm_rows={a:rows[a][name] for a in ARMS}
        successful={a:r for a,r in arm_rows.items() if r['status']=='ok'}
        if not successful:continue
        order=ARMS if index%2==0 else tuple(reversed(ARMS))
        assert all(successful[a]['arm_position_in_pair']==order.index(a) for a in successful)
        assert all(r['shared_prefix']==prefix for r in successful.values())
        result,case_errors=check_case(directory,name,prefix,successful)
        for key,error in case_errors.items():errors[key]=max(errors.get(key,0),error)
        if not args.geometry_only:
            for a,row in arm_rows.items():
                scoring=label_check(directory,name,row,a);result[a]['scores']=scoring['metrics'];labels+=scoring['landmarks']
                for key,error in scoring['maximum_errors'].items():score_errors[key]=max(score_errors.get(key,0),error)
        checked.append(dict(name=name,arms=result));print(json.dumps(dict(checked=name,maximum_objective_error=errors.get('total'))),flush=True)
    if not args.geometry_only:assert labels==4148
    summary={a:dict(minimum_corner_ratio=min(r['arms'][a]['minimum_corner_ratio'] for r in checked),
        suffix_gradient_range=[min(r['arms'][a]['suffix_gradient_steps'] for r in checked),max(r['arms'][a]['suffix_gradient_steps'] for r in checked)],
        total_suffix_counts={key:sum(r['arms'][a]['suffix_counts'][key] for r in checked) for key in checked[0]['arms'][a]['suffix_counts']},
        stop_reasons=dict(sum((Counter(r['arms'][a]['stops']) for r in checked),Counter())),
        pcg_reasons=dict(sum((Counter(r['arms'][a]['pcg_reasons']) for r in checked),Counter())),
        maximum_recursive_relative_residual=max(r['arms'][a]['maximum_recursive_relative_residual'] for r in checked),
        all_selected_final_axis=all(r['arms'][a]['selected_stage']==1 for r in checked),
        actual_stage_floors_reconstructed=sum(len(r['arms'][a]['actual_stage_floors_reconstructed']) for r in checked)) for a in ARMS}
    report=dict(all50_attempts_terminal=True,saved_successful_outputs_checked=sum(len(r['arms']) for r in checked),all50_saved_results_checked=not failures,
        failed_attempts=failures,annotation_scores_checked=not args.geometry_only,total_landmark_errors=labels,
        maximum_full_objective_component_errors=errors,maximum_label_errors=score_errors,summary=summary,rows=checked)
    (directory/('independent_geometry_check.json' if args.geometry_only else 'independent_check.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
