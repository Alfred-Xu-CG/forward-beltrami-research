"""Independent all50 absolute-map, refreshed-observation and original-CSV check."""
from pathlib import Path
from collections import Counter
import copy
import json
import sys
import numpy as np
import torch
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_data_metric_postrun_20261002 import load_map,objective_data,literal_objective,compare_objective,label_check,read,BASE,ARCHIVE
from independent_joint_pose_probe_20261002 import vector_p1
DIRECTORY=BASE/'refreshed_match_all50_t27'
ARMS=('frozen_suffix','refreshed_suffix')

def relocate(path):
    text=str(path).replace('\\','/');tail=text.split('/results/',1)[1] if '/results/' in text else text.removeprefix('results/')
    return BASE/tail

def table_arrays(table,a,b):
    q=np.array(table['source_points_unit']);p=np.array(table['target_points_unit']);c=np.array(table['confidence'])
    assert q.shape==p.shape==(len(c),2) and all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in (q,p,c))
    eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1);weight=c*eligible
    assert (weight>0).sum()>=8 and weight.sum()>0
    return q,p,c,eligible,weight/weight.sum()

def tables_check(name,incumbent,a,b,extraction):
    old=read(BASE/'match_fusion_all50_t23/fused_tables'/(name+'.json'))
    sg=read(relocate(old['original_sg_table']));new=read(DIRECTORY/'refreshed_tables'/(name+'.json'));fused=read(DIRECTORY/'fused_tables'/(name+'.json'))
    assert new['status']=='ok' and new['conditioning_rounds']==1 and not new['incumbent_disagreement_filter_used']
    assert new['targets_manual_landmarks_or_dense_teacher_loaded'] is False and not new['affine_estimated_or_changed'] and not new['global_geometric_ransac_used'] and not new['tissue_support_filter_used']
    for table in (old,sg,new,fused):
        assert np.array_equal(table['post_affine_matrix'],a) and np.array_equal(table['post_affine_offset'],b)
    q,p,c,eligible,weights=table_arrays(new,a,b);s=np.array(new['warped_target_points_unit'])
    assert s.shape==q.shape and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
    literal=vector_p1(incumbent,s);conversion_error=float(abs(literal-p).max());assert conversion_error<3e-16
    assert len(q)==new['raw_matches']==new['network_matches']-new['discarded_out_of_unit_domain']
    assert new['eligible_matches']==int((eligible&(c>0)).sum()) and new['static_outside_original_moving_matches']==int((~eligible).sum())
    assert new['zero_confidence_matches']==int((c==0).sum()) and abs(new['confidence_mass']-c.sum())<1e-10 and abs(new['eligible_confidence_mass']-c[eligible].sum())<1e-10
    assert new['occupied_quadrants_4x4']==len(set(map(tuple,np.minimum((q[eligible&(c>0)]*4).astype(int),3))))
    for points,prefix in ((q,'source'),(p,'target')):
        counts=np.unique(points,axis=0,return_counts=True)[1]
        assert new['unique_'+prefix+'_coordinates']==len(counts) and new['maximum_'+prefix+'_multiplicity']==max(counts)
    for key,value in extraction['extraction'].items():assert new[key]==value
    assert Path(new['incumbent']['path']).name==name+'_analytic.npz'
    normalized=[];offset=0
    for index,table in enumerate((sg,new)):
        qq,pp,cc,ee,ww=table_arrays(table,a,b);mass=float((cc*ee).sum());normalized.extend((cc/mass).tolist())
        assert fused['source_points_unit'][offset:offset+len(qq)]==table['source_points_unit'] and fused['target_points_unit'][offset:offset+len(qq)]==table['target_points_unit']
        if index==0:
            # The entire SG prefix, including ineligible rows, is bitwise the old
            # normalized SG prefix; this is not SG renormalized as a whole fusion.
            for key in ('source_points_unit','target_points_unit','confidence'):assert fused[key][:len(qq)]==old[key][:len(qq)]
        diagnostic=fused['composition'][index]
        assert diagnostic['original_rows']==len(qq) and abs(diagnostic['original_eligible_confidence_mass']-mass)<1e-10
        assert diagnostic['original_positive_eligible_rows']==int(((cc>0)&ee).sum()) and diagnostic['static_ineligible_rows']==int((~ee).sum())
        offset+=len(qq)
    assert fused['confidence']==normalized and len(normalized)==len(fused['source_points_unit'])
    fq,fp,fc,fe,fw=table_arrays(fused,a,b)
    assert abs((fc*fe).sum()-2)<1e-14 and fused['match_weight']==.2 and fused['rows_removed']==fused['rows_clipped']==0
    assert extraction['point_loader']['raw_matches']==len(fq) and abs(extraction['point_loader']['confidence_denominator']-2)<1e-14
    old_s_eligible=((s@a.T+b>=0)&(s@a.T+b<=1)).all(-1)
    return dict(q=fq,p=fp,weight=fw),dict(refreshed_matches=len(q),eligible_matches=int(eligible.sum()),conversion_error=conversion_error,
        changed_world_eligibility_after_P1=int((eligible!=old_s_eligible).sum()),unchanged_SG_prefix=True,each_source_effective_coefficient=.1)

def check_schedule(report):
    assert report['gradient_steps']==300 and report['failed_trials']==0 and report['evaluations']==310 and report['objective_evaluations']==332
    assert len(report['stages'])==10 and len(report['trace'])==310 and not report['failures'] and not report['landmarks_used']
    assert report['output_selection']=='best_full' and report['inner_steps_by_level']==[30]*5
    assert report['control_vertices']==257**2 and report['query_count']==512**2 and report['image_levels']==[32,64,128,256,512]
    selected=report['selected_stage'];values=[report['initial']['total']]+[s['accepted_full_total'] for s in report['stages']]
    assert report['final']['total']==min(values) and report['final']['total']==(values[0] if selected is None else values[selected+1])
    for i,stage in enumerate(report['stages']):
        records=report['trace'][31*i:31*(i+1)];level=[17,33,65,129,257][i//2];resolution=[32,64,128,256,512][i//2];direction=[[1.,0.],[0.,1.]][i%2]
        assert stage['level']==level and stage['control_side']==257 and stage['image_side']==resolution and stage['direction']==direction
        assert stage['inner_steps']==30 and stage['physical_lr']==.004*16/(level-1)
        assert [r['step'] for r in records]==list(range(31)) and stage['anchor_total']==records[0]['total']
        assert stage['accepted_total']==min(r['total'] for r in records)
        assert all(r['level']==level and r['direction']==direction and r['control_side']==257 and r['image_side']==resolution for r in records)
        if i%2:assert stage['anchor_total']==report['stages'][i-1]['accepted_total']
    return selected

def main():
    torch.set_num_threads(2);outer=read(DIRECTORY/'predictions.json');manifests={a:read(DIRECTORY/a/'predictions.json') for a in ARMS}
    assert outer['prediction_complete'] and outer['extraction_complete'] and outer['attempt_denominator']==50 and not outer['annotations_read']
    assert len(outer['extractions'])==25 and all(r['status']=='ok' for r in outer['extractions']) and outer['model_released_before_optimization']
    oldmodel=read(BASE/'matchanything_all25_t22/predictions.json')['model_setup'];newmodel=outer['model_setup']
    for key in ('source_revision','model'):assert oldmodel[key]==newmodel[key]
    for arm,m in manifests.items():assert m['prediction_complete'] and m['all50_terminal'] and not m['annotations_read'] and len(m['rows'])==25 and all(r['status']=='ok' for r in m['rows'])
    names=[r['name'] for r in outer['extractions']];assert all([r['name'] for r in m['rows']]==names for m in manifests.values())
    checked=[];errors={};score_errors={};labels=0;table_errors=0.
    for i,name in enumerate(names):
        old=read(ARCHIVE/(name+'_analytic.json'));incoming,a,b,minimum=load_map(ARCHIVE/(name+'_analytic.npz'))
        evidence=objective_data(name,old['configuration'],a,b)
        refreshed,table_info=tables_check(name,incoming,a,b,outer['extractions'][i]);table_errors=max(table_errors,table_info['conversion_error'])
        original_E=literal_objective(incoming,evidence);compare_objective(original_E,old['final'],errors)
        results={};starts=[]
        for arm in ARMS:
            row=manifests[arm]['rows'][i];report=read(DIRECTORY/arm/row['report']);starts.append(report['initial_map']['path'])
            assert row['incumbent']==report['initial_map']['path'] and Path(row['incumbent']).name==name+'_analytic.npz'
            assert abs(report['initial_map']['minimum_corner_ratio']-minimum)<1e-13 and report['initial_map']['saved_binary_certificate']['valid']
            order=ARMS if i%2==0 else tuple(reversed(ARMS));assert row['arm_position_in_pair']==order.index(arm)
            assert row['new_suffix_gradient_steps']==300
            changed={k for k in old['configuration'] if old['configuration'][k]!=report['configuration'][k]}
            assert changed==({'output'} if arm=='frozen_suffix' else {'output','matches'})
            for key in ('image_preprocessing','mind_frame_by_resolution','mind_order','mind_frame','strain_model','f2_floor_safety_fraction','image_objective','image_weight','mask','geometry_dtype','evidence_dtype','interpolation'):
                if key in old:assert report[key]==old[key]
            selected=check_schedule(report);v,_,_,qmin=load_map(DIRECTORY/arm/row['output'],a,b)
            assert abs(qmin-row['actual_minimum_corner_ratio'])<1e-13
            if selected is None:assert np.array_equal(v,incoming)
            own=copy.copy(evidence)
            if arm=='refreshed_suffix':own.update(refreshed)
            initial=original_E if arm=='frozen_suffix' else literal_objective(incoming,own)
            final=initial if selected is None else literal_objective(v,own)
            compare_objective(initial,report['initial'],errors);compare_objective(final,report['final'],errors)
            for key in ('image','strain','shape','oob','outside_fraction'):assert report['initial'][key]==old['final'][key]
            if arm=='frozen_suffix':assert report['initial']==old['final']
            scored=label_check(DIRECTORY,name,row,arm);labels+=scored['landmarks']
            for key,error in scored['maximum_errors'].items():score_errors[key]=max(score_errors.get(key,0),error)
            results[arm]=dict(selected_stage=selected,initial=report['initial'],final=report['final'],minimum_corner_ratio=qmin,scores=scored['metrics'])
        assert starts[0]==starts[1]
        checked.append(dict(name=name,table=table_info,arms=results));print(json.dumps(dict(checked=name,maximum_objective_error=errors['total'],maximum_conversion_error=table_errors)),flush=True)
    assert labels==4148
    result=dict(all50_maps_same_absolute_incumbents_budgets_objectives_and_CSV_scores_checked=True,all25_refreshed_target_conversions_and_unchanged_SG_checked=True,
        total_landmark_errors=labels,maximum_objective_component_errors=errors,maximum_original_CSV_score_errors=score_errors,maximum_refreshed_P1_target_error=table_errors,
        total_refreshed_matches=sum(r['table']['refreshed_matches'] for r in checked),total_post_P1_eligibility_differences=sum(r['table']['changed_world_eligibility_after_P1'] for r in checked),
        selected_stage_counts={a:dict(Counter(str(r['arms'][a]['selected_stage']) for r in checked)) for a in ARMS},
        minimum_corner_ratios={a:min(r['arms'][a]['minimum_corner_ratio'] for r in checked) for a in ARMS},rows=checked)
    (DIRECTORY/'independent_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
