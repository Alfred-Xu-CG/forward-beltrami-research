"""Read-only saved conditional-run audit; --scores only after all25 terminal.

No inference or optimization. Timing checks verify recorded arithmetic/scope,
not the historical clock or allocator independently. Fresh tables are checked
without requiring them to equal archives, then used in literal objectives.
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
    BASE,ARCHIVE,IDENTITY,read,load_map,objective_data,literal_objective,
    compare_objective,label_check)
KEYS=('source_points_unit','target_points_unit','confidence','post_affine_matrix','post_affine_offset')


def archive_path(value):
    value=str(value).replace('\\','/')
    if '/results/' in value:return BASE/value.split('/results/',1)[1]
    if value.startswith('results/'):return BASE/value[len('results/'):]
    path=Path(value)
    assert path.exists(),value
    return path


def compare(fresh,old,claimed):
    exact=True;maximum=0.;changed=[]
    for key in KEYS:
        a=np.asarray(fresh[key]);b=np.asarray(old[key]);record=claimed['arrays'][key]
        shape=a.shape==b.shape;finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        equal=bool(np.array_equal(a,b));difference=(float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.) if shape and finite else None
        assert record==dict(fresh_shape=list(a.shape),archived_shape=list(b.shape),exact=equal,finite=finite,maximum_absolute_difference=difference)
        exact &= equal
        if not equal:changed.append(key)
        if difference is not None:maximum=max(maximum,difference)
    assert claimed['all_arrays_exact']==exact
    return dict(exact=exact,changed_arrays=changed,maximum_absolute_difference=maximum)


def literal_fusion(sg,ma,fused):
    """Independent all-row concatenation and eligible confidence normalization."""
    a=np.array(sg['post_affine_matrix']);b=np.array(sg['post_affine_offset'])
    assert np.array_equal(a,ma['post_affine_matrix']) and np.array_equal(b,ma['post_affine_offset'])
    out={k:[] for k in ('source_points_unit','target_points_unit','confidence')}
    for table in (sg,ma):
        q=np.array(table['source_points_unit']);p=np.array(table['target_points_unit']);c=np.array(table['confidence'])
        assert q.shape==p.shape and q.shape[1]==2 and c.shape==(len(q),)
        assert np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(c).all()
        assert (q>=0).all() and (q<=1).all() and (p>=0).all() and (p<=1).all() and (c>=0).all()
        mapped=p@a.T+b;eligible=((mapped>=0)&(mapped<=1)).all(-1)
        mass=float(c[eligible].sum());assert mass>0
        for key,value in (('source_points_unit',q),('target_points_unit',p),('confidence',c/mass)):out[key].append(value)
    for key,parts in out.items():assert np.allclose(np.concatenate(parts),fused[key],rtol=0,atol=2e-16),key
    assert np.array_equal(a,fused['post_affine_matrix']) and np.array_equal(b,fused['post_affine_offset'])
    assert fused['rows_removed']==fused['rows_clipped']==0 and fused['match_weight']==.2
    assert fused['targets_manual_landmarks_or_dense_teacher_loaded'] is False


def scoring_scope_and_drift(directory,rows):
    adapters=read(directory/'miit_predictions.json');by_name={r['name']:r for r in rows}
    for record in adapters['rows']:
        assert record['methods']['analytic']==by_name[record['name']]
        assert record['raw_matches']['arm']=='fresh_fusion'
        assert '/'+directory.name+'/fused_tables/' in record['raw_matches']['path'].replace('\\','/')
        for method in ('dhr','f2'):
            old=record['methods'][method]
            assert old['pilot_reference']=='unchanged archived comparison; not rerun'
            for key in ('output','report','field','output_directory'):
                if key in old:assert directory.name not in str(old[key])
    existing=read(directory/'existing_predictions.json')
    assert existing['rows']==[r for r in rows if r['cohort']=='existing']
    def scores(folder):
        values={}
        for cohort in ('miit','existing'):
            for r in read(folder/(cohort+'_scores.json'))['rows']:
                metrics=r['methods']['analytic']['metrics'] if cohort=='miit' else r['metrics']
                values[r['name']]=metrics['canvas_pixels']
        return values
    current=scores(directory);previous=scores(ARCHIVE);drift={};pointmax=0.
    for name,value in current.items():
        old=previous[name];assert value['per_label'].keys()==old['per_label'].keys()
        pointmax=max(pointmax,max(abs(v-old['per_label'][k]) for k,v in value['per_label'].items()))
        drift[name]={k:value[k]-old[k] for k in ('mean','p90','maximum')}
    return dict(fresh_scored_method='analytic only; all25 fresh pipeline maps',
        archived_MIIT_methods='f2 and dhr are explicitly marked unchanged archived comparison; not rerun',
        maximum_original_landmark_TRE_change_from_retained_fusion=pointmax,
        maximum_absolute_case_metric_change_from_retained_fusion={k:max(abs(r[k]) for r in drift.values()) for k in ('mean','p90','maximum')},
        case_metric_deltas_new_minus_retained=drift)


def audit(directory,scores=False):
    manifest=read(directory/'predictions.json');old=read(ARCHIVE/'predictions.json')
    maold=read(BASE/'matchanything_all25_t22/predictions.json')
    assert manifest['prediction_complete'] and manifest['extraction_complete'] and manifest['annotations_read'] is False
    assert manifest['cohort_size']==manifest['attempt_denominator']==25 and manifest['specimen_count']==4
    rows=manifest['rows'];names=[r['name'] for r in rows]
    assert names==[r['name'] for r in old['rows']]==[r['name'] for r in maold['rows']] and len(set(names))==25
    assert all(r['status'] in ('ok','failed') for r in rows)
    successful=sum(r['status']=='ok' for r in rows)
    assert manifest['successful']==successful and manifest['failed']==25-successful
    whole=manifest['whole_batch_seconds'];assert whole>0
    assert manifest['batch_seconds_per_attempt']==whole/25 and manifest['successful_maps_per_batch_second']==successful/whole
    costs=0.;times=[];count=0;maxima={};label_errors={};checks=[];failures=[];labels=0
    max_component_allocated=manifest.get('ma_setup',{}).get('setup_peak_allocated_bytes',0);actual_exact=True
    extraction_components=0.
    for row,oldrow,marow in zip(rows,old['rows'],maold['rows'],strict=True):
        name=row['name'];oldcfg=oldrow['configuration'];cfg=row['configuration']
        assert set(cfg)==set(oldcfg) and {k for k in cfg if cfg[k]!=oldcfg[k]}<= {'output','matches'}
        assert row['original_configuration']==oldrow['original_configuration']
        c=row['costs'];assert all(np.isfinite(v) and v>=0 for v in c.values())
        assert row['complete_call_seconds']==sum(c.values());costs+=row['complete_call_seconds']
        extraction_components+=sum(v for k,v in c.items() if k.startswith(('sg_','ma_')))
        for stage in ('sg','ma'):
            value=row.get(stage+'_diagnostics',{}).get('peak_allocated_bytes')
            if value is not None:max_component_allocated=max(max_component_allocated,value)
        comparisons={};tables={}
        for stage,archive in (('sg',archive_path(oldrow['original_configuration']['matches'])),
                ('ma',BASE/'matchanything_all25_t22'/Path(marow['match_table']).name),
                ('fused',BASE/'match_fusion_all50_t23/fused_tables'/(name+'.json'))):
            status_key='fusion_status' if stage=='fused' else stage+'_status'
            if row[status_key]=='ok':
                fresh=read(directory/(stage+'_tables')/(name+'.json'));tables[stage]=fresh
                comparisons[stage]=compare(fresh,read(archive),row[stage+'_archive_comparison'])
                actual_exact &= comparisons[stage]['exact']
            else:actual_exact=False
        if 'fused' in tables:literal_fusion(tables['sg'],tables['ma'],tables['fused'])
        if row['status']=='failed':
            assert row.get('errors');failures.append(dict(name=name,errors=row['errors']));continue
        assert all(row[s+'_status']=='ok' for s in ('sg','ma','fusion','optimization'))
        time=row['completed_map_elapsed_from_batch_start'];assert 0<time<whole
        if times:assert time>times[-1]
        times.append(time)
        report=read(directory/row['report']);assert report['configuration']==cfg
        for key in ('gradient_steps','failed_trials','evaluations','saved_binary_certificate','initial','final','peak_allocated_bytes'):
            assert row[key]==report[key],key
        assert report['gradient_steps']==300 and report['failed_trials']==0 and not report['landmarks_used']
        max_component_allocated=max(max_component_allocated,report['peak_allocated_bytes'])
        v,a,b,qmin=load_map(directory/row['output'])
        previous,aa,bb,_=load_map(ARCHIVE/oldrow['output'])
        assert np.array_equal(a,aa) and np.array_equal(b,bb)
        assert abs(qmin-row['actual_minimum_corner_ratio'])<1e-13
        table=tables['fused'];evidence=objective_data(name,cfg,a,b)
        q=np.asarray(table['source_points_unit']);p=np.asarray(table['target_points_unit']);confidence=np.asarray(table['confidence'])
        eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1);weight=confidence*eligible;weight/=weight.sum()
        evidence.update(q=q,p=p,weight=weight)
        compare_objective(literal_objective(IDENTITY,evidence),report['initial'],maxima)
        compare_objective(literal_objective(v,evidence),report['final'],maxima)
        checked=dict(name=name,tables=comparisons,minimum_corner_ratio=qmin,
            saved_map_bitwise_archived=bool(np.array_equal(v,previous)),
            saved_map_maximum_difference=float(np.max(abs(v-previous))),
            completed_map_elapsed_from_batch_start=time,component_sum_seconds=row['complete_call_seconds'])
        if scores:
            result=label_check(directory,name,row,'');labels+=result['landmarks'];checked['scores']=result['metrics']
            for key,value in result['maximum_errors'].items():label_errors[key]=max(label_errors.get(key,0),value)
        checks.append(checked);count+=1
        print(json.dumps(dict(checked=name,tables_exact=all(v['exact'] for v in comparisons.values()),map_exact=checked['saved_map_bitwise_archived'])),flush=True)
    assert manifest['time_to_first_saved_map']==(times[0] if times else None)
    assert manifest['all_tables_exact']==actual_exact
    shared=manifest['ma_setup_seconds']+manifest['ma_close_seconds'];assert costs+shared<=whole
    if times:assert extraction_components+shared<times[0]
    memory=manifest['whole_batch_memory'];assert memory['peak_reserved_bytes']>=memory['peak_allocated_bytes']>=max_component_allocated
    assert memory['captured_segments']>=1 and manifest['sg_setup_attempts']==25 and manifest['ma_setup_attempts']==1
    if manifest['ma_setup_status']=='ok':assert manifest['ma_extraction_attempts']==25
    if scores and not failures:assert labels==2074
    score_scope=scoring_scope_and_drift(directory,rows) if scores and not failures else None
    result=dict(independent_saved_output_check_passed=True,attempt_denominator=25,successful=count,failures=failures,
        all_tables_exact=actual_exact,all_successful_maps_bitwise_archived=all(c['saved_map_bitwise_archived'] for c in checks),
        whole_batch_seconds=whole,time_to_first_saved_map=manifest['time_to_first_saved_map'],
        per_case_component_sum_seconds=costs,shared_MA_setup_and_close_seconds=shared,
        SG_MA_extraction_and_comparison_component_seconds=extraction_components,
        unallocated_bookkeeping_and_scope_seconds=whole-costs-shared,
        whole_batch_memory=memory,memory_scope='recorded peak arithmetic and independently reviewed capture lifecycle; historical CUDA allocator not independently replayed',
        maximum_objective_errors=maxima,original_CSVs_checked=scores,original_CSV_count=labels,
        maximum_original_CSV_score_errors=label_errors,scoring_scope_and_retained_fusion_drift=score_scope,rows=checks)
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--scores',action='store_true')
    args=parser.parse_args();audit(args.directory,args.scores)
