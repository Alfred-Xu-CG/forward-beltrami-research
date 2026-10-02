"""Independent saved reverse-MIIT maps/fields, frames, objectives and CSV errors.

No inference. Original CSVs are opened only after all nine declared attempts
have terminal statuses. Final native topology remains a local diagnostic.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import SimpleITK as sitk

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_miit_reverse_probe_20261002 import BASE,MIIT,NAMES,REVERSES,read,explicit_inverse
from independent_data_metric_postrun_20261002 import GRID,IDENTITY,load_map,literal_objective,compare_objective
from independent_joint_pose_probe_20261002 import bilinear,descriptor,vector_p1
from independent_native_shared_postrun_20261002 import sample,topology
from independent_native_shared_probe_20261002 import canvas_to_z,z_to_canvas
from independent_conditional_pipeline_postrun_20261002 import literal_fusion
METHODS=('sg1','fusion','native_shared')


def input_check(row,old,directory):
    forward=old['name'];cfg=old['configuration']
    assert row['source_forward_case']==forward and row['source_forward_configuration']==cfg
    _,m,_,f=forward.split('_');assert row['name']==f'miit_{f}_to_{m}'
    assert row['fixed_section']==int(m) and row['moving_section']==int(f)
    assert row['map_direction']=='fixed_canvas_to_moving_canvas'
    layout=read(directory/row['layout']);before=read(BASE/'miit_three_rotations_t153'/(forward+'_layout.json'))
    assert layout['side']==512 and layout['role_swap_of']==forward
    for role,oldrole in (('fixed','moving'),('moving','fixed')):
        original=BASE/'miit_three_rotations_t153'/Path(cfg[oldrole]).name
        assert (directory/row[role]).read_bytes()==original.read_bytes()
        assert row['accepted512_layouts'][role]==layout[role]
        for key,value in before[oldrole].items():
            if key not in ('source','canvas_png'):assert layout[role][key]==value
        assert Path(layout[role]['canvas_png']).name==row[role]
        assert layout[role]['source']==row['source_'+role]
        section=row[role+'_section'];assert Path(row['source_'+role]).parts[-3:]==(str(section),'images','image.tif')
        with Image.open(MIIT/str(section)/'images/image.tif') as image:
            assert list(image.size)==layout[role]['original_wh'] and image.getexif().get(274,1)==1
    with np.load(BASE/'miit_three_rotations_t153'/Path(cfg['affine']).name) as z:a=z['post_affine_matrix'].astype(float);b=z['post_affine_offset'].astype(float)
    ai,bi=explicit_inverse(a,b)
    with np.load(directory/row['affine']) as z:
        ar=z['post_affine_matrix'];br=z['post_affine_offset']
        assert ar.dtype==br.dtype==np.float32 and np.array_equal(ar,ai.astype(np.float32)) and np.array_equal(br,bi.astype(np.float32))
    assert np.array_equal(ar,row['shared_affine_matrix']) and np.array_equal(br,row['shared_affine_offset'])
    ar=ar.astype(float);br=br.astype(float);audit=row['inverse_audit']
    assert np.linalg.det(ar)>0 and audit['optimized_map_inverted'] is False and audit['stored_dtype']=='float32'
    assert np.array_equal(a,audit['original_matrix']) and np.array_equal(b,audit['original_offset'])
    assert np.max(abs(ai-audit['inverse_matrix_float64']))<3e-16 and np.max(abs(bi-audit['inverse_offset_float64']))<3e-16
    q=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    for key,error in [('reverse_after_forward_max_corner_canvas_pixels',np.linalg.norm(((q@a.T+b)@ar.T+br-q)*512,axis=1).max()),
            ('forward_after_reverse_max_corner_canvas_pixels',np.linalg.norm(((q@ar.T+br)@a.T+b-q)*512,axis=1).max())]:
        assert abs(audit[key]-error)<1e-12
    source=row['original_configuration'];assert set(source)==set(cfg)
    assert {k for k in source if source[k]!=cfg[k]}<={'fixed','moving','affine','match_weight'}
    assert source['match_weight']==.1
    for key in ('fixed','moving','affine'):assert Path(source[key]).name==row[key]
    return layout,ar,br


def original_csv(row,layout):
    points={};absent={}
    for role in ('fixed','moving'):
        section=row[role+'_section'];path=MIIT/str(section)/'landmarks'/f'{section:02}.csv'
        with path.open(encoding='utf-8-sig') as stream:records=list(csv.DictReader(stream))
        assert len(records)==124
        values={}
        for record in records:
            label=record['label'].strip();assert label and label not in values
            xy=np.array([float(record['x']),float(record['y'])]);wh=np.array(layout[role]['original_wh'])
            assert np.isposinf(xy).all() or (np.isfinite(xy).all() and (xy>=-.5).all() and (xy<=wh-.5).all())
            values[label]=xy
        points[role]=values;absent[role]=sorted(k for k,v in values.items() if np.isposinf(v).all())
    nominal=sorted(points['fixed']);assert nominal==sorted(points['moving']) and len(nominal)==124
    missing=sorted(set(absent['fixed'])|set(absent['moving']));ids=[i for i in nominal if i not in missing]
    return {role:np.stack([values[i] for i in ids]) for role,values in points.items()},ids,missing,absent


def score_check(predicted_native,target,layout,ids,record,tolerance):
    assert record['status']=='ok'
    error=predicted_native-target
    values={'native_moving_pixels':np.linalg.norm(error,axis=1),
            'canvas_pixels':np.linalg.norm(error*np.array(layout['effective_original_to_canvas_scale_xy']),axis=1)}
    errors={}
    for unit,expected in values.items():
        reported=record['metrics'][unit];actual=np.array([reported['per_label'][i] for i in ids])
        assert set(reported['per_label'])==set(ids)
        error=float(abs(expected-actual).max());assert error<tolerance[unit],(unit,error)
        for key,value in [('mean',actual.mean()),('p90',np.percentile(actual,90)),('maximum',actual.max())]:
            assert abs(value-reported[key])<1e-11,(key,value,reported[key])
        errors[unit]=error
    return errors


def safe_check(directory,row,method,layout,points,ids,scored,a,b):
    record=row['methods'][method];report=read(directory/record['report']);cfg=report['configuration']
    assert record['configuration']==cfg and report['landmarks_used'] is False
    assert report['gradient_steps']==300 and report['failed_trials']==0
    assert report['query_count']==512**2 and report['control_vertices']==257**2
    source=row['original_configuration'];assert set(cfg)==set(source)
    assert {k for k in cfg if cfg[k]!=source[k]}<={'matches','output','match_weight'}
    match_weight=.1 if method=='sg1' else .2;assert cfg['match_weight']==match_weight
    v,_,_,qmin=load_map(directory/record['output'],a,b)
    assert abs(qmin-record['actual_minimum_corner_ratio'])<1e-13
    table=read(directory/(row['name']+('_sg.json' if method=='sg1' else '_fused_matches.json')))
    assert Path(cfg['matches']).name==row['name']+('_sg.json' if method=='sg1' else '_fused_matches.json')
    assert Path(cfg['matches'])!=Path(cfg['output']).with_suffix('.json')
    images={role:1-np.asarray(Image.open(directory/row[role]).convert('L'),dtype=np.float32)/255 for role in ('fixed','moving')}
    aligned=bilinear(images['moving'],((2*(GRID@a.T+b)-1).astype(np.float32).astype(float)+1)/2)[...,0]
    q=np.asarray(table['source_points_unit']);p=np.asarray(table['target_points_unit']);c=np.asarray(table['confidence'])
    eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1);weight=c*eligible;weight/=weight.sum()
    evidence=dict(a=a,b=b,fixed_feature=descriptor(images['fixed']),moving_feature=descriptor(aligned),mask=images['fixed']>.04,q=q,p=p,weight=weight)
    maxima={}
    for vertex,key in ((IDENTITY,'initial'),(v,'final')):
        expected=literal_objective(vertex,evidence)
        expected['total']=expected['image']+3*expected['strain']+1e-4*expected['shape']+match_weight*expected['match']+expected['oob']
        compare_objective(expected,report[key],maxima)
    query=((points['fixed']+.5)*layout['fixed']['effective_original_to_canvas_scale_xy']+layout['fixed']['padding_xy'])/512
    predicted=vector_p1(v,query)@a.T+b
    native=(predicted*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5
    errors=score_check(native,points['moving'],layout['moving'],ids,scored,dict(native_moving_pixels=1e-8,canvas_pixels=1e-9))
    return dict(minimum_corner_ratio=qmin,maximum_objective_errors=maxima,maximum_original_CSV_errors=errors)


def native_check(directory,row,layout,points,ids,scored,a,b):
    record=row['methods']['native_shared'];assert np.array_equal(record['shared_affine_matrix'],a) and np.array_equal(record['shared_affine_offset'],b)
    config=read(directory/record['configuration']);preset=read(directory/Path(record['configuration']).parent/'released_preset.json')
    allowed={'device','case_name','logging_path','save_final_images','loading_params','saving_params','preprocessing_params','initial_registration_params','nonrigid_registration_params'}
    assert {k for k in preset if preset[k]!=config[k]}<=allowed
    for section,changed in [('preprocessing_params',{'save_results'}),('nonrigid_registration_params',{'save_results','device'}),
        ('initial_registration_params',{'save_results','device','cuda'}),('loading_params',{'loader','source_resample_ratio','target_resample_ratio'}),('saving_params',{'final_saver'})]:
        assert {k for k in preset[section] if preset[section][k]!=config[section][k]}<=changed
    assert config['nonrigid_registration_params']['iterations']==[100]*7+[200]
    assert config['nonrigid_registration_params']['registration_size']==4096
    params=read(directory/record['postprocessing_params']);assert params==record['native_padding_params']
    assert params['source_resample_ratio']==params['target_resample_ratio']==1
    fixedwh=np.array(layout['fixed']['original_wh']);movingwh=np.array(layout['moving']['original_wh']);padded=np.maximum(fixedwh,movingwh)
    for role,key,wh in [('fixed','pad_2',fixedwh),('moving','pad_1',movingwh)]:
        assert params[key]==[[int(d//2),int(d-d//2)] for d in (padded-wh)[::-1]]
    ratio=max(1.,min(padded)/4096);assert params['initial_resample_ratio']==ratio
    wh=np.floor(padded/ratio).astype(int);h,w=wh[::-1]
    assert record['preprocessed_shape']==[1,1,int(h),int(w)] and record['field_shape']==[1,int(h),int(w),2]
    pf=np.array([params['pad_2'][1][0],params['pad_2'][0][0]]);pm=np.array([params['pad_1'][1][0],params['pad_1'][0][0]])
    z=np.array([[0.,0.],[1,0],[0,1]])
    canvas=z_to_canvas(z,layout['fixed'],pf,1.,ratio,wh)
    mapped=canvas_to_z(canvas@a.T+b,layout['moving'],pm,1.,ratio,wh)
    theta=np.column_stack((mapped[1]-mapped[0],mapped[2]-mapped[0],mapped[0]))
    theta_error=float(abs(theta-record['conjugate_theta64']).max());assert theta_error<4e-15
    assert np.array_equal(record['initial_transform'][0],np.asarray(record['conjugate_theta64'],np.float32).astype(float))
    audit=record['initial_field_audit'];query=audit['accepted512_query_audit']
    assert audit['preprocessed_nodes']==h*w and audit['maximum_canvas_pixel_error']<.001 and audit['dense_extrapolation_outside_lattice_guaranteed'] is False
    assert query['query_count']==512**2 and query['no_queries_dropped']
    z=canvas_to_z(GRID,layout['fixed'],pf,1.,ratio,wh)
    expected=GRID@a.T+b
    cast=np.array(record['initial_transform'])[0]
    exactq=z_to_canvas(z@theta[:,:2].T+theta[:,2],layout['moving'],pm,1.,ratio,wh)
    castq=z_to_canvas(z@cast[:,:2].T+cast[:,2],layout['moving'],pm,1.,ratio,wh)
    exact_error=float(np.linalg.norm((exactq-expected)*512,axis=-1).max());cast_error=float(np.linalg.norm((castq-expected)*512,axis=-1).max())
    assert exact_error<1e-10 and abs(cast_error-query['cast_theta32_max_canvas_pixels'])<2e-10
    inside=(abs(z[...,0])<=1-1/w)&(abs(z[...,1])<=1-1/h)
    assert query['native_lattice_inside_count']==int(inside.sum()) and query['native_lattice_outside_count']==int((~inside).sum())
    field=sitk.GetArrayFromImage(sitk.ReadImage(str(directory/record['field'])))
    assert field.shape==(2,h,w) and field.dtype==np.float32 and np.isfinite(field).all()
    lattice=(points['fixed']+.5+pf)/ratio-.5
    native=(lattice+sample(field.astype(float),lattice)+.5)*ratio-pm-.5
    errors=score_check(native,points['moving'],layout['moving'],ids,scored,dict(native_moving_pixels=.01,canvas_pixels=.001))
    topo=topology(field);reported=scored['metadata']['topology']
    for key,value in topo.items():assert reported[key]==value,(key,value,reported[key])
    assert reported['local_orientation_valid']==(topo['nonpositive_corners']==0) and reported['global_homeomorphism_certified'] is False
    return dict(maximum_original_CSV_errors=errors,topology=topo,independent_theta64_error=theta_error,
        analytic_initializer_canvas_error=exact_error,cast_initializer_canvas_error=cast_error,
        initial_field_scope='theta independently reconstructed; unavailable initial field not falsely claimed reread',field_bytes=(directory/record['field']).stat().st_size)


def audit(directory):
    manifest=read(directory/'predictions.json')
    assert manifest['prediction_complete'] and manifest['all9_terminal'] and manifest['annotations_read'] is False
    assert manifest['attempt_denominator']==9 and manifest['cohort_size']==3
    rows=manifest['rows'];assert [r['name'] for r in rows]==list(REVERSES)
    assert all(set(r['methods'])==set(METHODS) and all(r['methods'][m]['status'] in ('ok','failed') for m in METHODS) for r in rows)
    successful=sum(r['methods'][m]['status']=='ok' for r in rows for m in METHODS)
    assert manifest['successful_calls']==successful and manifest['failed_calls']==9-successful
    scores=read(directory/'scores.json');assert [r['name'] for r in scores['rows']]==list(REVERSES)
    originals=read(BASE/'match_fusion_all50_t23/fusion/predictions.json')['rows'][:3]
    checks=[];failures=[];labels=0;method_labels={m:0 for m in METHODS}
    for row,old,scored in zip(rows,originals,scores['rows'],strict=True):
        if row['input_status']!='ok':
            failures.append(dict(name=row['name'],stage='input',error=row.get('input_error')));continue
        layout,a,b=input_check(row,old,directory)
        points,ids,missing,absent=original_csv(row,layout);labels+=len(ids)
        assert scored['available_pair_labels']==ids and scored['unavailable_pair_labels']==missing and scored['nominal_landmarks']==124
        assert scored['unavailable_fixed_labels']==absent['fixed'] and scored['unavailable_moving_labels']==absent['moving']
        assert scored['scored_landmarks']==len(ids)
        q=((points['fixed']+.5)*layout['fixed']['effective_original_to_canvas_scale_xy']+layout['fixed']['padding_xy'])/512
        native=((q@a.T+b)*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5
        affine_errors=score_check(native,points['moving'],layout['moving'],ids,scored['methods']['common_affine'],dict(native_moving_pixels=1e-8,canvas_pixels=1e-9))
        if row['sg_status']=='ok' and row['ma_status']=='ok':
            literal_fusion(read(directory/row['sg_table']),read(directory/row['ma_table']),read(directory/(row['name']+'_fused_matches.json')))
        result=dict(name=row['name'],available_IDs=len(ids),inverse_audit=row['inverse_audit'],common_affine_score_errors=affine_errors,methods={})
        for method in METHODS:
            if row['methods'][method]['status']!='ok':
                assert scored['methods'][method]['status']=='failed'
                failures.append(dict(name=row['name'],method=method,error=row['methods'][method].get('error')));continue
            result['methods'][method]=(native_check(directory,row,layout,points,ids,scored['methods'][method],a,b) if method=='native_shared' else
                safe_check(directory,row,method,layout,points,ids,scored['methods'][method],a,b))
            method_labels[method]+=len(ids)
        checks.append(result);print(json.dumps(dict(checked=row['name'],available_IDs=len(ids),methods=list(result['methods']))),flush=True)
    for method in ('common_affine',)+METHODS:
        good=[r for r in scores['rows'] if r['methods'][method]['status']=='ok'];a=scores['aggregate'][method]
        assert a['pair_denominator']==3 and a['scored_pairs']==len(good) and a['failed_pairs']==3-len(good)
        assert (a['all_three'] is not None)==(len(good)==3)
        if good:
            for unit in ('native_moving_pixels','canvas_pixels'):
                metrics=[r['methods'][method]['metrics'][unit] for r in good]
                expected=dict(mean_pair_mean=float(np.mean([m['mean'] for m in metrics])),mean_pair_p90=float(np.mean([m['p90'] for m in metrics])),maximum=max(m['maximum'] for m in metrics))
                assert a['successful_only'][unit]==expected
                if len(good)==3:assert a['all_three'][unit]==expected
    if not failures:assert labels==328 and method_labels==dict(sg1=328,fusion=328,native_shared=328)
    memory=manifest['whole_batch_memory'];assert memory['peak_reserved_bytes']>=memory['peak_allocated_bytes']>0
    for row in rows:
        for method in METHODS:
            p=row['methods'][method].get('peak_allocated_bytes')
            if p is not None:assert p<=memory['peak_allocated_bytes']
    result=dict(independent_reverse_review_passed=True,attempt_denominator=9,successful_calls=successful,failures=failures,
        original_available_ID_count=labels,original_errors_checked_per_method=method_labels,original_CSV_errors_checked=sum(method_labels.values()),
        whole_batch_memory=memory,aggregate=scores['aggregate'],rows=checks)
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    audit(parser.parse_args().directory)
