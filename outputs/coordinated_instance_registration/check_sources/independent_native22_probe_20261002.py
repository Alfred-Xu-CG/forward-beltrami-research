"""Bounded native22 wrapper oracle; no registration and no real label access."""
import sys,json
from pathlib import Path
import numpy as np
root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src')]
from tools.coordinated_dhr_existing_score import native_metrics,_aggregate
from tools.coordinated_dhr_existing_inputs import existing_rows

def sample(field,points):
    h,w=field.shape[1:];q=np.clip(points,[0,0],[w-1,h-1]);i=np.floor(q).astype(int);t=q-i
    j=np.minimum(i+1,[w-1,h-1]);out=[]
    for (x,y),(xx,yy),(u,v) in zip(i,j,t):
        out.append((1-u)*(1-v)*field[:,y,x]+u*(1-v)*field[:,y,xx]+(1-u)*v*field[:,yy,x]+u*v*field[:,yy,xx])
    return np.array(out)

results=[]
for sf,sm,r,pf,pm,hw in [(1.,1.,1.37,[0,6],[7,0],(75,92)),(.73,.61,1.19,[0,0],[12,2],(55,77))]:
    fixed_wh=(127,91);moving_wh=(113,103);h,w=hw
    y,x=np.mgrid[:h,:w];field=np.stack((.2*np.sin(x*.13)+.03*y,-.3*np.cos(y*.11)+.017*x)).astype(np.float32)
    params=dict(target_resample_ratio=sf,source_resample_ratio=sm,initial_resample_ratio=r,
        pad_2=[[pf[1],pf[1]],[pf[0],pf[0]]],pad_1=[[pm[1],pm[1]],[pm[0],pm[0]]])
    fixed=np.array([[21.17,22.31],[83.13,61.77],[47.63,39.21]])
    lattice=((fixed+.5)*sf+pf)/r-.5
    expected=((lattice+sample(field,lattice)+.5)*r-pm)/sm-.5
    target=expected+np.array([[.4,-.7],[-1.3,.9],[.8,1.7]])
    scale=np.array([2.3,3.1]);layout=dict(effective_original_to_canvas_scale_xy=scale.tolist(),padding_xy=[19,23])
    scored=native_metrics(field,params,fixed,target,fixed_wh,moving_wh,layout,['a','b','c'])
    native=np.linalg.norm(expected-target,axis=1);canvas=np.linalg.norm((expected-target)*scale,axis=1)
    en=max(abs(native[k]-scored['native_moving_pixels']['per_label'][label]) for k,label in enumerate(['a','b','c']))
    ec=max(abs(canvas[k]-scored['canvas_pixels']['per_label'][label]) for k,label in enumerate(['a','b','c']))
    assert en<3e-5 and ec<8e-5
    results.append(dict(source_ratio=sm,target_ratio=sf,initial_ratio=r,native_error=en,canvas_error=ec))

rows=existing_rows(Path('unopened_root'))
assert len(rows)==22 and len({r['name'] for r in rows})==22
assert len({r[k] for r in rows for k in ['fixed','moving']})==9
assert len(rows[:20])==20 and {r['name'] for r in rows[:20]}=={f'{a}_to_{b}' for a in ['he','cc10','cd31','ki67','prospc'] for b in ['he','cc10','cd31','ki67','prospc'] if a!=b}
assert rows[-2]['fixed'].endswith('Images_CD4.jpg') and rows[-2]['moving'].endswith('Images_CD68.jpg')
assert rows[-1]['fixed'].endswith('Rat-Kidney_HE.jpg') and rows[-1]['moving'].endswith('Rat-Kidney_PanCytokeratin.jpg')
fake=[dict(status='ok',metrics={unit:dict(mean=float(i),p90=float(i+1)) for unit in ['canvas_pixels','native_moving_pixels']}) for i in range(20)]
assert _aggregate(fake)['all_directions']['canvas_pixels']['mean_pair_mean']==9.5
fake[-1]=dict(status='failed')
assert _aggregate(fake)['all_directions'] is None and _aggregate(fake)['failed_directions']==1
from tools.coordinated_existing_shared_frame import source_cases,configuration
base=root/'outputs/coordinated_instance_registration'
cases=source_cases(base/'lung_all20_f2reserve_t134/predictions.json',base/'development_p1arap3_stage30_t88')
assert [c['name'] for c in cases]==[r['name'] for r in rows]
for case in cases:
    old=case['original_configuration'];new=vars(configuration(old,Path('unwritten/new.npz')))
    assert new['mind_frame']=='shared_affine'
    for key in set(old)|set(new):
        if key not in ['mind_frame','output']:
            assert key in old and key in new
            assert new[key]==(Path(old[key]) if key in ['fixed','moving','affine','matches'] else old[key])
    assert old.get('seed_initializer','identity')=='identity'
    assert old.get('capture_prefix','none')=='none'
    if case['name'] in ['histo','rat_kidney']:
        expected=(case['name']+'_fixed512.png',case['name']+'_moving512.png')
    else:
        fixed,moving=case['name'].split('_to_')
        name=lambda stain:'cc10_fixed512.png' if stain=='he' else stain+'_moving512.png'
        expected=(name(fixed),name(moving))
    assert (Path(old['fixed']).name,Path(old['moving']).name)==expected
def postrun_check():
    import SimpleITK as sitk
    from tools.coordinated_dhr_existing_score import _case_data
    root_data=Path('D:/QC_optimization_data/digital_topology_wsi')
    output={};saved_scores={}
    for dirname in ['native22_dhr_standard_t20','existing22_shared_affine_a300_t20']:
        directory=base/dirname
        manifest=json.loads((directory/'predictions.json').read_text());scored=json.loads((directory/'landmark_scores.json').read_text())
        assert manifest['prediction_complete'] and not manifest['annotations_read']
        assert [r['name'] for r in scored['rows']]==[r['name'] for r in rows]
        assert [r['scored_landmarks'] for r in scored['rows']]==[80]*20+[77,69]
        assert all(r['status']=='ok' for r in manifest['rows']+scored['rows'])
        for r in scored['rows']:
            for key in ['canvas_pixels','native_moving_pixels']:
                m=r['metrics'][key];errors=np.array(list(m['per_label'].values()))
                assert len(errors)==r['scored_landmarks'] and np.isfinite(errors).all()
                assert abs(errors.mean()-m['mean'])<1e-12 and abs(np.percentile(errors,90)-m['p90'])<1e-12
        groups=[scored['rows'][:20],scored['rows'][20:21],scored['rows'][21:]]
        means=[float(np.mean([r['metrics']['canvas_pixels']['mean'] for r in group])) for group in groups]
        for value,key in zip(means,['lung_all20','histo','rat_kidney']):
            assert abs(value-scored[key]['all_directions']['canvas_pixels']['mean_pair_mean'])<1e-12
        assert abs(np.mean(means)-scored['equal_specimen_canvas']['mean_pair_mean'])<1e-12
        if dirname.startswith('existing'):
            assert all(r['gradient_steps']==300 and r['failed_trials']==0 for r in manifest['rows'])
            assert all(r['map_certificate']['composite_representation_valid'] for r in scored['rows'])
            for r in manifest['rows']:
                before=r['original_configuration'];after=r['configuration']
                for key in set(before)|set(after):
                    if key not in ['mind_frame','output']:
                        assert (Path(before[key])==Path(after[key]) if key in ['fixed','moving','affine','matches'] else before[key]==after[key]),(r['name'],key,before.get(key),after.get(key))
        else:assert all(r['topology']['nonpositive_corners']>0 and not r['topology']['global_homeomorphism_certified'] for r in scored['rows'])
        output[dirname]=dict(canvas_specimen_means=means,equal_specimen_mean=float(np.mean(means)),all22complete=True,total_labels=sum(r['scored_landmarks'] for r in scored['rows']),elapsed_seconds=manifest['elapsed_seconds'])
        saved_scores[dirname]=(manifest,scored)
    native_manifest,native_score=saved_scores['native22_dhr_standard_t20'];_,analytic_score=saved_scores['existing22_shared_affine_a300_t20']
    assert all(n['available_pair_labels']==a['available_pair_labels'] for n,a in zip(native_score['rows'],analytic_score['rows'],strict=True))
    # Actual nonuniform native fields: one direction per specimen, all its labels.
    probes=[];directory=base/'native22_dhr_standard_t20'
    for index in [0,20,21]:
        r=native_manifest['rows'][index];s=native_score['rows'][index]
        wh,layouts,points,ids,_,_=_case_data(r,root_data)
        field=sitk.GetArrayFromImage(sitk.ReadImage(str((directory/r['field']).resolve())))
        params=json.loads((directory/r['postprocessing_params']).resolve().read_text())
        pf=np.array([params['pad_2'][1][0],params['pad_2'][0][0]]);pm=np.array([params['pad_1'][1][0],params['pad_1'][0][0]])
        sf,sm,ratio=params['target_resample_ratio'],params['source_resample_ratio'],params['initial_resample_ratio']
        q=((points['fixed']+.5)*sf+pf)/ratio-.5
        predicted=((q+sample(field,q)+.5)*ratio-pm)/sm-.5
        native=np.linalg.norm(predicted-points['moving'],axis=1)
        canvas=np.linalg.norm((predicted-points['moving'])*layouts['moving']['effective_original_to_canvas_scale_xy'],axis=1)
        en=max(abs(native[k]-s['metrics']['native_moving_pixels']['per_label'][label]) for k,label in enumerate(ids))
        ec=max(abs(canvas[k]-s['metrics']['canvas_pixels']['per_label'][label]) for k,label in enumerate(ids))
        assert en<.01 and ec<.001
        probes.append(dict(name=r['name'],labels=len(ids),float64_oracle_native_error=en,float64_oracle_canvas_error=ec))
    output['actual_nonuniform_field_probes']=probes
    initial=json.loads((directory/'initial_only_scores.json').read_text())
    assert initial['scored_pairs']==22 and initial['final_dense_fields_read'] is False
    assert initial['registration_or_selection_performed'] is False
    for r,s in zip(initial['rows'],native_score['rows'],strict=True):
        assert r['name']==s['name'] and r['available_pair_labels']==s['available_pair_labels'] and r['status']=='ok'
        for unit in ['canvas_pixels','native_moving_pixels']:
            errors=np.array(list(r['metrics'][unit]['per_label'].values()))
            assert abs(errors.mean()-r['metrics'][unit]['mean'])<1e-12
            assert abs(np.percentile(errors,90)-r['metrics'][unit]['p90'])<1e-12
    group_means=[float(np.mean([r['metrics']['canvas_pixels']['mean'] for r in group])) for group in [initial['rows'][:20],initial['rows'][20:21],initial['rows'][21:]]]
    assert abs(np.mean(group_means)-initial['equal_specimen_canvas']['mean_pair_mean'])<1e-12
    output['native_initial_canvas_specimen_means']=group_means
    reserve=json.loads((base/'lung_all20_f2reserve_t134/all_id_scores.json').read_text())
    f2means=[]
    for r,s in zip(reserve['rows'],analytic_score['rows'][:20],strict=True):
        assert r['name']==s['name'] and r['methods']['f2']['status']=='ok'
        e=r['methods']['f2']['metrics']['canvas_pixels']['per_landmark']
        assert sorted(e)==s['available_pair_labels'];f2means.append(float(np.mean(list(e.values()))))
    output['correct_reserve_t134_F2_lung_mean']=float(np.mean(f2means))
    return output

print(json.dumps(postrun_check() if '--postrun' in sys.argv else dict(coordinates=results,exact22directions=True,nine_images=True,failed_denominator_retained=True,actual_analytic22_sources_checked=True,only_config_changes=['mind_frame','output'],real_annotations_read=False)))
