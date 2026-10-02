"""All25 saved native fields, exact frames and original-CSV scoring oracle."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
import SimpleITK as sitk
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_stain_proxy_probe_20261002 import original_landmark_case
from independent_native_shared_probe_20261002 import canvas_to_z,z_to_canvas
BASE=ROOT/'outputs/coordinated_instance_registration'
DIRECTORY=BASE/'native_standard_shared25_t27'
DATA=Path('D:/QC_optimization_data/digital_topology_wsi')
MIIT=Path('D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

def sample(field,points):
    h,w=field.shape[1:];q=np.clip(points,[0,0],[w-1,h-1]);i=np.floor(q).astype(int);t=q-i;j=np.minimum(i+1,[w-1,h-1]);out=[]
    for (x,y),(xx,yy),(u,v) in zip(i,j,t):
        out.append((1-u)*(1-v)*field[:,y,x]+u*(1-v)*field[:,y,xx]+(1-u)*v*field[:,yy,x]+u*v*field[:,yy,xx])
    return np.array(out)

def topology(field):
    h,w=field.shape[1:];minimum=float('inf');nonpositive=0
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for j in range(0,h-1,96):
        end=min(h-1,j+96);y,x=np.meshgrid(np.arange(j,end+1),np.arange(w),indexing='ij')
        vertex=field[:,j:end+1].transpose(1,2,0).astype(float)+np.stack((x,y),-1)
        a,b,c,d=vertex[:-1,:-1],vertex[:-1,1:],vertex[1:,1:],vertex[1:,:-1]
        # Local horizontal and vertical edge pairs at each corner, separately
        # expressed rather than calling the production diagnostic.
        for det in (cross(b-a,d-a),cross(b-a,c-b),cross(c-d,c-b),cross(c-d,d-a)):
            minimum=min(minimum,float(det.min()));nonpositive+=int((det<=0).sum())
    return dict(minimum_corner_ratio=minimum,nonpositive_corners=nonpositive,checked_corner_count=4*(h-1)*(w-1))

def main():
    manifest=read(DIRECTORY/'predictions.json');rows=manifest['rows']
    assert manifest['prediction_complete'] and manifest['all25_terminal'] and not manifest['annotations_read'] and len(rows)==25 and manifest['successful_pairs']==25
    assert all(r['status']=='ok' for r in rows)
    scores={r['name']:r for cohort in ('miit','existing') for r in read(DIRECTORY/(cohort+'_scores.json'))['rows']}
    sources={r['name']:r for r in read(BASE/'native22_inputs_t20/local_inputs.json')['rows']}
    preset=read(DIRECTORY/'released_preset.json')
    archive=BASE/'match_fusion_all50_t23/fusion'
    yy,xx=np.meshgrid(np.arange(512),np.arange(512),indexing='ij');canvas=np.stack(((xx+.5)/512,(yy+.5)/512),-1)
    errors=dict(theta64=0.,analytic_initializer_canvas=0.,cast_initializer_canvas=0.,canvas_score=0.,native_score=0.)
    checked=[];labels=0;field_bytes=0
    for row in rows:
        name=row['name'];score=scores[name];assert score['status']=='ok'
        layout,points,ids=original_landmark_case(name);assert ids==score['available_pair_labels'];labels+=len(ids)
        for role in ('fixed','moving'):
            for key in ('original_wh','resized_wh','padding_xy','effective_original_to_canvas_scale_xy'):
                assert layout[role][key]==row['accepted512_layouts'][role][key]
            if name.startswith('miit_'):
                section=row[role+'_section'];path=MIIT/str(section)/'images/image.tif'
            else:path=Path(sources[name][role])
            with Image.open(path) as im:
                assert list(im.size)==row[role+'_original_wh']==layout[role]['original_wh'] and im.mode==row[role+'_mode'] and im.getexif().get(274,1)==1
            assert path.stat().st_size==row[role+'_file_bytes']
        config=read(DIRECTORY/row['configuration'])
        allowed={'device','case_name','logging_path','save_final_images','loading_params','saving_params','preprocessing_params','initial_registration_params','nonrigid_registration_params'}
        assert {k for k in preset if preset[k]!=config[k]}<=allowed
        for section,allowed in [('preprocessing_params',{'save_results'}),('nonrigid_registration_params',{'save_results','device'}),('initial_registration_params',{'save_results','device','cuda'}),('loading_params',{'loader','source_resample_ratio','target_resample_ratio'}),('saving_params',{'final_saver'})]:
            assert {k for k in preset[section] if preset[section][k]!=config[section][k]}<=allowed
        assert config['nonrigid_registration_params']['iterations']==[100]*7+[200] and config['nonrigid_registration_params']['registration_size']==4096
        params=read(DIRECTORY/row['postprocessing_params']);assert params==row['native_padding_params']
        assert params['source_resample_ratio']==params['target_resample_ratio']==1
        fixedwh=np.array(layout['fixed']['original_wh']);movingwh=np.array(layout['moving']['original_wh']);padded=np.maximum(fixedwh,movingwh)
        for role,key,wh0 in [('fixed','pad_2',fixedwh),('moving','pad_1',movingwh)]:
            gap=padded-wh0;expected=[[int(d//2),int(d-d//2)] for d in gap[::-1]];assert params[key]==expected
        ratio=max(1.,min(padded)/4096);assert params['initial_resample_ratio']==ratio
        wh=np.floor(padded*(1/ratio)).astype(int);h,w=wh[::-1]
        assert row['loaded_padded_shape']==[1,3,int(padded[1]),int(padded[0])] and row['preprocessed_shape']==[1,1,int(h),int(w)] and row['field_shape']==[1,int(h),int(w),2]
        pf=np.array([params['pad_2'][1][0],params['pad_2'][0][0]]);pm=np.array([params['pad_1'][1][0],params['pad_1'][0][0]])
        with np.load(archive/(name+'_analytic.npz')) as original:a=original['post_affine_matrix'].astype(float);b=original['post_affine_offset'].astype(float)
        assert np.array_equal(row['shared_affine_matrix'],a) and np.array_equal(row['shared_affine_offset'],b)
        z=np.array([[0.,0.],[1,0],[0,1]])
        q=z_to_canvas(z,layout['fixed'],pf,1.,ratio,wh)
        mapped=canvas_to_z(q@a.T+b,layout['moving'],pm,1.,ratio,wh)
        theta=np.column_stack((mapped[1]-mapped[0],mapped[2]-mapped[0],mapped[0]))
        theta_error=float(abs(theta-np.array(row['conjugate_theta64'])).max());assert theta_error<3e-15
        theta32=np.array(row['initial_transform'])[0];assert np.array_equal(theta32,np.array(row['conjugate_theta64'],np.float32).astype(float))
        assert abs(np.linalg.det(theta32[:,:2])-row['initial_transform_determinant'])<1e-15
        z=canvas_to_z(canvas,layout['fixed'],pf,1.,ratio,wh)
        target=canvas@a.T+b
        exact=z_to_canvas(z@theta[:,:2].T+theta[:,2],layout['moving'],pm,1.,ratio,wh)
        cast=z_to_canvas(z@theta32[:,:2].T+theta32[:,2],layout['moving'],pm,1.,ratio,wh)
        exact_error=float(np.linalg.norm((exact-target)*512,axis=-1).max());cast_error=float(np.linalg.norm((cast-target)*512,axis=-1).max())
        audit=row['initial_field_audit'];query=audit['accepted512_query_audit'];inside=(abs(z[...,0])<=1-1/w)&(abs(z[...,1])<=1-1/h)
        assert audit['preprocessed_nodes']==h*w and audit['maximum_canvas_pixel_error']<.001 and not audit['dense_extrapolation_outside_lattice_guaranteed']
        assert exact_error<1e-10 and abs(cast_error-query['cast_theta32_max_canvas_pixels'])<2e-10
        assert query['query_count']==512**2 and query['no_queries_dropped'] and query['native_lattice_inside_count']==int(inside.sum()) and query['native_lattice_outside_count']==int((~inside).sum())
        errors['theta64']=max(errors['theta64'],theta_error);errors['analytic_initializer_canvas']=max(errors['analytic_initializer_canvas'],exact_error);errors['cast_initializer_canvas']=max(errors['cast_initializer_canvas'],cast_error)
        field_path=DIRECTORY/row['field'];field_bytes+=field_path.stat().st_size
        field=sitk.GetArrayFromImage(sitk.ReadImage(str(field_path)));assert field.shape==(2,h,w) and field.dtype==np.float32 and np.isfinite(field).all()
        lattice=((points['fixed']+.5)+pf)/ratio-.5
        predicted=(lattice+sample(field,lattice)+.5)*ratio-pm-.5
        expected={'native_moving_pixels':np.linalg.norm(predicted-points['moving'],axis=1),
            'canvas_pixels':np.linalg.norm((predicted-points['moving'])*np.array(layout['moving']['effective_original_to_canvas_scale_xy']),axis=1)}
        score_error={}
        for unit,values in expected.items():
            metrics=score['metrics'][unit];recorded=np.array([metrics['per_label'][key] for key in ids]);difference=float(abs(values-recorded).max())
            assert difference<(.01 if unit=='native_moving_pixels' else .001),(name,unit,difference)
            assert abs(recorded.mean()-metrics['mean'])<1e-12 and abs(np.percentile(recorded,90)-metrics['p90'])<1e-12 and max(recorded)==metrics['maximum']
            errors['native_score' if unit=='native_moving_pixels' else 'canvas_score']=max(errors['native_score' if unit=='native_moving_pixels' else 'canvas_score'],difference);score_error[unit]=difference
        topo=topology(field)
        for key,value in topo.items():assert value==score['topology'][key],(name,key)
        assert score['topology']['local_orientation_valid']==(topo['nonpositive_corners']==0) and not score['topology']['global_homeomorphism_certified']
        checked.append(dict(name=name,labels=len(ids),theta_error=theta_error,cast_initializer_canvas=cast_error,original_csv_float64_score_errors=score_error,
            topology=topo,canvas_metrics={k:score['metrics']['canvas_pixels'][k] for k in ('mean','p90','maximum')}))
        print(json.dumps(dict(checked=name,labels=len(ids),canvas_score_error=score_error['canvas_pixels'],nonpositive_corners=topo['nonpositive_corners'])),flush=True)
    assert labels==2074
    result=dict(all25_original_affines_layouts_configs_and_saved_fields_checked=True,all_original_CSV_errors_recomputed=labels,maximum_errors=errors,
        actual_initial_GPU_field_was_not_exported='Stored theta verified independently; online all-node audit inspected, not falsely called a re-read of unavailable initial field',
        field_bytes=field_bytes,minimum_saved_field_corner_ratio=min(r['topology']['minimum_corner_ratio'] for r in checked),
        nonpositive_corner_cases=sum(r['topology']['nonpositive_corners']>0 for r in checked),total_nonpositive_corners=sum(r['topology']['nonpositive_corners'] for r in checked),rows=checked)
    (DIRECTORY/'independent_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

def comparison_check():
    report=read(DIRECTORY/'independent_check.json');comparison=read(DIRECTORY/'comparison.json')
    assert report['all_original_CSV_errors_recomputed']==2074
    oldmiit=BASE/'miit_dhr_released_standard_t19';oldexisting=BASE/'native22_dhr_standard_t20';fusion=BASE/'match_fusion_all50_t23/fusion'
    scores={
        'native_shared_init':read(DIRECTORY/'miit_scores.json')['rows']+read(DIRECTORY/'existing_scores.json')['rows'],
        'native_own_init':read(oldmiit/'landmark_scores_complete.json')['rows']+read(oldexisting/'landmark_scores.json')['rows'],
        'fusion300':[dict(name=r['name'],**r['methods']['analytic']) for r in read(fusion/'miit_scores.json')['rows']]+read(fusion/'existing_scores.json')['rows']}
    manifests={'native_shared_init':read(DIRECTORY/'predictions.json')['rows'],'native_own_init':read(oldmiit/'predictions.json')['rows']+read(oldexisting/'predictions.json')['rows'],'fusion300':read(fusion/'predictions.json')['rows']}
    names=[r['name'] for r in report['rows']];contrast={'shared_minus_own':('native_shared_init','native_own_init'),'shared_minus_fusion':('native_shared_init','fusion300'),'fusion_minus_own':('fusion300','native_own_init')}
    metric={a:{r['name']:r['metrics']['canvas_pixels'] for r in values} for a,values in scores.items()}
    for a,values in scores.items():assert [r['name'] for r in values]==names and all(r['status']=='ok' for r in values)
    for index,row in enumerate(comparison['rows']):
        name=row['name'];assert name==names[index]
        ids=scores['native_shared_init'][index]['available_pair_labels']
        for a,values in scores.items():
            assert sorted(values[index]['metrics']['canvas_pixels']['per_label'])==ids
            assert row['scored_landmarks']==len(ids)
        for key in ('mean','p90','maximum'):
            for a in scores:assert row['metrics'][key]['values'][a]==metric[a][name][key]
            for label,(a,b) in contrast.items():assert row['metrics'][key]['deltas'][label]==metric[a][name][key]-metric[b][name][key]
        oldrow=manifests['native_own_init'][index];newrow=manifests['native_shared_init'][index]
        olddir=oldmiit if index<3 else oldexisting
        oldcfg=read(olddir/oldrow['configuration']);newcfg=read(DIRECTORY/newrow['configuration'])
        assert {k for k in oldcfg if oldcfg[k]!=newcfg[k]}<={'logging_path'}
        for key in ('preprocessed_shape','field_shape','loaded_padded_shape','fixed_original_wh','moving_original_wh'):
            assert oldrow[key]==newrow[key]
        for a in ('native_own_init','native_shared_init'):assert row['native_topology'][a]==scores[a][index]['topology']
        assert row['initializer_audit']==newrow['initial_field_audit']
    groups={c:[r['name'] for r in comparison['rows'] if r['cohort']==c] for c in comparison['cohorts']}
    assert {c:len(n) for c,n in groups.items()}==dict(miit=3,lung_all20=20,histo=1,rat_kidney=1)
    for c,names_in in groups.items():
        for a in scores:
            values=comparison['cohorts'][c]['methods'][a];assert values['scored_pairs']==values['pair_denominator']==len(names_in) and values['failure_count']==0
            for key,item in [('mean_pair_mean','mean'),('mean_pair_p90','p90'),('worst_pair_maximum','maximum')]:
                sequence=[metric[a][n][item] for n in names_in];expected=max(sequence) if item=='maximum' else sum(sequence)/len(sequence)
                assert abs(values['all_directions_canvas_pixels'][key]-expected)<1e-12
        for label,(a,b) in contrast.items():
            for key in ('mean_pair_mean','mean_pair_p90','worst_pair_maximum'):
                assert comparison['cohorts'][c]['deltas'][label][key]==comparison['cohorts'][c]['methods'][a]['all_directions_canvas_pixels'][key]-comparison['cohorts'][c]['methods'][b]['all_directions_canvas_pixels'][key]
    for label,(a,b) in contrast.items():
        for key in ('mean','p90','maximum'):
            delta={n:metric[a][n][key]-metric[b][n][key] for n in names};actual=comparison['regressions'][label][key]
            assert actual['denominator']==actual['compared']==25 and actual['equal']==sum(v==0 for v in delta.values())
            assert actual['regressed']==sum(v>0 for v in delta.values()) and actual['improved']==sum(v<0 for v in delta.values())
            assert actual['regression_rows']==sorted([dict(name=n,delta=v) for n,v in delta.items() if v>0],key=lambda r:r['delta'],reverse=True)
    def stats(value,sequence,*,memory=False):
        present=[float(v) for v in sequence if v is not None]
        assert value['recorded_count']==len(present) and value['denominator']==25 and value['missing_count']==25-len(present)
        if present:
            assert value['minimum']==min(present) and value['maximum']==max(present) and abs(value['mean']-sum(present)/len(present))<1e-8
            assert value['total'] is None if memory else abs(value['total']-sum(present))<1e-8
        else:assert value['mean'] is value['minimum'] is value['maximum'] is value['total'] is None
    for a,rows in manifests.items():
        stats(comparison['costs'][a]['complete_calls'],[r.get('complete_call_seconds') for r in rows])
        stats(comparison['costs'][a]['peak_allocated_bytes'],[r.get('peak_allocated_bytes') for r in rows],memory=True)
        for field,value in comparison['costs'][a]['phase_seconds'].items():stats(value,[r.get(field) for r in rows])
        if a!='fusion300':
            topo=comparison['native_topology'][a];actual=[r['topology'] for r in scores[a]]
            assert topo['locally_nonpositive_pairs']==sum(r['nonpositive_corners']>0 for r in actual)
            assert topo['nonpositive_corners']==sum(r['nonpositive_corners'] for r in actual) and topo['checked_corners']==sum(r['checked_corner_count'] for r in actual)
            assert topo['minimum_corner_ratio']==min(r['minimum_corner_ratio'] for r in actual) and not topo['global_homeomorphism_certified'] and not topo['boundary_injectivity_checked']
    audits=[r['initial_field_audit'] for r in manifests['native_shared_init']];queries=[r['accepted512_query_audit'] for r in audits];a=comparison['initializer_consistency']
    assert a['full_preprocessed_nodes']==sum(r['preprocessed_nodes'] for r in audits) and a['all512_query_count']==25*512**2
    assert a['outside_query_count']==sum(r['native_lattice_outside_count'] for r in queries) and a['queries_dropped']==0
    for field,source in [('analytic_theta64_max_canvas_pixels','analytic_theta64_max_canvas_pixels'),('cast_theta32_max_canvas_pixels','cast_theta32_max_canvas_pixels'),('inside_max_canvas_pixels','literal_border_sample_inside_max_canvas_pixels'),('outside_max_canvas_pixels','literal_border_sample_outside_max_canvas_pixels')]:
        assert a[field]==max(r[source] for r in queries if r[source] is not None)
    assert abs(a['audit_seconds']-sum(r['audit_seconds'] for r in audits))<1e-10
    report.update(comparison_percase_cohort_regressions_costs_checked=True,cohorts=comparison['cohorts'],regressions=comparison['regressions'],costs=comparison['costs'],initializer_consistency=a)
    (DIRECTORY/'independent_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(comparison_verified=True,shared_mean_regressions_vs_own=comparison['regressions']['shared_minus_own']['mean']['regressed'],shared_mean_regressions_vs_fusion=comparison['regressions']['shared_minus_fusion']['mean']['regressed']),indent=2))

if __name__=='__main__':comparison_check() if '--comparison-only' in sys.argv else main()
