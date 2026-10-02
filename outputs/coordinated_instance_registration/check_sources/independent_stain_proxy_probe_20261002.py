"""Independent scalar OD, order-statistic, area and affine-order checks; no GT."""
import json
import copy
import csv
import tempfile
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools import coordinated_real_case as application
from tools.coordinated_stain_proxy import hematoxylin_proxy
torch.set_num_threads(1)


def linear_quantile(values,q):
    values=sorted(map(float,np.asarray(values).reshape(-1)))
    if not values:return 0.
    index=q*(len(values)-1);left=int(np.floor(index));right=int(np.ceil(index))
    return values[left]+(index-left)*(values[right]-values[left])


def oracle(rgb,support):
    red,green,blue=np.moveaxis(-np.log(np.maximum(np.asarray(rgb,dtype=np.float64),1e-6)),-1,0)
    concentration=(.7095*red-.0249*green-.2274*blue)/.377799
    h=-np.expm1(-np.maximum(concentration,0))
    scale=linear_quantile(h[support],.99)
    return np.clip(h/max(scale,1e-6),0,1).astype(np.float32),scale


def absorption(h,e,d):
    return np.exp(-np.stack((.65*h+.07*e+.27*d,.70*h+.99*e+.57*d,.29*h+.11*e+.78*d),-1))


def scalar_checks():
    rng=np.random.default_rng(5349);rgb=rng.uniform(0,1,(23,29,3));support=rng.random((23,29))>.27
    rgb[0,:4]=[[1,1,1],[0,0,0],[1,1,0],[1,0,1]]
    expected,scale=oracle(rgb,support);actual,diag=hematoxylin_proxy(rgb,support)
    error=float(np.abs(expected-actual).max());assert error<2e-7 and abs(scale-diag['s_raw'])<2e-15
    h=rng.uniform(0,3,(23,29));wanted=-np.expm1(-h);wanted=np.clip(wanted/linear_quantile(wanted[support],.99),0,1)
    beer_errors=[]
    for e,d in [(np.zeros_like(h),np.zeros_like(h)),(h*.9+.2,h*.2+.5),(h*.1,h*.7)]:
        got,_=hematoxylin_proxy(absorption(h,e,d),support);beer_errors.append(float(np.abs(got-wanted).max()))
        assert beer_errors[-1]<1e-7
    pure_max=[]
    for e,d in [(h,h*0),(h*0,h)]:
        got,_=hematoxylin_proxy(absorption(h*0,e,d),support);pure_max.append(float(got.max()));assert pure_max[-1]<2e-9
    white=np.ones((20,20,3));ones=np.ones((20,20),bool)
    got,diag=hematoxylin_proxy(white,ones);assert not got.any() and diag['s']==1e-6
    white[2,7]=absorption(np.array(.4),np.array(.2),np.array(.7))
    got,diag=hematoxylin_proxy(white,ones);assert np.count_nonzero(got)==1 and got[2,7]==1 and diag['s_raw']==0
    empty,diag=hematoxylin_proxy(white,np.zeros_like(ones));expected,_=oracle(white,np.zeros_like(ones))
    assert np.array_equal(empty,expected) and diag['s_raw']==0 and diag['h_zero_fraction'] is None
    return dict(scalar_maximum_error=error,beer_maximum_error=max(beer_errors),pure_other_stain_max=max(pure_max),sparse_zero_quantile=True,empty_formula=True)


def area(image,side):
    ratio=image.shape[0]//side
    return image.reshape(side,ratio,side,ratio).mean((1,3))


def independent_affine(image,a,b):
    side=image.shape[0];out=np.zeros_like(image)
    for y in range(side):
        for x in range(side):
            q=a@((np.array([x,y])+.5)/side)+b
            # Production freezes geometry in double, then supplies float32 grid.
            grid=(2*q-1).astype(np.float32)
            p=((grid+np.float32(1))*np.float32(side)-np.float32(1))/np.float32(2)
            lo=np.floor(p).astype(int);fraction=p-lo
            for dy in (0,1):
                for dx in (0,1):
                    xx,yy=lo+[dx,dy]
                    if 0<=xx<side and 0<=yy<side:
                        out[y,x]+=image[yy,xx]*(fraction[0] if dx else 1-fraction[0])*(fraction[1] if dy else 1-fraction[1])
    return out


def integration_checks():
    rng=np.random.default_rng(144);images=[];expected=[];scales=[]
    with tempfile.TemporaryDirectory(prefix='independent_h_proxy_',dir=ROOT/'outputs/coordinated_instance_registration') as directory:
        paths=[]
        for role in ('fixed','moving'):
            rgb=rng.integers(18,254,(512,512,3),dtype=np.uint8);rgb[:70]=255;rgb[80:100]=[255,255,5]
            path=Path(directory)/(role+'.png');Image.fromarray(rgb).save(path);paths.append(path);images.append(rgb)
        raw=application.load_registration_evidence(*paths,512)
        explicit=application.load_registration_evidence(*paths,512,preprocessing='raw_inverted')
        assert all(torch.equal(x,y) for x,y in zip(raw[:3],explicit[:3])) and raw[3]==explicit[3]
        for rgb in images:
            gray=1-np.asarray(Image.fromarray(rgb).convert('L'),dtype=np.float32)/255
            expected.append(oracle(rgb.astype(np.float64)/255,gray>.04)[0]);scales.append(oracle(rgb.astype(np.float64)/255,gray>.04)[1])
        h=application.load_registration_evidence(*paths,512,preprocessing='hematoxylin_proxy')
        assert torch.equal(raw[2],h[2])
        for index,key in enumerate(('fixed','moving')):
            assert np.abs(h[index].numpy()[0,0]-expected[index]).max()<2e-7
            assert abs(h[3][key]['s_raw']-scales[index])<2e-15
        side=32;fixed=F.interpolate(h[0],size=(side,side),mode='area');moving=F.interpolate(h[1],size=(side,side),mode='area')
        mask=F.interpolate(h[2],size=(side,side),mode='area')
        reduced=area(expected[1],side);assert np.abs(moving.numpy()[0,0]-reduced).max()<6e-7
        assert np.array_equal(mask.numpy()[0,0],area(raw[2].numpy()[0,0],side))
        # Strong non-axis-aligned affine, with genuine partial outside support.
        a=np.array([[.32,-1.1],[.93,.17]]);b=np.array([.88,-.12]);captured=[]
        original=application.self_similarity
        def capture(value):
            captured.append(value.detach().clone());return original(value)
        application.self_similarity=capture
        try:
            evidence=application.Evidence(fixed,moving,torch.from_numpy(a),torch.from_numpy(b),'mind',3.,1.,fixed_mask=mask,interpolation='p1_ac',mind_frame='shared_affine')
        finally:application.self_similarity=original
        assert len(captured)==2 and torch.equal(captured[0],fixed) and torch.equal(evidence.mask,mask)
        expected_warp=independent_affine(moving.numpy()[0,0],a,b)
        error=float(np.abs(captured[1].numpy()[0,0]-expected_warp).max());assert error<2e-6
        assert torch.equal(evidence.matrix,torch.from_numpy(a)) and torch.equal(evidence.offset,torch.from_numpy(b))
        assert float(moving.max())<.95 and float(captured[1].max())<.95
        assert (captured[1]==0).any() and float(evidence.denominator)==float(mask.sum())
    return dict(raw_default_bitwise=True,raw_mask_and_area_exact=True,normalized512_then_area_then_affine=True,affine_sampler_maximum_error=error,no_level_or_warp_renormalization=True)


def protocol_checks():
    from tools.coordinated_stain_proxy import source_cases,configuration,write_scoring_manifests
    base=ROOT/'outputs/coordinated_instance_registration'
    rows=source_cases(base/'miit_shared_affine_pilot_t17',base/'existing22_shared_affine_a300_t20')
    assert len(rows)==25 and len({r['name'] for r in rows})==25
    originals=copy.deepcopy(rows)
    for row in rows:
        original=row['original_configuration'];updated=vars(configuration(original,'new.npz'))
        assert set(original)==set(updated)
        differences=[]
        for key in original:
            old=Path(original[key]) if isinstance(updated[key],Path) else original[key]
            if old!=updated[key]:differences.append(key)
        assert set(differences)=={'preprocessing','output'}
    assert rows==originals
    with tempfile.TemporaryDirectory(prefix='independent_h_protocol_',dir=base) as directory:
        output=Path(directory);failing=copy.deepcopy(rows)
        for row in failing:row.update(status='failed',error='deliberate checker fixture',output=row['name']+'.npz')
        manifest=dict(prediction_complete=True,annotations_read=False,rows=failing,protocol='checker fixture',changed_variables=['preprocessing','output'])
        pending=copy.deepcopy(manifest);pending['rows'][-1]['status']='pending'
        try:write_scoring_manifests(pending,output)
        except ValueError:pass
        else:raise AssertionError('scoring adapter accepted nonterminal case25')
        assert not list(output.iterdir())
        write_scoring_manifests(manifest,output)
        miit=json.loads((output/'miit_predictions.json').read_text());existing=json.loads((output/'existing_predictions.json').read_text())
        assert len(miit['rows'])==3 and len(existing['rows'])==22
        assert all(r['status']=='failed' for r in existing['rows'])
        for value in (miit,existing):assert value['all25_terminal'] and not value['annotations_read']
        source_path=Path(rows[0]['source_manifest']);source=json.loads(source_path.read_text())
        def same_path(original,updated):
            a=Path(original);b=Path(updated)
            assert (a if a.is_absolute() else source_path.parent/a).resolve()==(b if b.is_absolute() else output/b).resolve()
        for old,new in zip(source['rows'],miit['rows']):
            assert new['methods']['analytic']['status']=='failed'
            for key in ('fixed','moving','affine','layout'):same_path(old[key],new[key])
            same_path(old['raw_matches']['path'],new['raw_matches']['path'])
            same_path(old['affine_estimation']['report'],new['affine_estimation']['report'])
            for method in ('f2','dhr'):
                for key in ('output','report','output_directory','field','configuration','postprocessing_params'):
                    if isinstance(old['methods'][method].get(key),str):same_path(old['methods'][method][key],new['methods'][method][key])
    return dict(actual_controls=25,configuration_delta=['preprocessing','output'],pending25_blocks_scoring=True,all_failed_denominators=[3,22],archived_miit_paths_preserved=True)


def literal_p1(vertices,query):
    n=vertices.shape[0]-1;cell=np.minimum(np.floor(query*n).astype(int),n-1);uv=query*n-cell
    result=[]
    for (x,y),point in zip(cell,uv):
        triangle=[(0,0),(1,0),(1,1)] if point[1]<=point[0] else [(0,0),(1,1),(0,1)]
        weights=np.linalg.solve(np.vstack((np.array(triangle).T,np.ones(3))),np.r_[point,1.])
        result.append(sum(w*vertices[y+dy,x+dx] for w,(dx,dy) in zip(weights,triangle)))
    return np.array(result)


def postrun_checks(directory_name='stain_proxy_all25_t21',point_substitution=False):
    base=ROOT/'outputs/coordinated_instance_registration';directory=base/directory_name
    data=Path('D:/QC_optimization_data/digital_topology_wsi')
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    manifest=read(directory/'predictions.json')
    assert manifest['prediction_complete'] and not manifest['annotations_read'] and len(manifest['rows'])==25
    assert all(r['status']=='ok' and r['budget_complete'] for r in manifest['rows'])
    scores={r['name']:r for r in read(directory/'miit_scores.json')['rows']+read(directory/'existing_scores.json')['rows']}
    control_m=read(base/'miit_multiscale_control_t19/predictions.json')
    control_e=read(base/'existing22_shared_affine_a300_t20/predictions.json')
    controls={r['name']:(base/'miit_multiscale_control_t19'/r['methods']['analytic']['report']) for r in control_m['rows']}
    controls.update({r['name']:base/'existing22_shared_affine_a300_t20'/r['report'] for r in control_e['rows']})
    def csv_points(path):
        with path.open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream));columns=list(rows[0]);x='x' if 'x' in columns else 'X';y='y' if 'y' in columns else 'Y'
            label='label' if 'label' in columns else next(k for k in columns if k not in (x,y))
            return {r[label].strip():np.array([float(r[x]),float(r[y])]) for r in rows}
    miit_root=Path('D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')
    stains=dict(he='He',cc10='Cc10-5',cd31='CD31-3',ki67='Ki67-7',prospc='proSPC-4')
    def case_data(name):
        if name.startswith('miit_'):
            moving,fixed=map(int,name.removeprefix('miit_').split('_to_'))
            layout=read(base/'miit_three_rotations_t153'/(name+'_layout.json'))
            points={role:csv_points(miit_root/str(section)/'landmarks'/f'{section:02}.csv') for role,section in [('fixed',fixed),('moving',moving)]}
            assert len(points['fixed'])==len(points['moving'])==124
        elif name in ('histo','rat_kidney'):
            layout=read(data/'birl_anhir_dev/canvas'/(name+'_layout.json'))
            folder=data/('HistoReg_CD68_CD4' if name=='histo' else 'birl_anhir_dev/labels_eval_only/rat-kidney_/scale-5pc')
            filenames=('Landmarks_CD4.csv','Landmarks_CD68.csv') if name=='histo' else ('Rat-Kidney_HE.csv','Rat-Kidney_PanCytokeratin.csv')
            points={role:csv_points(folder/file) for role,file in zip(('fixed','moving'),filenames)}
            assert set(points['fixed'])-set(points['moving'])==({'70','71'} if name=='rat_kidney' else set())
        else:
            layout={};points={}
            for role,stain in zip(('fixed','moving'),name.split('_to_')):
                layout[role]=read(data/'lung_lesion3_eval/canvas'/(('cc10' if stain=='he' else stain)+'_layout.json'))['fixed' if stain=='he' else 'moving']
                raw=csv_points(data/'lung_lesion3_eval/annotations50'/('29-041-Izd2-w35-'+stains[stain]+'-les3.csv'))
                points[role]={k:(p+.5)/10-.5 for k,p in raw.items()}
            assert len(points['fixed'])==len(points['moving'])==80
        ids=sorted(k for k in set(points['fixed'])&set(points['moving']) if np.isfinite(points['fixed'][k]).all() and np.isfinite(points['moving'][k]).all())
        return layout,{role:np.stack([points[role][k] for k in ids]) for role in ('fixed','moving')},ids
    def image_path(name,role,configuration):
        filename=Path(configuration[role]).name
        if name.startswith('miit_'):return base/'miit_three_rotations_t153'/filename
        if name in ('histo','rat_kidney'):return data/'birl_anhir_dev/canvas'/filename
        return data/'lung_lesion3_eval/canvas'/filename
    cache={};output=[];weak=None
    for row in manifest['rows']:
        name=row['name'];report=read(directory/row['report']);cfg=report['configuration'];original=read(controls[name])['configuration']
        changed_keys={'matches','output'} if point_substitution else {'preprocessing','output'}
        assert set(original)==set(cfg) and {k for k in cfg if cfg[k]!=original[k]}==changed_keys
        assert report['gradient_steps']==300 and report['failed_trials']==0 and report['objective_evaluations']==332
        assert report['saved_binary_certificate']['valid'] and not report['landmarks_used']
        assert report['final']['total']==min([report['initial']['total']]+[r['accepted_full_total'] for r in report['stages']])
        with np.load(directory/row['output'],allow_pickle=False) as saved:
            v=saved['vertices'][0];ref=saved['boundary_reference'][0];a=saved['post_affine_matrix'].astype(float);b=saved['post_affine_offset'].astype(float)
            assert str(saved['interpolation'])=='p1_ac' and v.shape==(257,257,2) and v.dtype==np.float64
        with np.load(controls[name].with_suffix('.npz'),allow_pickle=False) as control_map:
            assert np.array_equal(a,control_map['post_affine_matrix']) and np.array_equal(b,control_map['post_affine_offset'])
        assert np.isfinite(v).all() and np.linalg.det(a)>0
        axis=np.arange(257)/256;yy,xx=np.meshgrid(axis,axis,indexing='ij');identity=np.stack((xx,yy),-1)
        assert np.array_equal(ref,identity)
        assert all(np.array_equal(x,y) for x,y in [(v[0],ref[0]),(v[-1],ref[-1]),(v[:,0],ref[:,0]),(v[:,-1],ref[:,-1])])
        va,vb,vc,vd=v[:-1,:-1],v[:-1,1:],v[1:,1:],v[1:,:-1]
        det=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
        minimum=float(np.stack((det(vb-va,vd-va),det(vb-va,vc-va),det(vb-vd,vc-vd),det(vc-va,vd-va))).min()*256**2)
        assert minimum>.001 and abs(minimum-row['actual_minimum_corner_ratio'])<2e-14
        layout,points,ids=case_data(name);score=scores[name];assert ids==score['available_pair_labels']
        unit=lambda p,l:((p+.5)*l['effective_original_to_canvas_scale_xy']+l['padding_xy'])/512
        query=unit(points['fixed'],layout['fixed']);predicted=literal_p1(v@a.T+b,query)
        errors={'canvas_pixels':np.linalg.norm((predicted-unit(points['moving'],layout['moving']))*512,axis=1),
                'native_moving_pixels':np.linalg.norm((predicted*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5-points['moving'],axis=1)}
        metrics=score['methods']['analytic']['metrics'] if name.startswith('miit_') else score['metrics']
        discrepancies={}
        for key,values in errors.items():
            discrepancies[key]=max(abs(values[i]-metrics[key]['per_label'][label]) for i,label in enumerate(ids))
            assert discrepancies[key]<1e-9
            assert abs(values.mean()-metrics[key]['mean'])<1e-10 and abs(np.percentile(values,90)-metrics[key]['p90'])<1e-10
        if point_substitution:
            control_report=read(controls[name])
            assert report['image_preprocessing']['name']=='raw_inverted'
            assert report['image_preprocessing']==control_report['image_preprocessing']
            assert report['mind_frame_by_resolution']==control_report['mind_frame_by_resolution']
            output.append(dict(name=name,labels=len(ids),minimum_corner_ratio=minimum,score_discrepancy=discrepancies,mean=metrics['canvas_pixels']['mean'],p90=metrics['canvas_pixels']['p90']))
            continue
        metadata=report['image_preprocessing'];assert metadata['name']=='hematoxylin_proxy'
        for role in ('fixed','moving'):
            path=image_path(name,role,cfg)
            if path not in cache:
                image=Image.open(path);rgb=np.asarray(image,dtype=np.float64)/255;gray=1-np.asarray(image.convert('L'),dtype=np.float32)/255;support=gray>.04
                feature,scale=oracle(rgb,support);od=-np.log(np.maximum(rgb,1e-6));ch=(.7095*od[...,0]-.0249*od[...,1]-.2274*od[...,2])/.377799
                cache[path]=(feature,support,scale,float((ch[support]<=0).mean()))
            feature,support,scale,zeros=cache[path];diagnostic=metadata[role]
            assert diagnostic['original_support_pixels']==int(support.sum()) and abs(diagnostic['s_raw']-scale)<2e-14
            assert diagnostic['s']==max(scale,1e-6) or abs(diagnostic['s']-max(scale,1e-6))<2e-14
            assert abs(diagnostic['h_zero_fraction']-zeros)<1e-14 and not diagnostic['numerical_floor_active']
            if role=='fixed':
                for side in (32,64,128,256,512):
                    frame=report['mind_frame_by_resolution'][str(side)]
                    assert frame['fixed_mask_denominator']==float(support.sum())*(side/512)**2
                    assert frame['intensity_prewarp_count']==frame['descriptor_construction_count']==1
            if name=='miit_7_to_8' and role=='moving':
                variance_errors=[]
                for side in (32,64,128,256,512):
                    f=area(feature,side).astype(np.float64);weights=area(support.astype(np.float32),side);pad=np.pad(f,2,mode='edge');distances=[]
                    for dx,dy in [(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2)]:
                        squared=(f-pad[2+dy:2+dy+side,2+dx:2+dx+side])**2;z=np.pad(squared,1);count=np.pad(np.ones_like(squared),1)
                        distances.append(sum(z[y:y+side,x:x+side] for y in range(3) for x in range(3))/sum(count[y:y+side,x:x+side] for y in range(3) for x in range(3)))
                    variance=np.mean(distances,axis=0);median=linear_quantile(variance[weights>0],.5);incidence=float((weights*(variance<=1e-4)).sum()/weights.sum())
                    d=diagnostic['unwarped_normalized_mind_by_resolution'][str(side)]
                    variance_errors.append(abs(median-d['positive_support_linear_variance_median']))
                    assert variance_errors[-1]<2e-7 and abs(incidence-d['original_mask_weighted_fraction_at_or_below_epsilon'])<1e-7
                weak=dict(raw_H_q99=scale,zero_fraction=zeros,normalized512_variance_median=diagnostic['unwarped_normalized_mind_by_resolution']['512']['positive_support_linear_variance_median'],normalized512_below_epsilon_fraction=diagnostic['unwarped_normalized_mind_by_resolution']['512']['original_mask_weighted_fraction_at_or_below_epsilon'],independent_variance_median_maximum_error=max(variance_errors))
        output.append(dict(name=name,labels=len(ids),minimum_corner_ratio=minimum,score_discrepancy=discrepancies,mean=metrics['canvas_pixels']['mean'],p90=metrics['canvas_pixels']['p90']))
    assert sum(r['labels'] for r in output)==2074 and (point_substitution or len(cache)==15)
    groups={'miit':output[:3],'lung20':output[3:23],'histo':output[23:24],'rat_kidney':output[24:25]}
    return dict(all25complete300=True,all25_objective_calls=332,all25_same_recipe_two_key_delta=True,total_landmarks=2074,unique_original_canvas_images=len(cache),minimum_corner_ratio=min(r['minimum_corner_ratio'] for r in output),maximum_canvas_score_error=max(r['score_discrepancy']['canvas_pixels'] for r in output),maximum_native_score_error=max(r['score_discrepancy']['native_moving_pixels'] for r in output),cohorts={k:{metric:float(np.mean([r[metric] for r in group])) for metric in ('mean','p90')} for k,group in groups.items()},weak_miit7moving=weak)


if __name__=='__main__':
    print(json.dumps(postrun_checks() if '--postrun' in sys.argv else dict(scalar=scalar_checks(),integration=integration_checks(),protocol=protocol_checks()),indent=2))
