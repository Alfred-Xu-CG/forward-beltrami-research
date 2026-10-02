"""Independent fixed point-table coordinates, robust P1 value/VJP; no model or GT."""
import json
import tempfile
import importlib.util
import logging
import types
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.coordinated_real_case import load_image_matches
torch.set_num_threads(1)


def literal_value_gradient(vertices,source,target,confidence,a,b):
    world=target@a.T+b;eligible=((world>=0)&(world<=1)).all(-1)
    weights=confidence*eligible;weights/=weights.sum()
    total=0.;gradient=np.zeros_like(vertices);n=vertices.shape[0]-1
    for q,p,w in zip(source,target,weights):
        cell=np.minimum(np.floor(q*n).astype(int),n-1);local=q*n-cell;x,y=cell
        nodes=[(0,0),(1,0),(1,1)] if local[1]<=local[0] else [(0,0),(1,1),(0,1)]
        bary=np.linalg.solve(np.vstack((np.array(nodes).T,np.ones(3))),np.r_[local,1.])
        mapped=sum(lam*vertices[y+dy,x+dx] for lam,(dx,dy) in zip(bary,nodes));delta=mapped-p
        error=64*delta@a.T;length=np.sqrt(1+error@error);total+=w*(length-1)
        derivative=w*64**2*delta@a.T@a/length
        for lam,(dx,dy) in zip(bary,nodes):gradient[y+dy,x+dx]+=lam*derivative
    return total,gradient,weights


def point_loss_check():
    rng=np.random.default_rng(341);source=rng.uniform(.02,.98,(32,2));target=np.clip(source+rng.normal(0,.08,(32,2)),0,1)
    source[:4]=[[0,0],[1,0],[0,1],[1,1]];target[:4]=source[:4]
    confidence=np.linspace(.15,1,32);confidence[7]=0
    a=np.array([[.7,-.8],[.65,.85]]);b=np.array([.39,-.13])
    axis=np.linspace(0,1,9);yy,xx=np.meshgrid(axis,axis,indexing='ij');vertices=np.stack((xx,yy),-1)
    vertices[1:-1,1:-1]+=rng.normal(0,.008,(7,7,2))
    expected,gradient,weights=literal_value_gradient(vertices,source,target,confidence,a,b)
    assert np.count_nonzero(weights)>=8 and np.count_nonzero(weights)<len(weights)
    record=dict(source_points_unit=source.tolist(),target_points_unit=target.tolist(),confidence=confidence.tolist(),
                post_affine_matrix=a.tolist(),post_affine_offset=b.tolist(),targets_manual_landmarks_or_dense_teacher_loaded=False,
                image_side=512,fixed='fixed.png',moving='moving.png')
    with tempfile.TemporaryDirectory(prefix='independent_matchanything_',dir=ROOT/'outputs/coordinated_instance_registration') as directory:
        path=Path(directory)/'matches.json';path.write_text(json.dumps(record),encoding='utf-8')
        matches,metadata=load_image_matches(path,a,b,fixed_path=Path('fixed.png'),moving_path=Path('moving.png'),image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
        assert np.allclose(matches.weights.numpy(),weights,rtol=0,atol=2e-17)
        value_errors=[];vjp_errors=[]
        for frozen in (False,True):
            if frozen:matches.prepare_fixed_p1_sampling(9,9,'ac')
            tensor=torch.tensor(vertices[None],requires_grad=True);value=matches(tensor,torch.from_numpy(a),'p1_ac');actual=torch.autograd.grad(value,tensor)[0].numpy()[0]
            value_errors.append(abs(float(value.detach())-expected));vjp_errors.append(float(np.abs(actual-gradient).max()))
            assert value_errors[-1]<1e-13 and vjp_errors[-1]<1e-12
        direction=rng.normal(0,1,vertices.shape);step=1e-6
        finite=(literal_value_gradient(vertices+step*direction,source,target,confidence,a,b)[0]-literal_value_gradient(vertices-step*direction,source,target,confidence,a,b)[0])/(2*step)
        assert abs(finite-np.sum(gradient*direction))<2e-7
        duplicate={**record,**{key:record[key]*3 for key in ('source_points_unit','target_points_unit','confidence')}}
        path.write_text(json.dumps(duplicate),encoding='utf-8')
        repeated,_=load_image_matches(path,a,b,fixed_path=Path('fixed.png'),moving_path=Path('moving.png'),image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
        assert abs(float(repeated(torch.from_numpy(vertices[None]),torch.from_numpy(a),'p1_ac'))-expected)<1e-13
        insufficient={**record,'confidence':[1. if i<7 else 0. for i in range(32)]}
        path.write_text(json.dumps(insufficient),encoding='utf-8')
        try:load_image_matches(path,a,b,fixed_path=Path('fixed.png'),moving_path=Path('moving.png'),image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
        except ValueError:pass
        else:raise AssertionError('fewer than8positiveeligible accepted')
    return dict(raw_points=len(source),positive_eligible=int(np.count_nonzero(weights)),value_maximum_error=max(value_errors),VJP_maximum_error=max(vjp_errors),directional_finite_difference_error=abs(finite-np.sum(gradient*direction)),replication_keeps_total_strength=True,insufficient_case_fails=True)


def adapter_checks():
    from tools.coordinated_matchanything import point_record,FrozenMatchAnything,read_matcher_gray
    rng=np.random.default_rng(714);q=rng.uniform(.1,.9,(30,2));p=rng.uniform(.2,.8,(30,2))
    q[:4]=[[0,0],[1,0],[0,1],[1,1]];p[:4]=q[:4];confidence=np.linspace(.01,1,30);confidence[7]=0
    q[-2,0]=-1e-6;p[-1,1]=1.000001
    a=np.array([[.5,-.75],[.75,.5]],dtype=np.float32);b=np.array([.625,-.125],dtype=np.float32)
    p0=q*512-.5;p1=p*512-.5;inside=np.array([all(0<=v<=1 for v in list(x)+list(y)) for x,y in zip(q,p)])
    # Preserve the arithmetic of the actual raw-pixel fixture, including boundary values.
    source=(p0[inside]+.5)/512;target=(p1[inside]+.5)/512;c=confidence[inside]
    world=target@a.astype(float).T+b;positive=np.array([all(0<=v<=1 for v in xy) and cf>0 for xy,cf in zip(world,c)])
    support=np.indices((512,512)).sum(axis=0)%7==0
    record=point_record(p0,p1,confidence,a,b,fixed_support=support)
    assert np.array_equal(record['source_points_unit'],source) and np.array_equal(record['target_points_unit'],target)
    assert record['confidence']==c.tolist() and record['discarded_out_of_unit_domain']==2
    assert record['eligible_matches']==int(positive.sum())>=8 and record['status']=='ok'
    assert np.any(c<.1) and 0. in c and record['raw_matches']==28
    assert record['source_points_unit'][0]==[0.,0.] and record['source_points_unit'][3]==[1.,1.]
    for values in [(np.where(np.arange(p0.size).reshape(p0.shape)==0,np.nan,p0),p1,confidence),
                   (p0,p1,np.where(np.arange(30)==29,np.inf,confidence)),
                   (p0,p1,np.where(np.arange(30)==29,-.1,confidence))]:
        try:point_record(*values,a,b)
        except ValueError:pass
        else:raise AssertionError('invalid raw output escaped rejection before domain filtering')
    assert point_record(p0[:7],p1[:7],confidence[:7],a,b)['status']=='insufficient_matches'
    assert point_record(np.empty((0,2)),np.empty((0,2)),np.empty(0),a,b)['status']=='insufficient_matches'
    captured=[]
    class InjectedModel:
        def __call__(self,batch):
            assert torch.is_inference_mode_enabled() and set(batch)=={'image0','image1'}
            captured.extend([batch['image0'].clone(),batch['image1'].clone()])
            for key,value in [('mkpts0_f',p0),('mkpts1_f',p1),('mconf',confidence)]:batch[key]=torch.from_numpy(value)
    with tempfile.TemporaryDirectory(prefix='independent_matchanything_input_',dir=ROOT/'outputs/coordinated_instance_registration') as directory:
        folder=Path(directory);yy,xx=np.indices((512,512));gray=np.round((xx+2*yy)/1533*255).astype(np.uint8)
        moving=np.repeat(gray[...,None],3,axis=-1);fixed=np.stack((gray,np.flipud(gray),np.fliplr(gray)),axis=-1)
        fp=folder/'fixed.png';mp=folder/'moving.png';ap=folder/'affine.npz';out=folder/'matches.json'
        Image.fromarray(fixed).save(fp);Image.fromarray(moving).save(mp);np.savez(ap,post_affine_matrix=a,post_affine_offset=b)
        matcher=object.__new__(FrozenMatchAnything);matcher.device=torch.device('cpu');matcher.model=InjectedModel()
        report=matcher.extract(fp,mp,ap,output=out);matcher.close()
        expected_fixed=np.asarray(Image.fromarray(fixed).convert('L'),dtype=np.float32)/255
        assert np.array_equal(captured[0].numpy()[0,0],expected_fixed)
        assert np.array_equal(read_matcher_gray(mp).numpy()[0,0],gray.astype(np.float32)/255)
        coordinates=np.stack(((xx+.5)/512,(yy+.5)/512),-1);original=coordinates@a.astype(float).T+b
        pixel=np.clip(original*512-.5,0,511);low=np.floor(pixel).astype(int);high=np.minimum(low+1,511);fraction=pixel-low
        image=gray.astype(float)/255;x,y=low[...,0],low[...,1];hx,hy=high[...,0],high[...,1];u,v=fraction[...,0],fraction[...,1]
        expected=(1-u)*(1-v)*image[y,x]+u*(1-v)*image[y,hx]+(1-u)*v*image[hy,x]+u*v*image[hy,hx]
        error=float(np.abs(expected-captured[1].numpy()[0,0]).max());assert error<2e-7
        assert not np.allclose(expected,1-expected) and ((original<0)|(original>1)).any()
        for key in ('source_points_unit','target_points_unit','confidence'):assert report[key]==record[key]
        loaded,metadata=load_image_matches(out,a,b,fixed_path=fp,moving_path=mp,image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
        assert int(torch.count_nonzero(loaded.weights))==record['eligible_matches']
        assert metadata['eligible_matches']==int(((world>=0)&(world<=1)).all(-1).sum())
        assert report['source_original_gray_support_fraction']==float(np.array([(1-expected_fixed)[min(int(s[1]*512),511),min(int(s[0]*512),511)]>.04 for s in source]).mean())
    from tools.coordinated_matchanything_batch import configuration
    from tools.coordinated_stain_proxy import source_cases
    base=ROOT/'outputs/coordinated_instance_registration';cases=source_cases(base/'miit_multiscale_control_t19',base/'existing22_shared_affine_a300_t20')
    for case in cases:
        original=case['original_configuration'];changed=vars(configuration(original,'new_matches.json','new_map.npz'))
        assert set(changed)==set(original)
        assert {k for k in changed if changed[k]!=(Path(original[k]) if isinstance(changed[k],Path) else original[k])}=={'matches','output'}
    return dict(raw_output_count=30,retained_after_domain=28,positive_eligible=record['eligible_matches'],ordinary_PIL_grayscale_exact=True,border_affine_sampler_maximum_error=error,nonfinite_invalid_confidence_case_failure=True,no_second_affine_or_clipping=True,no_extra_confidence_threshold=True,all25_exact_two_key_delta=True)


def released_source_smoke_check():
    folder=Path('D:/QC_optimization_data/digital_topology_wsi/matchanything_eloftr')
    path=folder/'source/src/loftr/utils/coarse_matching.py'
    spec=importlib.util.spec_from_file_location('independent_official_coarse',path);module=importlib.util.module_from_spec(spec)
    # This isolated method fixture does not call logging. No package installation
    # or source edit is needed merely to import its unused optional logger.
    prior=sys.modules.get('loguru');stub=types.ModuleType('loguru');stub.logger=logging.getLogger('coarse-fixture')
    if prior is None:sys.modules['loguru']=stub
    try:spec.loader.exec_module(module)
    finally:
        if prior is None:del sys.modules['loguru']
    cfg=dict(thr=.1,border_rm=2,train_coarse_percent=.1,train_pad_num_gt_min=200,match_type='dual_softmax',dsmax_temperature=.1,mtd_spvs=True,fix_bias=False,force_nearest=True)
    coarse=module.CoarseMatching(cfg).eval();conf=torch.zeros(1,64,64)
    for i,j,value in [(18,18,.9),(18,19,.4),(19,18,.3),(20,20,.1),(0,18,.95)]:conf[0,i,j]=value
    output=coarse.get_coarse_match(conf,dict(hw0_i=(64,64),hw1_i=(64,64),hw0_c=(8,8),hw1_c=(8,8)))
    # Non-mutual neighbors .4 and .3 survive the active MTD branch; exact threshold
    # .1 and coarse-border row0 do not. Preserve this released behavior explicitly.
    assert output['i_ids'].tolist()==[18,18,19] and output['j_ids'].tolist()==[18,19,18]
    assert torch.equal(output['mconf'],torch.tensor([.9,.4,.3]))
    assert output['mkpts0_c'].tolist()==[[16.,16.],[16.,16.],[24.,16.]]
    checkpoint=torch.load(folder/'matchanything_eloftr.ckpt',map_location='cpu',weights_only=True)
    state=checkpoint['state_dict'];assert len(state)==447 and all(torch.is_tensor(x) and torch.isfinite(x).all() for x in state.values())
    setup=json.loads((folder/'setup_checked_cpu.json').read_text());model=setup['model'];actual=model['model_config']
    assert setup['source_revision']=='6a7bcb589ec8da3a9e861e799122beaa5eba2193'
    assert model['strict_keys'] and model['weights_only'] and model['state_dict_entries']==447 and model['parameter_count']==16025216
    assert model['coarse_npe']==[832,832,512,512] and model['inference_dtype']=='float32' and not actual['fp16']
    assert actual['match_coarse']['thr']==.1 and actual['match_coarse']['mtd_spvs'] and actual['match_coarse']['border_rm']==2
    assert actual['match_fine']['topk']==1 and actual['match_fine']['local_regress_nomask'] and actual['fine']['mtd_spvs']
    assert model['configuration_force_nearest'] and 'unused' in model['actual_coarse_match_policy'] and 'force_mutual_nearest' not in model
    smoke=json.loads((folder/'smoke_miit_2_to_3.json').read_text());q=np.array(smoke['source_points_unit']);p=np.array(smoke['target_points_unit']);c=np.array(smoke['confidence'])
    a=np.array(smoke['post_affine_matrix']);b=np.array(smoke['post_affine_offset']);world=p@a.T+b
    eligible=((world>=0)&(world<=1)).all(-1)&(c>0)
    assert np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(c).all()
    assert ((q>=0)&(q<=1)).all() and ((p>=0)&(p<=1)).all() and ((c>.1)&(c<=1)).all()
    assert len(q)==2910==smoke['raw_matches'] and int(eligible.sum())==2908==smoke['eligible_matches']
    assert abs(c[eligible].sum()-smoke['eligible_confidence_mass'])<1e-10
    affine=ROOT/'outputs/coordinated_instance_registration/miit_three_rotations_t153/miit_2_to_3_affine.npz'
    with np.load(affine) as saved:assert np.array_equal(a,saved['post_affine_matrix']) and np.array_equal(b,saved['post_affine_offset'])
    return dict(pinned_source_active_policy='all threshold>.1 coarse pairs after border2 removal; not mutual-nearest',nonmutual_injected_pairs_retained=True,checkpoint_safe_loaded_finite_state_entries=447,smoke_retained=2910,smoke_positive_eligible=2908,smoke_same_affine=True,smoke_unique_sources=len(np.unique(q,axis=0)),smoke_unique_targets=len(np.unique(p,axis=0)))


def actual_postrun():
    from independent_stain_proxy_probe_20261002 import postrun_checks
    result=postrun_checks('matchanything_all25_t22',point_substitution=True)
    result.pop('unique_original_canvas_images');result.pop('weak_miit7moving')
    result['original_gray_preparation_and_all_scale_mask_metadata_exact']=True
    directory=ROOT/'outputs/coordinated_instance_registration/matchanything_all25_t22'
    manifest=json.loads((directory/'predictions.json').read_text())
    assert manifest['extraction_complete'] and manifest['model_released_before_optimization']
    assert manifest['cost_totals']['extraction_calls']==manifest['cost_totals']['optimizer_calls']==25
    assert manifest['model_setup_status']=='ok' and manifest['cost_totals']['model_setup_count']==1
    points=[];source_counts=[];target_counts=[]
    for row in manifest['rows']:
        record=json.loads((directory/row['match_table']).read_text());report=json.loads((directory/row['report']).read_text())
        assert record['status']=='ok' and row['extraction_status']=='ok' and row['optimization_status']=='ok'
        assert not record['targets_manual_landmarks_or_dense_teacher_loaded'] and not record['global_geometric_ransac_used'] and not record['tissue_support_filter_used']
        q=np.array(record['source_points_unit']);p=np.array(record['target_points_unit']);confidence=np.array(record['confidence'])
        assert np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(confidence).all()
        assert ((q>=0)&(q<=1)).all() and ((p>=0)&(p<=1)).all() and ((confidence>.1)&(confidence<=1)).all()
        with np.load(directory/row['output']) as saved:
            a=saved['post_affine_matrix'];b=saved['post_affine_offset']
            assert np.array_equal(record['post_affine_matrix'],a) and np.array_equal(record['post_affine_offset'],b)
        world=p@a.astype(float).T+b;eligible=((world>=0)&(world<=1)).all(-1)&(confidence>0)
        assert len(q)==record['raw_matches'] and int(eligible.sum())==record['eligible_matches']>=8
        assert record['network_matches']==record['raw_matches']+record['discarded_out_of_unit_domain']
        assert abs(confidence[eligible].sum()-record['eligible_confidence_mass'])<1e-10
        for coords,key,counts in [(q,'source',source_counts),(p,'target',target_counts)]:
            _,count=np.unique(coords,axis=0,return_counts=True)
            assert len(count)==record['unique_'+key+'_coordinates'] and int(count.max())==record['maximum_'+key+'_multiplicity']
            counts.append(int(count.max()))
        loader=row['point_loader_metadata']
        assert loader['eligible_matches']==int(eligible.sum()) and abs(loader['confidence_denominator']-confidence[eligible].sum())<1e-10
        assert Path(report['configuration']['matches']).name==row['match_table']
        used=report['image_match_evidence']
        assert Path(used['path']).name==row['match_table'] and used['raw_matches']==len(q) and used['eligible_matches']==int(eligible.sum())
        assert abs(used['confidence_denominator']-confidence[eligible].sum())<1e-10
        if report['configuration'].get('match_p1_sampling','existing')=='frozen':
            assert used['fixed_p1_sampling']['prepared'] and used['fixed_p1_sampling']['queries']==len(q)
        points.append(dict(name=row['name'],raw=len(q),eligible=int(eligible.sum()),confidence_mass=float(confidence[eligible].sum())))
    result.update(point_tables=25,retained_point_range=[min(x['raw'] for x in points),max(x['raw'] for x in points)],positive_eligible_range=[min(x['eligible'] for x in points),max(x['eligible'] for x in points)],largest_source_multiplicity=max(source_counts),largest_target_multiplicity=max(target_counts),one_model_then_all25extract_then_all25opt=True,point_rows=points)
    comparison=json.loads((directory/'comparison.json').read_text())
    assert comparison['pair_denominator']==25 and comparison['prediction_failures']==comparison['scoring_failures']==0
    for name,metric in result['cohorts'].items():
        group=comparison['cohorts']['lung_all20' if name=='lung20' else name]
        for key in ('mean','p90'):assert abs(metric[key]-group['matchanything']['all_directions_canvas_pixels']['mean_pair_'+key])<1e-12
    for input_key,comparison_key in [('extraction_complete_call_seconds','extraction_complete_call_seconds'),('optimization_complete_call_seconds','optimizer_complete_call_seconds')]:
        assert abs(sum(r[input_key] for r in manifest['rows'])-comparison['costs'][comparison_key]['total'])<1e-10
    result['comparison_aggregates_and_phase_call_totals_reproduced']=True
    return result


if __name__=='__main__':
    print(json.dumps(actual_postrun() if '--postrun' in sys.argv else dict(point_functional=point_loss_check(),adapter=adapter_checks(),released_source_smoke=released_source_smoke_check()),indent=2))
