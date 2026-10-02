"""Bounded independent rendering/frame/terminal-hook checks; no labels or GPU."""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests'),str(Path(__file__).parent)]
from tools import coordinated_real_case as app
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_terminal_detail_inputs import input_pair,native_cases,prepare_pair,render_original,decoder_cache
from tools.coordinated_terminal_detail import load_terminal_bundle,validate_terminal_tensors
from tools.coordinated_terminal_detail_batch import configuration as terminal_configuration
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from independent_joint_pose_probe_20261002 import fixture,numpy_image,numpy_priors
from independent_matchanything_probe_20261002 import literal_value_gradient

torch.set_num_threads(1)
BASE=ROOT/'outputs/coordinated_instance_registration'
DATA=Path('D:/QC_optimization_data/digital_topology_wsi')
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def lift_oracle(array):
    """Independent separable half-pixel bilinear enlargement with border clamp."""
    n=len(array);q=(np.arange(2*n)+.5)/2-.5;left=np.floor(q).astype(int);alpha=(q-left).astype(np.float32)
    lo=np.clip(left,0,n-1);hi=np.clip(left+1,0,n-1)
    horizontal=array[:,lo]*(1-alpha)+array[:,hi]*alpha
    return horizontal[lo,:]*(1-alpha[:,None])+horizontal[hi,:]*alpha[:,None]


def actual_layout_and_render_checks(directory):
    cases=source_cases(BASE/'match_fusion_all50_t23/fusion');native=native_cases(BASE/'native22_inputs_t20/local_inputs.json')
    layouts=directory/'layouts';layouts.mkdir()
    for stain in ('cc10','cd31','ki67','prospc'):
        shutil.copyfile(DATA/'lung_lesion3_eval/canvas'/(stain+'_layout.json'),layouts/(stain+'_layout.json'))
    for name in ('histo','rat_kidney'):
        shutil.copyfile(DATA/'birl_anhir_dev/canvas'/(name+'_layout.json'),layouts/(name+'_layout.json'))
    checked=[];chosen=None
    for case in cases:
        cfg=case['original_configuration']
        for role in ('fixed','moving'):
            filename=Path(cfg[role]).name
            cfg[role]=str((BASE/'miit_three_rotations_t153' if case['cohort']=='miit' else DATA/('birl_anhir_dev/canvas' if case['name'] in ('histo','rat_kidney') else 'lung_lesion3_eval/canvas'))/filename)
        source,roles=input_pair(case,layouts,native,'D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')
        name=case['name']
        if case['cohort']=='miit':
            moving,fixed=name.removeprefix('miit_').split('_to_')
            assert source['moving'].parts[-3]==moving and source['fixed'].parts[-3]==fixed
            expected=read(BASE/'miit_three_rotations_t153'/(name+'_layout.json'))
            assert all(roles[role]==expected[role] for role in ('fixed','moving'))
        elif name in ('histo','rat_kidney'):
            expected=read(DATA/'birl_anhir_dev/canvas'/(name+'_layout.json'))
            assert all(roles[role]==expected[role] for role in ('fixed','moving'))
        else:
            for role,stain in zip(('fixed','moving'),name.split('_to_')):
                expected=read(DATA/'lung_lesion3_eval/canvas'/(('cc10' if stain=='he' else stain)+'_layout.json'))['fixed' if stain=='he' else 'moving']
                assert roles[role]==expected
        for role in ('fixed','moving'):
            assert Path(str(roles[role]['source']).replace('\\','/')).name==source[role].name
        if name=='he_to_cc10':chosen=(case,source,roles)
        checked.append(name)
    assert len(checked)==25 and chosen is not None
    cache=decoder_cache(DATA/'historical_rgb_decode_cache/manifest.json')
    case,sources,roles=chosen;record=prepare_pair(case,directory/'prepared',sources,roles,decoded_sources=cache)
    old={role:1-np.asarray(Image.open(case['original_configuration'][role]).convert('L'),dtype=np.float32)/255 for role in ('fixed','moving')}
    oldmask=old['fixed']>.04;newmask=np.repeat(np.repeat(oldmask,2,0),2,1).astype(np.float32)
    lift_error=0.;coordinate_error=0.
    for arm in ('direct_original','lift512'):
        with np.load(record['bundles'][arm],allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            assert metadata['arm']==arm and metadata['source_image_side']==512 and metadata['terminal_image_side']==1024
            assert metadata['point_prediction_side']==metadata['point_pixel_scale']==512 and metadata['point_robust_scale']==8
            assert metadata['annotations_read'] is False and metadata['uniform_native1024_resolution'] is False
            assert np.array_equal(archive['fixed_mask'][0,0],newmask) and newmask.mean()==oldmask.mean()
            for role in ('fixed','moving'):
                layout=roles[role];wh=np.array(layout['original_wh']);n=np.array(layout['resized_wh']);p=np.array(layout['padding_xy'])
                assert wh[0]!=wh[1] and (p>0).any()
                q=np.array([[0,0],wh-1,[.317*wh[0],.683*wh[1]]],dtype=float)
                u=((q+.5)*(n/wh)+p)/512;u2=((q+.5)*((2*n)/wh)+2*p)/1024
                coordinate_error=max(coordinate_error,float(np.abs(u-u2).max()));assert np.array_equal(u,u2)
                c=u*512-.5;c2=u2*1024-.5;assert np.array_equal(c2,2*c+.5)
                if arm=='direct_original':
                    source=Image.open(cache[sources[role].name]['decoded_rgb']).convert('RGB');canvas=Image.new('RGB',(1024,1024),'white')
                    canvas.paste(source.resize(tuple(2*n),Image.Resampling.BILINEAR),tuple(2*p))
                    expected=1-np.asarray(canvas.convert('L'),dtype=np.float32)/255
                    assert np.array_equal(archive[role][0,0],expected)
                else:
                    expected=lift_oracle(old[role]);error=float(np.abs(archive[role][0,0]-expected).max());lift_error=max(lift_error,error)
                    assert error<1.3e-7 and np.any(np.abs(archive[role]*255-np.round(archive[role]*255))>.01)
                role_meta=metadata['roles'][role]
                assert role_meta['upscaled_axes']==(2*n>wh).tolist() and np.array_equal(role_meta['terminal_to_original_resize_ratio_xy'],2*n/wh)
        cfg=terminal_configuration(case['original_configuration'],directory/(arm+'.npz'))
        bundle=load_terminal_bundle(cfg,record['bundles'][arm]);_,_,mask,meta=validate_terminal_tensors(bundle,cfg,torch.tensor(oldmask[None,None],dtype=torch.float32),device='cpu',dtype=torch.float32)
        assert torch.equal(mask,torch.tensor(newmask[None,None])) and meta['source_mask_normalized_mass']==meta['terminal_mask_normalized_mass']
    # One wrong accepted preparation must fail exact source reconstruction.
    bad=directory/'wrong_accepted.png';wrong=np.array(Image.open(case['original_configuration']['fixed']).convert('RGB'));wrong[0,0,0]^=1;Image.fromarray(wrong).save(bad)
    try:render_original(sources['fixed'],roles['fixed'],bad,decoded=cache[sources['fixed'].name])
    except ValueError as error:assert 'reproduce' in str(error)
    else:raise AssertionError('wrong accepted preparation silently rendered')
    return dict(actual_layout_roles_checked=25,actual_nonsquare_pair='he_to_cc10',original_wh=roles['fixed']['original_wh'],accepted_resize=roles['fixed']['resized_wh'],padding=roles['fixed']['padding_xy'],
        normalized_coordinate_error=coordinate_error,direct_original_PIL_order_exact=True,lift_independent_bilinear_maximum_error=lift_error,
        mask_repeat_and_normalized_mass_exact=True,metadata_records_original_upsampling_and512_point_units=True,wrong_accepted_RGB_rejected=True)


def point_check():
    item=fixture();vertices=item['vertices'];a=item['a'];b=item['b'];q=item['q'];p=item['p'];confidence=item['confidence']
    eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1)
    literal_value,literal_gradient,_=literal_value_gradient(vertices,q,p,confidence,a,b)
    values=[];gradients=[]
    for pixels,robust in ((512,8),(1024,16)):
        points=ImageCorrespondences(torch.tensor(q),torch.tensor(p),torch.tensor(confidence*eligible),pixel_scale=pixels,robust_scale=robust)
        y=torch.tensor(vertices[None],requires_grad=True);value=points(y,torch.tensor(a),'p1_ac');gradient=torch.autograd.grad(value,y)[0][0]
        values.append(value.detach());gradients.append(gradient)
        assert abs(float(value.detach())-literal_value)<1e-13 and np.abs(gradient.numpy()-literal_gradient).max()<1e-13
    assert torch.equal(values[0],values[1]) and torch.equal(gradients[0],gradients[1])
    return dict(point_value_and_vertex_VJP_512_over8_equals1024_over16_bitwise=True,independent_static_point_oracle_agrees=True)


def tiny_core_hook_checks(directory):
    from tests.test_coordinated_shared_affine_evidence import configuration
    cfg=configuration(directory,'analytic');cfg.mind_frame='shared_affine';cfg.p1_sampling='frozen'
    fixed,moving,mask,_=app.load_registration_evidence(cfg.fixed,cfg.moving,cfg.image_side)
    oldside=cfg.image_side;original=copy.deepcopy(cfg);frames={};accepted={};reports={}
    real_evidence=app.Evidence
    def run(label,args,terminal=None):
        frames[label]={};accepted[label]=[]
        def capture(*a,**kw):
            evidence=real_evidence(*a,**kw);side=evidence.fixed.shape[-1]
            frames[label][side]=(evidence.fixed.detach().clone(),evidence.moving.detach().clone(),evidence.mask.detach().clone(),evidence.fixed_feature.detach().clone(),evidence.moving_feature.detach().clone())
            return evidence
        app.Evidence=capture
        try:reports[label]=app.optimize(args,accepted_stage_callback=lambda y,s,t:accepted[label].append((y.detach().clone(),copy.deepcopy(s))),terminal_evidence=terminal)
        finally:app.Evidence=real_evidence
        return reports[label]
    old=run('old',cfg)
    same=dict(fixed=fixed.numpy(),moving=moving.numpy(),fixed_mask=mask.numpy(),source_image_side=oldside,metadata=dict(arm='same_raster_checker'))
    samecfg=copy.deepcopy(original);samecfg.output=directory/'same_override.npz';same_report=run('same',samecfg,same)
    for key in ('initial','final','stages','trace','gradient_steps','objective_evaluations'):assert old[key]==same_report[key]
    with np.load(cfg.output) as a,np.load(samecfg.output) as b:
        assert all(np.array_equal(a[key],b[key]) for key in a.files)
    changed=dict(fixed=(1-F.interpolate(fixed,scale_factor=2,mode='bilinear',align_corners=False)).numpy(),
        moving=F.interpolate(moving,scale_factor=2,mode='bilinear',align_corners=False).numpy(),
        fixed_mask=mask.repeat_interleave(2,-2).repeat_interleave(2,-1).numpy(),source_image_side=oldside,metadata=dict(arm='changed_terminal_checker'))
    highcfg=copy.deepcopy(original);highcfg.output=directory/'changed_terminal.npz';highcfg.image_side=2*oldside;highcfg.image_levels=[original.image_levels[0],2*oldside]
    high=run('changed',highcfg,changed)
    low=original.image_levels[0]
    assert all(torch.equal(x,y) for x,y in zip(frames['old'][low],frames['changed'][low]))
    assert all(torch.equal(x[0],y[0]) for x,y in zip(accepted['old'][:2],accepted['changed'][:2]))
    assert high['pyramid_source_image_side']==high['point_pixel_scale']==oldside and high['query_count']==(2*oldside)**2
    assert high['control_vertices']==old['control_vertices'] and high['gradient_steps']==old['gradient_steps']
    assert high['final']['total']==min([high['initial']['total']]+[s['accepted_full_total'] for s in high['stages']])
    # Independently evaluate every accepted map under the CHANGED terminal
    # evidence to distinguish terminal selection from the source-raster selector.
    with np.load(original.affine) as saved:a=saved['post_affine_matrix'].astype(float);b=saved['post_affine_offset'].astype(float)
    errors=[]
    for y,stage in accepted['changed']:
        vertices=y.numpy()[0];image,oob,_=numpy_image(vertices,np.zeros(6),a,b,changed['fixed'][0,0],changed['moving'][0,0],changed['fixed_mask'][0,0])
        strain,shape=numpy_priors(vertices);value=image+3*strain+1e-4*shape+oob
        errors.append(abs(value-stage['accepted_full_total']))
    assert max(errors)<3e-6
    return dict(same_raster_override_objectives_updates_and_saved_arrays_bitwise=True,lower_stage_intensities_masks_descriptors_and_maps_exact=True,
        terminal_queries_not_new_controls=True,full_terminal_selector_maximum_oracle_error=max(errors),tiny_saved_certificate_valid=high['saved_binary_certificate']['valid'])


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='independent_terminal_detail_',dir=BASE) as folder:
        path=Path(folder);(path/'tiny').mkdir()
        result=dict(render=actual_layout_and_render_checks(path),points=point_check(),core_hook=tiny_core_hook_checks(path/'tiny'))
    print(json.dumps(result,indent=2))
