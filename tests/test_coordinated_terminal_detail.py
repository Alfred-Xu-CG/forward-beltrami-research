"""Narrow terminal-only evidence override; no extra matching or control grid."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools import coordinated_real_case as app
from tools import coordinated_terminal_detail as detail
from tests.test_coordinated_shared_affine_evidence import configuration,identity
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences


def override(args,side):
    fixed,moving,mask,_=app.load_registration_evidence(args.fixed,args.moving,args.image_side)
    factor=side//args.image_side
    return dict(fixed=F.interpolate(fixed,size=(side,side),mode='bilinear',align_corners=False).numpy(),
        moving=F.interpolate(moving,size=(side,side),mode='bilinear',align_corners=False).numpy(),
        fixed_mask=mask.repeat_interleave(factor,-2).repeat_interleave(factor,-1).numpy(),
        source_image_side=args.image_side,metadata=dict(arm='same_raster_fixture'))


def test_same_raster_override_is_bitwise_old_optimizer(tmp_path):
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    base=app.optimize(args)
    with np.load(args.output) as f:vertices=f['vertices'].copy()
    args.output=tmp_path/'override.npz'
    result=app.optimize(args,terminal_evidence=override(args,args.image_side))
    for key in ('initial','final','stages','trace','gradient_steps','failures','objective_evaluations'):
        assert result[key]==base[key],key
    with np.load(args.output) as f:np.testing.assert_array_equal(vertices,f['vertices'])
    assert result['point_pixel_scale']==16 and result['query_count']==16**2
    assert result['pyramid_source_image_side']==16


def test_terminal_changed_but_lower_stage_accepted_maps_exact(tmp_path):
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    original=copy.deepcopy(args)
    old=[];app.optimize(args,accepted_stage_callback=lambda y,s,t:old.append((y.clone(),s)))
    data=override(original,32)
    # Deliberately different terminal evidence cannot alter first8pixel stage.
    data['fixed']=1-data['fixed']
    args.output=tmp_path/'terminal32.npz';args.image_side=32;args.image_levels=[8,32]
    new=[];result=app.optimize(args,accepted_stage_callback=lambda y,s,t:new.append((y.clone(),s)),terminal_evidence=data)
    for (a,sa),(b,sb) in zip(old[:2],new[:2]):
        assert torch.equal(a,b)
        for key in ('accepted_total','anchor_total','physical_lr'):assert sa[key]==sb[key]
    assert result['image_levels']==[8,32] and result['query_count']==1024
    assert result['configuration']['image_side']==32 and result['point_pixel_scale']==16
    assert set(result['mind_frame_by_resolution'])=={'8','32'}
    assert result['gradient_steps']==4 and result['saved_binary_certificate']['valid']


def test_point_pixel_and_robust_scale_doubling_exact_value_vjp():
    q=torch.tensor([[.17,.23],[.61,.72]],dtype=torch.float64);p=q+.03
    a=ImageCorrespondences(q,p,torch.ones(2,dtype=torch.float64),pixel_scale=512,robust_scale=8)
    b=ImageCorrespondences(q,p,torch.ones(2,dtype=torch.float64),pixel_scale=1024,robust_scale=16)
    Y=identity().requires_grad_();A=torch.tensor([[1.1,.04],[-.03,.9]],dtype=torch.float64)
    va,vb=a(Y,A,'p1_ac'),b(Y,A,'p1_ac')
    assert torch.equal(va,vb)
    ga=torch.autograd.grad(va,Y,retain_graph=True)[0];gb=torch.autograd.grad(vb,Y)[0]
    assert torch.equal(ga,gb)


def test_preserve_masks_and_validate_float_arrays(tmp_path):
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    _,_,mask,_=app.load_registration_evidence(args.fixed,args.moving,args.image_side)
    data=override(args,32);args.image_side=32
    f,m,w,metadata=detail.validate_terminal_tensors(data,args,mask,device='cpu',dtype=torch.float32)
    assert float(w.mean())==float(mask.mean())
    assert torch.equal(w,mask.repeat_interleave(2,-2).repeat_interleave(2,-1))
    for key in ('fixed','moving','fixed_mask'):
        broken=copy.deepcopy(data);broken[key]=broken[key].astype(np.float64)
        with pytest.raises(ValueError,match='float32'):detail.validate_terminal_tensors(broken,args,mask,device='cpu',dtype=torch.float32)
    broken=copy.deepcopy(data);broken['fixed_mask'][...,0,0]=1-broken['fixed_mask'][...,0,0]
    with pytest.raises(ValueError,match='repeat'):detail.validate_terminal_tensors(broken,args,mask,device='cpu',dtype=torch.float32)
    broken=copy.deepcopy(data);broken['moving'][...,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):detail.validate_terminal_tensors(broken,args,mask,device='cpu',dtype=torch.float32)


def test_bundle_loader_preserves_float_information_and_pair_metadata(tmp_path):
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    args.image_side=1024;args.image_levels=[32,64,128,256,1024]
    array=np.full((1,1,1024,1024),.1234567,dtype=np.float32)
    _,_,base_mask,_=app.load_registration_evidence(args.fixed,args.moving,512)
    mask=base_mask.repeat_interleave(2,-2).repeat_interleave(2,-1).numpy()
    metadata=dict(arm='direct_original',source_image_side=512,terminal_image_side=1024,fixed=str(args.fixed),moving=str(args.moving))
    path=tmp_path/'bundle.npz';np.savez(path,fixed=array,moving=array,fixed_mask=mask,metadata=np.asarray(json.dumps(metadata)))
    data=detail.load_terminal_bundle(args,path)
    np.testing.assert_array_equal(data['fixed'],array)
    assert data['metadata']['bundle_path']==str(path)
    metadata['fixed']='wrong.png'
    np.savez(path,fixed=array,moving=array,fixed_mask=mask,metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError,match='pair'):detail.load_terminal_bundle(args,path)


def test_unsupported_override_modes_rejected_before_loading(tmp_path,monkeypatch):
    args=configuration(tmp_path,'analytic');args.mind_frame='original'
    data=override(args,16)
    monkeypatch.setattr(app,'load_registration_evidence',lambda *a,**kw:pytest.fail('unsupported images loaded'))
    with pytest.raises(ValueError,match='terminal'):app.optimize(args,terminal_evidence=data)
