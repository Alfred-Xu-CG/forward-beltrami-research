"""Strict-floor reserve: historical behavior, nonlinear path and local VJP."""
import argparse
import math

import torch
import numpy as np
from PIL import Image
import pytest

from qcopt.neural_bijection.dense.digital_q1 import SafePatchQ1Pass, StaggeredPatchQ1Layer


def grid(side, dtype=torch.float64):
    y, x = torch.meshgrid(torch.arange(side, dtype=dtype)/(side-1),
                          torch.arange(side, dtype=dtype)/(side-1), indexing="ij")
    return torch.stack((x, y), -1)[None]


def cross(e, f):
    return e[..., 0]*f[..., 1]-e[..., 1]*f[..., 0]


def corners(v):
    a, b, c, d = v[:, :-1, :-1], v[:, :-1, 1:], v[:, 1:, 1:], v[:, 1:, :-1]
    return torch.stack([cross(b-a, d-a), cross(b-a, c-b),
                        cross(c-d, d-a), cross(c-d, c-b)], -1)


def historical_candidate(layer, base, logits):
    """Literal pre-change forward arithmetic: remaining floor budget has no reserve."""
    side, cells = layer.side, layer.patch_cells
    batch = len(base)
    current = base.reshape(batch, side*side, 2)
    patch = current[:, layer.patch_ids]
    selected = logits.reshape(batch, -1, 2)[:, layer.latent_ids]
    raw = layer.raw_span*cells/(side-1)*torch.tanh(selected).reshape(batch, patch.shape[1], cells-1, cells-1, 2)
    delta = torch.zeros_like(patch); delta[:, :, 1:-1, 1:-1] = raw
    a,b,c,d = patch[:, :, :-1, :-1],patch[:, :, :-1, 1:],patch[:, :, 1:, 1:],patch[:, :, 1:, :-1]
    da,db,dc,dd = delta[:, :, :-1, :-1],delta[:, :, :-1, 1:],delta[:, :, 1:, 1:],delta[:, :, 1:, :-1]
    terms=[]
    for p,q,r,dp,dq,dr in ((a,b,d,da,db,dd),(a,b,c,da,db,dc),(d,b,c,dd,db,dc),(a,c,d,da,dc,dd)):
        e,f=q-p,r-p; de,df=dq-dp,dr-dp
        area=cross(e,f); linear=cross(de,f)+cross(e,df); quadratic=cross(de,df)
        terms.append((area,(-linear).clamp_min(0)+(-quadratic).clamp_min(0)))
    area=torch.stack([t[0] for t in terms],-1).reshape(batch,patch.shape[1],-1)
    bounds=torch.stack([t[1] for t in terms],-1).reshape_as(area)
    allowance=layer.safety_fraction*area
    if layer.minimum_jacobian is not None:
        allowance=torch.minimum(allowance,(area-layer.minimum_jacobian/(side-1)**2).clamp_min(0))
    guard=math.sqrt(torch.finfo(base.dtype).eps)/(side-1)**2
    quotient=allowance/torch.maximum(torch.maximum(bounds,allowance),bounds.new_tensor(guard))
    scales=torch.where(bounds>0,quotient,torch.ones_like(bounds)).amin(-1)
    updated=patch[:, :, 1:-1, 1:-1]+layer.accepted_gain*scales[:, :, None, None, None]*raw
    return current.index_copy(1,layer.interior_ids,updated.reshape(batch,-1,2)).reshape_as(base)


def test_linear_exact_floor_contact_and_strict_reserve():
    base=grid(3); base[0,1,1]=base.new_tensor([.001,.5])
    logits=base.new_tensor([[[[-1.,0.]]]])
    old=SafePatchQ1Pass(3,2,minimum_jacobian=.001)(base,logits)
    new=SafePatchQ1Pass(3,2,minimum_jacobian=.001,floor_safety_fraction=.95)(base,logits)
    assert float((corners(old)*4).min()) == pytest.approx(.001,abs=1e-12)
    assert float((corners(new)*4).min()) > .001
    assert float(new[0,1,1,0]) == pytest.approx(.000525,abs=1e-12)
    assert float((corners(new)*4).min()) == pytest.approx(.00105,abs=1e-12)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("floor",[None,.001])
def test_default_bit_identical_to_historical_arithmetic(dtype,floor):
    generator=torch.Generator().manual_seed(712)
    base=grid(9,dtype).repeat(2,1,1,1)
    base[:,1:-1,1:-1]+=torch.randn(2,7,7,2,generator=generator,dtype=dtype)*.004
    logits=torch.randn(2,7,7,2,generator=generator,dtype=dtype)
    layer=SafePatchQ1Pass(9,4,minimum_jacobian=floor)
    explicit=SafePatchQ1Pass(9,4,minimum_jacobian=floor,floor_safety_fraction=1.)
    assert torch.equal(layer(base,logits),historical_candidate(layer,base,logits))
    assert torch.equal(layer(base,logits),explicit(base,logits))


@pytest.mark.parametrize("fraction",[.95,.5])
def test_nonlinear_path_bound_and_exact_boundary(fraction):
    base=grid(5);base[:,1:-1,1,0]=.001
    generator=torch.Generator().manual_seed(47)
    logits=3*torch.randn(1,3,3,2,generator=generator,dtype=base.dtype)
    layer=SafePatchQ1Pass(5,4,minimum_jacobian=.001,floor_safety_fraction=fraction)
    out=layer(base,logits); delta=out-base
    q0=corners(base); floor=.001/16
    assert bool((q0>floor).all())
    # At least one genuine quadratic path coefficient, not only a linear test.
    polynomial_quadratic=corners(base+delta)-2*corners(base+.5*delta)+q0
    # Thin-cell scaling makes this ~1e-7, still >1e9 times arithmetic noise.
    assert float(polynomial_quadratic.abs().max())>1e6*torch.finfo(base.dtype).eps*float(q0.abs().max())
    reserve=floor+(1-fraction)*(q0-floor)
    for t in torch.linspace(0,1,41,dtype=base.dtype):
        assert bool((corners(base+t*delta)>=reserve-5e-16).all())
    assert torch.equal(out[:,0],base[:,0]) and torch.equal(out[:,-1],base[:,-1])
    assert torch.equal(out[:,:,0],base[:,:,0]) and torch.equal(out[:,:,-1],base[:,:,-1])


@pytest.mark.parametrize("bad",[0.,-1.,1.001,float('nan'),float('inf'),-float('inf'),True,'0.95',torch.tensor(.95)])
def test_bad_fraction_rejected(bad):
    with pytest.raises((ValueError,TypeError)):
        SafePatchQ1Pass(5,4,floor_safety_fraction=bad)


def test_staggered_passes_receive_reserve_and_have_strict_actual_slack():
    layer=StaggeredPatchQ1Layer(9,4,minimum_jacobian=.001,floor_safety_fraction=.95)
    assert all(p.floor_safety_fraction==.95 for p in layer.passes)
    base=grid(9); generator=torch.Generator().manual_seed(8)
    logits=torch.randn(1,7,7,2,dtype=base.dtype,generator=generator)*2
    out=layer(base,tuple(logits for _ in layer.passes))
    assert float((corners(out)*64).min())>.001


def test_directional_gradient_away_branch_switch():
    base=grid(5);base[:,1:-1,1,0]=.001
    generator=torch.Generator().manual_seed(199)
    logits=(torch.randn(1,3,3,2,generator=generator,dtype=base.dtype)*.9).requires_grad_()
    tangent=torch.randn(logits.shape,generator=generator,dtype=base.dtype)
    upstream=torch.randn(base.shape,generator=generator,dtype=base.dtype)
    layer=SafePatchQ1Pass(5,4,minimum_jacobian=.001,floor_safety_fraction=.95)
    loss=(layer(base,logits)*upstream).sum()
    gradient,=torch.autograd.grad(loss,logits)
    derivative=(gradient*tangent).sum()
    finite=( (layer(base,logits+1e-6*tangent)*upstream).sum()
            -(layer(base,logits-1e-6*tangent)*upstream).sum())/(2e-6)
    torch.testing.assert_close(derivative,finite,atol=5e-9,rtol=2e-6)


def test_current_vertices_and_logits_joint_directional_gradient():
    base=grid(5);base[:,1:-1,1,0]=.001;base.requires_grad_()
    generator=torch.Generator().manual_seed(199)
    logits=(torch.randn(1,3,3,2,generator=generator,dtype=base.dtype)*.9).requires_grad_()
    dz=torch.randn(logits.shape,generator=generator,dtype=base.dtype)
    dy=torch.randn(base.shape,generator=generator,dtype=base.dtype)*.02
    upstream=torch.randn(base.shape,generator=generator,dtype=base.dtype)
    layer=SafePatchQ1Pass(5,4,minimum_jacobian=.001,floor_safety_fraction=.95)
    gy,gz=torch.autograd.grad((layer(base,logits)*upstream).sum(),(base,logits))
    derivative=(gy*dy).sum()+(gz*dz).sum()
    finite=((layer(base+1e-6*dy,logits+1e-6*dz)*upstream).sum()
            -(layer(base-1e-6*dy,logits-1e-6*dz)*upstream).sum())/(2e-6)
    torch.testing.assert_close(derivative,finite,atol=5e-9,rtol=2e-6)


def test_application_invalid_fraction_rejected_before_input_loading(tmp_path):
    from tools.coordinated_lung_all20 import make_configuration
    from tools.coordinated_real_case import optimize
    args=argparse.Namespace(output=tmp_path,device='cpu',threads=1)
    pair=dict(name='tiny',fixed=tmp_path/'missing.png',moving=tmp_path/'missing.png',affine=tmp_path/'missing.npz')
    config=make_configuration(pair,'f2',args,production=False)
    config.f2_floor_safety_fraction=float('nan')
    with pytest.raises(ValueError,match='floor_safety_fraction'):
        optimize(config)


def test_application_old_namespace_records_default_and_explicit_reserve(tmp_path):
    from tools.coordinated_lung_all20 import make_configuration
    from tools.coordinated_real_case import optimize
    image=tmp_path/'image.png'
    Image.fromarray(np.random.default_rng(23).integers(20,230,(16,16),dtype=np.uint8)).save(image)
    affine=tmp_path/'affine.npz'
    np.savez(affine,post_affine_matrix=np.eye(2,dtype=np.float32),post_affine_offset=np.zeros(2,dtype=np.float32))
    args=argparse.Namespace(output=tmp_path,device='cpu',threads=1)
    pair=dict(name='tiny',fixed=image,moving=image,affine=affine)
    for fraction in (None,.95):
        config=make_configuration(pair,'f2',args,production=False)
        config.matches=None;config.match_weight=0
        config.output=tmp_path/('default.npz' if fraction is None else 'reserve.npz')
        if fraction is not None:config.f2_floor_safety_fraction=fraction
        report=optimize(config)
        expected=1. if fraction is None else fraction
        assert report['configuration']['f2_floor_safety_fraction']==expected
        assert report['f2_floor_safety_fraction']==expected
        assert report['saved_binary_certificate']['valid']


def test_compare_tool_only_floor_fraction_changes_recipe(tmp_path):
    from tools.coordinated_f2_floor_compare import make_configuration
    args=argparse.Namespace(canvas=tmp_path/'canvas',affines_from=tmp_path/'affines',
        predictions=tmp_path/'frozen',output=tmp_path/'result',device='cpu',threads=1)
    low=make_configuration('he_to_ki67',args,1.)
    high=make_configuration('he_to_ki67',args,.95)
    left=dict(vars(low));right=dict(vars(high))
    for key in ('output','f2_floor_safety_fraction'):
        left.pop(key);right.pop(key)
    assert left==right
    assert low.inner_steps*low.cycles*len(low.levels)==300
    assert low.cycles==2 and low.f2_accepted_gain==1.
    assert low.interpolation=='p1_ac' and low.strain_model=='p1_arap' and low.strain_weight==3.
    assert low.matches==args.predictions/'he_to_ki67_raw_matches.json'
    assert low.f2_floor_safety_fraction==1. and high.f2_floor_safety_fraction==.95


def comparison_assets(tmp_path):
    import json
    from tools.coordinated_f2_floor_compare import CASES
    canvas=tmp_path/'canvas';canvas.mkdir()
    affines=tmp_path/'affines';affines.mkdir()
    frozen=tmp_path/'frozen';frozen.mkdir()
    raster=np.random.default_rng(88).integers(30,220,(16,16),dtype=np.uint8)
    for filename in ('cc10_fixed512.png','cc10_moving512.png','ki67_moving512.png'):
        Image.fromarray(raster).save(canvas/filename)
    rows=[];sources={}
    points=[[x,y] for x in (.2,.5,.8) for y in (.2,.5,.8)]
    for name in CASES:
        fixed,moving=name.split('_to_')
        first=canvas/('cc10_fixed512.png' if fixed=='he' else fixed+'_moving512.png')
        second=canvas/('cc10_fixed512.png' if moving=='he' else moving+'_moving512.png')
        np.savez(affines/(name+'_affine.npz'),post_affine_matrix=np.eye(2,dtype=np.float32),
                 post_affine_offset=np.zeros(2,dtype=np.float32))
        record=dict(status='ok',fixed=str(first),moving=str(second),image_side=16,
            source_points_unit=points,target_points_unit=points,confidence=[.8]*len(points),
            post_affine_matrix=np.eye(2).tolist(),post_affine_offset=[0,0],
            targets_manual_landmarks_or_dense_teacher_loaded=False,global_geometric_ransac_used=False)
        path=frozen/(name+'_raw_matches.json')
        path.write_text(json.dumps(record));sources[path]=path.read_bytes()
        rows.append(dict(name=name,raw_matches=dict(status='ok',path=path.name)))
    (frozen/'predictions.json').write_text(json.dumps(dict(prediction_complete=True,annotations_read=False,rows=rows)))
    args=argparse.Namespace(canvas=canvas,affines_from=affines,predictions=frozen,
                            output=tmp_path/'comparison',device='cpu',threads=1)
    return args,sources


def test_compare_tool_actual_tiny_paired_runs_and_frozen_inputs(tmp_path):
    from tools.coordinated_f2_floor_compare import run
    args,sources=comparison_assets(tmp_path)
    result=run(args,production=False)
    assert result['attempts']==6 and len(result['rows'])==6
    assert all(row['status']=='ok' and row['expected_gradient_steps']==4 for row in result['rows'])
    assert all(row['actual_minimum_corner_ratio']>.001 for row in result['rows'])
    assert all(row['gradient_steps']==4 and row['budget_complete'] for row in result['rows'])
    assert result['annotations_read'] is False
    for path,content in sources.items():assert path.read_bytes()==content
    with pytest.raises(FileExistsError):run(args,production=False)


@pytest.mark.parametrize('invalid',['certificate','strict_floor'])
def test_compare_tool_invalid_export_is_failed_attempt(tmp_path,monkeypatch,invalid):
    from tools import coordinated_f2_floor_compare as tool
    args,_=comparison_assets(tmp_path)
    if invalid=='strict_floor':
        # Simulate an unexpected saved floor-contact result, not a relaxed guard.
        monkeypatch.setattr(tool,'_actual_ratio',lambda path:.001)
    else:
        original=tool.optimize
        def invalid_certificate(config):
            report=original(config)
            report['saved_binary_certificate']={**report['saved_binary_certificate'],'valid':False}
            return report
        monkeypatch.setattr(tool,'optimize',invalid_certificate)
    result=tool.run(args,production=False)
    assert result['attempts']==6 and len(result['rows'])==6
    assert all(row['status']=='failed' for row in result['rows'])
    assert all('actual_minimum_corner_ratio' in row and 'saved_binary_certificate' in row
               for row in result['rows'])
    assert sum(row['status']=='ok' for row in result['rows'])==0
