import copy
import numpy as np
import pytest
import torch
from tools import coordinated_refreshed_matches as refresh
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_q1_dhr_distill import identity_vertices
from tools.coordinated_registration_start import load_incumbent
from tools import coordinated_real_case as original
from tests.test_coordinated_shared_affine_evidence import configuration


def literal_p1(vertices,queries):
    table=vertices[0].cpu().numpy();n=len(table);out=[]
    for x,y in queries:
        col,row=min(int(np.floor(x*(n-1))),n-2),min(int(np.floor(y*(n-1))),n-2)
        u,v=x*(n-1)-col,y*(n-1)-row
        a,b,c,d=table[row,col],table[row,col+1],table[row+1,col+1],table[row+1,col]
        out.append((1-u)*a+(u-v)*b+v*c if v<=u else (1-v)*a+u*c+(v-u)*d)
    return np.asarray(out).reshape(-1,2)


def deformed():
    value=identity_vertices(3,device='cpu').double();value[0,1,1,0]+=.12
    return value


def test_identity_renderer_exact_old_float32_affine_and_single_sample():
    moving=torch.rand(1,1,512,512,generator=torch.Generator().manual_seed(16))
    a=torch.tensor([[1.07,.13],[-.05,.94]]);b=torch.tensor([-.08,.04])
    actual=refresh.render_incumbent(moving,identity_vertices(257,device='cpu').double(),a,b)
    old=warp_moving_to_fixed(moving,a,b,height=512,width=512)
    assert torch.equal(actual,old)


def test_nonlinear_p1_targets_native_residual_and_eligibility_after_transform():
    v=deformed();s=np.array([[.35,.15]]*7+[[.9,.5]])
    q=np.array([[.2,.2]]*8);a=np.array([[1.2,.0],[.07,.9]],dtype=np.float32);b=np.array([-.1,.02],dtype=np.float32)
    c=np.linspace(.2,.9,8)
    result=refresh.refreshed_point_record(q*512-.5,s*512-.5,c,a,b,v)
    p=literal_p1(v,s)
    np.testing.assert_array_equal(result['warped_target_points_unit'],s)
    np.testing.assert_allclose(result['target_points_unit'],p,atol=1e-15,rtol=0)
    assert abs(p[0,0]-(.35+.7*.3*.12))>.01  # Q1 would use the product .7*.3.
    assert result['eligible_matches']==7 and result['status']=='insufficient_matches'
    assert result['static_outside_original_moving_matches']==1
    assert ((s@a.astype(float).T+b>=0)&(s@a.astype(float).T+b<=1)).all()  # s-based eligibility is wrong.
    candidate=literal_p1(v,q)+.002
    left=(candidate@a.astype(float).T+b)-(p@a.astype(float).T+b)
    np.testing.assert_allclose(left,(candidate-p)@a.astype(float).T,atol=3e-16,rtol=0)
    np.testing.assert_array_equal(result['confidence'],c)


def test_raw_validation_precedes_filter_and_invalid_transformed_target_fails():
    points=np.array([[20.,20.]]*8)
    outside=np.concatenate((points,[[-1.,20.]]));weights=np.ones(9)
    result=refresh.refreshed_point_record(outside,outside,weights,np.eye(2),np.zeros(2),deformed())
    assert result['discarded_out_of_unit_domain']==1 and result['raw_matches']==8
    bad=outside.copy();bad[-1,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        refresh.refreshed_point_record(bad,outside,weights,np.eye(2),np.zeros(2),deformed())
    invalid=deformed()+2
    with pytest.raises(ValueError,match='transformed'):
        refresh.refreshed_point_record(points,points,np.ones(8),np.eye(2),np.zeros(2),invalid)
    empty=refresh.refreshed_point_record(np.empty((0,2)),np.empty((0,2)),np.empty(0),np.eye(2),np.zeros(2),deformed())
    assert empty['status']=='insufficient_matches'
    endpoints=np.array([[-.5,-.5],[511.5,511.5]])
    result=refresh.refreshed_point_record(endpoints,endpoints,np.ones(2),np.eye(2),np.zeros(2),deformed())
    np.testing.assert_array_equal(result['target_points_unit'],[[0,0],[1,1]])


def save_map(path,vertices,a,b,reference=None):
    np.savez(path,vertices=vertices.numpy(),boundary_reference=(identity_vertices(vertices.shape[1],device='cpu').double()
        if reference is None else reference).numpy(),post_affine_matrix=a,post_affine_offset=b,interpolation=np.asarray('p1_ac'))


def test_incoming_identity_default_bitwise_and_actual_nonidentity_start(tmp_path):
    cfg=configuration(tmp_path,'analytic');cfg.mind_frame='shared_affine'
    with np.load(cfg.affine) as data:a,b=data['post_affine_matrix'],data['post_affine_offset']
    identity=identity_vertices(cfg.grid_side,device='cpu').double()
    path=tmp_path/'identity.npz';save_map(path,identity,a,b)
    baseline=original.optimize(cfg);cfg.output=tmp_path/'from_identity.npz'
    explicit=original.optimize(cfg,initial_map=path)
    for key in ('initial','final','stages','trace','gradient_steps','objective_evaluations'):
        assert baseline[key]==explicit[key]
    displaced=identity.clone();displaced[:,1:-1,1:-1,0]+=.008
    start=tmp_path/'nonidentity.npz';save_map(start,displaced,a,b)
    loaded,metadata=load_incumbent(start,a,b,cfg.grid_side)
    assert torch.equal(loaded,displaced) and metadata['minimum_corner_ratio']>.001
    from tools.coordinated_data_metric_application import _build_evidence
    evidence,_,_=_build_evidence(cfg,a,b,torch.device('cpu'))
    expected,parts=evidence(displaced)
    cfg.output=tmp_path/'nonidentity_suffix.npz';report=original.optimize(cfg,initial_map=start)
    assert report['initial']==dict(total=float(expected),**{k:float(v) for k,v in parts.items()})
    assert report['final']['total']<=report['initial']['total']
    assert report['final']['total']==min([report['initial']['total']]+[s['accepted_full_total'] for s in report['stages']])
    assert report['saved_binary_certificate']['valid'] and report['initial_map']['path']==str(start.resolve())
    assert 'incoming' in report['selected_stage_scope']


def test_incumbent_rejects_changed_affine_reference_boundary_and_floor(tmp_path):
    v=deformed();a=np.eye(2,dtype=np.float32);b=np.zeros(2,dtype=np.float32)
    path=tmp_path/'valid.npz';save_map(path,v,a,b)
    with pytest.raises(ValueError,match='affine'):load_incumbent(path,a,b+.01,3)
    with pytest.raises(ValueError,match='floor'):load_incumbent(path,a,b,3,minimum_jacobian=.99)
    changed=v.clone();changed[:,0,:,0]+=.01;bad=tmp_path/'boundary.npz';save_map(bad,changed,a,b)
    with pytest.raises(ValueError):load_incumbent(bad,a,b,3)
    translated=tmp_path/'reference.npz';save_map(translated,v+2,a,b,reference=identity_vertices(3,device='cpu').double()+2)
    with pytest.raises(ValueError,match='unit-square'):load_incumbent(translated,a,b,3)


def test_incoming_map_is_retained_when_no_trial_improves_own_objective(tmp_path,monkeypatch):
    cfg=configuration(tmp_path,'analytic');cfg.mind_frame='shared_affine'
    with np.load(cfg.affine) as data:a,b=data['post_affine_matrix'],data['post_affine_offset']
    incoming=identity_vertices(cfg.grid_side,device='cpu').double();incoming[:,1:-1,1:-1,1]+=.009
    path=tmp_path/'incoming.npz';save_map(path,incoming,a,b)
    def objective(self,vertices):
        value=(vertices-incoming).square().sum()
        return value,dict(image=value,strain=value*0,oob=value*0,shape=value*0,match=value*0,outside_fraction=value*0)
    monkeypatch.setattr(original.Evidence,'__call__',objective)
    report=original.optimize(cfg,initial_map=path)
    assert report['selected_stage'] is None and report['initial']['total']==report['final']['total']==0
    with np.load(cfg.output) as saved:np.testing.assert_array_equal(saved['vertices'],incoming.numpy())
