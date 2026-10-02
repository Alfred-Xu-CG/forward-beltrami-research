"""Tiny literal pose algebra, derivatives, objective and serialization tests."""
import argparse
import json
import numpy as np
import pytest
import torch
from PIL import Image

from tools import coordinated_joint_pose as jp
from tools.coordinated_real_case import Evidence
from tools.digital_q1_dhr_distill import identity_vertices
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries


def fixture(dtype=torch.float64):
    torch.manual_seed(42)
    fixed=torch.rand(1,1,16,16,dtype=dtype)
    moving=torch.rand_like(fixed)
    A=torch.tensor([[.89,.04],[-.03,.92]],dtype=torch.float64)
    b=torch.tensor([.039,.047],dtype=torch.float64)
    Y=identity_vertices(9,device='cpu').double()
    q=torch.rand(12,2,dtype=torch.float64)*.6+.2
    points=ImageCorrespondences(q,q+.012,torch.ones(12,dtype=torch.float64),pixel_scale=512,robust_scale=8)
    mask=torch.ones_like(fixed)
    evidence=jp.JointPoseEvidence(fixed,moving,A,b,mask,points,grid_side=9)
    return Y,A,b,evidence,points


def test_identity_pose_matches_existing_complete_value_and_vjp():
    Y,A,b,e,points=fixture(torch.float32)
    old=Evidence(e.fixed,e.moving,A,b,'mind',3.,1.,.0001,fixed_mask=e.mask,
        interpolation='p1_ac',matches=points,match_weight=.2,strain_model='p1_arap',mind_frame='shared_affine')
    old.prepare_fixed_p1_sampling(9,9,dtype=torch.float64,device='cpu')
    Y=Y.requires_grad_();physical=torch.zeros(6,dtype=torch.float64)
    a,pa=old(Y);z,pz=e(Y,physical)
    torch.testing.assert_close(a,z,rtol=0,atol=1e-12)
    for k in pa:torch.testing.assert_close(pa[k],pz[k],rtol=0,atol=1e-12)
    ga=torch.autograd.grad(a,Y,retain_graph=True)[0];gz=torch.autograd.grad(z,Y)[0]
    torch.testing.assert_close(ga,gz,rtol=1e-11,atol=1e-11)


def test_literal_composition_point_frame_and_corner_scaling():
    Y,A,b,e,points=fixture()
    physical=torch.tensor([.02,-.03,.12,.07,.03,-.02],dtype=torch.float64)
    B,c=jp.pose_affine(physical);Aout,bout=jp.combined_affine(A,b,physical)
    world=(Y@B.T+c)@A.T+b
    torch.testing.assert_close(world,Y@Aout.T+bout,rtol=1e-14,atol=1e-14)
    mapped=p1_map_at_queries(Y,points.source,'ac',validate_queries=False)
    error=((mapped@B.T+c-points.target)@A.T)*(512/8)
    squared=error.square().sum(-1)[0]
    literal=(squared/(torch.sqrt(1+squared)+1)*points.weights).sum()
    torch.testing.assert_close(e.point_term(Y,physical),literal,rtol=1e-14,atol=1e-14)
    q=jp.q1_corner_determinants(Y)
    torch.testing.assert_close(jp.q1_corner_determinants(world),q*torch.linalg.det(Aout),rtol=1e-12,atol=1e-14)


def test_pose_chart_rebase_preserves_physical_map_and_freezes_pose():
    Y,A,b,e,_=fixture()
    physical=torch.tensor([.01,.02,.08,-.04,.01,.02],dtype=torch.float64)
    chart,lower=jp.rebase_pose(physical,Y,.001)
    torch.testing.assert_close(jp.pose_from_chart(chart,lower),physical,rtol=1e-13,atol=1e-13)
    Y2=Y.clone();Y2[:,4,4,0]+=.03
    chart2,lower2=jp.rebase_pose(physical,Y2,.001)
    assert lower2!=lower
    torch.testing.assert_close(jp.pose_from_chart(chart2,lower2),physical,rtol=1e-13,atol=1e-13)
    cached=e.moving_feature(physical).detach()
    v=Y2.requires_grad_();total,_=e(v,physical,cached_feature=cached)
    assert torch.isfinite(torch.autograd.grad(total,v)[0]).all()
    assert not physical.requires_grad and not cached.requires_grad


def test_all_six_pose_derivatives_and_rms_include_chart_scale():
    Y,A,b,e,_=fixture()
    physical=torch.tensor([.014,-.019,.043,-.021,.012,.017],dtype=torch.float64)
    chart,lower=jp.rebase_pose(physical,Y,.001);chart=chart.requires_grad_()
    total=e(Y,jp.pose_from_chart(chart,lower))[0]
    gradient=torch.autograd.grad(total,chart)[0];fd=[]
    for j in range(6):
        d=torch.zeros_like(chart);d[j]=1e-7
        fd.append((e(Y,jp.pose_from_chart(chart.detach()+d,lower))[0]-e(Y,jp.pose_from_chart(chart.detach()-d,lower))[0])/(2e-7))
    torch.testing.assert_close(gradient,torch.stack(fd),rtol=3e-5,atol=3e-5)
    scales=jp.pose_rms_scales(chart.detach(),lower,Y,e)
    query=e.query(Y);expected=[]
    for j in range(6):
        d=torch.zeros_like(chart);d[j]=1e-6
        bp,cp=jp.pose_affine(jp.pose_from_chart(chart.detach()+d,lower))
        bm,cm=jp.pose_affine(jp.pose_from_chart(chart.detach()-d,lower))
        velocity=((query@(bp-bm).T+(cp-cm))@A.T)*512/(2e-6)
        expected.append(((velocity.square().sum(-1)[:,None]*e.mask).sum()/e.mask.sum()).sqrt())
    torch.testing.assert_close(scales,torch.stack(expected),rtol=1e-8,atol=1e-8)


def test_prewarp_can_use_original_inside_despite_old_canvas_outside():
    Y,A,b,e,_=fixture()
    # A shrinks: G(z)>1 can still map inside ORIGINAL moving image.
    e.matrix=torch.eye(2,dtype=torch.float64)*.5;e.offset=torch.tensor([.1,.1],dtype=torch.float64)
    physical=torch.tensor([.4,0.,0.,0.,0.,0.],dtype=torch.float64)
    B,c=jp.pose_affine(physical)
    raw=e.raster_queries@B.T+c
    assert bool((raw[...,0]>1).any())
    assert bool(((raw@e.matrix.T+e.offset)>=0).all() and ((raw@e.matrix.T+e.offset)<=1).all())
    feature=e.moving_feature(physical)
    assert bool(torch.isfinite(feature).all())
    assert float(feature[:,:,:,-2:].abs().sum())>0


def test_export_checks_combined_orientation_and_original_floor(tmp_path):
    Y,A,b,e,_=fixture();physical=torch.tensor([0.,0.,0.,-.2,.03,0.],dtype=torch.float64)
    Aout,bout=jp.combined_affine(A,b,physical)
    path=tmp_path/'map.npz'
    np.savez(path,vertices=Y.numpy(),boundary_reference=Y.numpy(),post_affine_matrix=Aout.numpy(),post_affine_offset=bout.numpy(),original_post_affine_matrix=A.numpy(),original_post_affine_offset=b.numpy(),pose_parameters=physical.numpy(),interpolation=np.array('p1_ac'))
    r=jp.validate_joint_export(path)
    assert r['valid'] and r['residual_minimum_corner_ratio']==pytest.approx(1.)
    assert r['original_affine_normalized_minimum_corner_ratio']==pytest.approx(np.exp(-.4))
    np.savez(path,vertices=Y.numpy(),boundary_reference=Y.numpy(),post_affine_matrix=(Aout*.01).numpy(),post_affine_offset=bout.numpy(),original_post_affine_matrix=A.numpy(),original_post_affine_offset=b.numpy(),pose_parameters=physical.numpy(),interpolation=np.array('p1_ac'))
    assert not jp.validate_joint_export(path)['valid']


def test_export_rejects_shifted_reference_even_with_positive_global_certificate(tmp_path):
    Y,A,b,e,_=fixture();physical=torch.zeros(6,dtype=torch.float64)
    shifted=Y+.1;path=tmp_path/'shifted.npz'
    np.savez(path,vertices=shifted.numpy(),boundary_reference=shifted.numpy(),post_affine_matrix=A.numpy(),post_affine_offset=b.numpy(),original_post_affine_matrix=A.numpy(),original_post_affine_offset=b.numpy(),pose_parameters=physical.numpy(),interpolation=np.array('p1_ac'))
    assert jp.certify_q1_binary_map(path)['valid']
    assert not jp.validate_joint_export(path)['valid']


def test_cached_pose_graph_and_degenerate_rms_are_rejected():
    Y,A,b,e,_=fixture();p=torch.zeros(6,dtype=torch.float64)
    cached=e.moving_feature(p).detach()
    with pytest.raises(ValueError,match='cached descriptor'):
        e(Y,p.requires_grad_(),cached_feature=cached)
    p=p.detach();chart,lower=jp.rebase_pose(p,Y,.001)
    e.matrix=torch.eye(2,dtype=torch.float64)*1e-17
    with pytest.raises(ValueError,match='degenerate'):
        jp.pose_rms_scales(chart,lower,Y,e)


def test_tiny_optimizer_budget_pose_metadata_and_stored_pair(tmp_path):
    fixed=tmp_path/'fixed.png';moving=tmp_path/'moving.png'
    image=(np.random.default_rng(7).uniform(.1,.9,(16,16))*255).astype('uint8')
    Image.fromarray(image).save(fixed);Image.fromarray(np.roll(image,1,axis=1)).save(moving)
    affine=tmp_path/'affine.npz';np.savez(affine,post_affine_matrix=np.eye(2,dtype=np.float32),post_affine_offset=np.zeros(2,dtype=np.float32))
    args=argparse.Namespace(fixed=fixed,moving=moving,affine=affine,output=tmp_path/'out.npz',
        grid_side=9,image_side=16,levels=[5,9],image_levels=[8,16],inner_steps=1,inner_steps_by_level=[1,1],pose_steps_per_level=1,
        learning_rate=.004,lr_calibration='edge',device='cpu',threads=2,minimum_jacobian=.001,
        match_weight=0.,matches=None,match_robust_scale=8.,strain_weight=3.,shape_weight=.0001,oob_weight=1.,
        cycles=1,method='analytic',loss='mind',interpolation='p1_ac',strain_model='p1_arap',precision='float64',
        image_precision='float32',mind_frame='shared_affine',preprocessing='raw_inverted',output_selection='best_full')
    result=jp.optimize_joint_pose(args)
    assert result['gradient_steps']==6 and result['pose_gradient_steps']==2 and result['residual_gradient_steps']==4
    assert result['failed_trials']==0 and result['saved_binary_certificate']['valid']
    assert result['pose_mode']=='joint_positive_affine' and jp.validate_joint_export(args.output)['valid']
    assert result['final']['total']<=result['initial']['total']+1e-12
    assert json.loads(args.output.with_suffix('.json').read_text())['landmarks_used'] is False
