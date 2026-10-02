"""Bounded independent render, correspondence-VJP and incoming-winner checks."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests'),str(Path(__file__).parent)]
from tools.coordinated_refreshed_matches import render_incumbent,refreshed_point_record
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools import coordinated_real_case as application
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from independent_joint_pose_probe_20261002 import vector_p1,corners,pixels
from test_coordinated_shared_affine_evidence import configuration

def identity(side):
    y,x=np.meshgrid(np.arange(side)/(side-1),np.arange(side)/(side-1),indexing='ij')
    return np.stack((x,y),-1)

def border(raster,unit):
    h,w=raster.shape;x=np.clip(unit[...,0]*w-.5,0,w-1);y=np.clip(unit[...,1]*h-.5,0,h-1)
    i=x.astype(int);j=y.astype(int);u=x-i;v=y-j
    return (1-u)*(1-v)*raster[j,i]+u*(1-v)*raster[j,np.minimum(i+1,w-1)]+u*v*raster[np.minimum(j+1,h-1),np.minimum(i+1,w-1)]+(1-u)*v*raster[np.minimum(j+1,h-1),i]

def p1_rows(side,q):
    result=np.zeros((len(q),side*side))
    for index,(x,y) in enumerate(q):
        col=min(int(x*(side-1)),side-2);row=min(int(y*(side-1)),side-2);u=x*(side-1)-col;v=y*(side-1)-row
        ids=[row*side+col,row*side+col+1,(row+1)*side+col+1,(row+1)*side+col]
        result[index,ids]=[1-u,u-v,v,0] if v<=u else [1-v,0,u,v-u]
    return result

def main():
    torch.set_num_threads(1);results={}
    a=np.array([[1.08,.13],[-.04,.96]],np.float32);b=np.array([-.06,.024],np.float32)
    at=torch.from_numpy(a);bt=torch.from_numpy(b)
    moving=torch.rand(1,1,512,512,generator=torch.Generator().manual_seed(8621))
    v=identity(257);actual=render_incumbent(moving,torch.tensor(v)[None],at,bt)
    old=warp_moving_to_fixed(moving,at,bt,height=512,width=512);assert torch.equal(actual,old)
    deformation=.033*np.sin(np.pi*v[...,0])*np.sin(np.pi*v[...,1]);v[1:-1,1:-1,0]+=deformation[1:-1,1:-1]
    assert corners(v).min()>.8
    literal=vector_p1(v,pixels(512));grid32=2*(torch.tensor(literal).float()@at.T+bt)-1
    expected=border(moving[0,0].numpy(),(grid32.double().numpy()+1)/2)
    actual=render_incumbent(moving,torch.tensor(v)[None],at,bt)[0,0].numpy()
    render_error=float(abs(actual-expected).max());assert render_error<2e-7
    twice=F.grid_sample(old,(2*torch.tensor(literal)[None]-1).float(),padding_mode='border',align_corners=False)[0,0].numpy()
    double_difference=float(abs(twice-actual).max());assert double_difference>.05
    results.update(identity_render_exact_old_helper=True,nonidentity_literal_border_error=render_error,double_interpolation_difference=double_difference)
    # Unequal confidences, duplicate points, affine coupling and a target that
    # becomes ineligible only AFTER f0(s); no inverse or offset in point error.
    v=identity(3);v[1,1,0]+=.12
    s=np.array([[.3,.15],[.61,.41],[.75,.2],[.2,.7],[.4,.65],[.42,.12],[.6,.3],[.2,.2],[.9,.5]])
    q=np.array([[.2,.16],[.4,.6],[.2,.16],[.4,.3],[.67,.73],[.81,.55],[.57,.15],[.1,.6],[.7,.45]])
    a=np.array([[1.2,0],[.07,.9]],np.float32);b=np.array([-.1,.02],np.float32);c=np.linspace(.2,.9,len(q))
    rec=refreshed_point_record(q*512-.5,s*512-.5,c,a,b,torch.tensor(v)[None])
    p=vector_p1(v,s);assert np.max(abs(np.asarray(rec['target_points_unit'])-p))<1e-15 and rec['status']=='ok'
    eligible=((p@a.astype(float).T+b>=0)&(p@a.astype(float).T+b<=1)).all(-1)
    assert eligible.sum()==8 and ((s@a.T+b>=0)&(s@a.T+b<=1)).all()
    weight=c*eligible;weight/=weight.sum();sampling=p1_rows(3,q)
    candidate=v.copy();candidate[1,1]+=[-.023,.019];flat=candidate.reshape(-1,2)
    z=(sampling@flat-p)@a.astype(float).T*64
    norm=np.sqrt(1+(z*z).sum(-1));expected_loss=float(((norm-1)*weight).sum())
    expected_gradient=sampling.T@((weight/norm)[:,None]*z@a.astype(float)*64)
    variable=torch.tensor(candidate[None],requires_grad=True)
    obj=ImageCorrespondences(torch.tensor(q),torch.tensor(p),torch.tensor(c*eligible))
    loss=obj(variable,torch.tensor(a.astype(float)),'p1_ac');gradient=torch.autograd.grad(loss,variable)[0].numpy()[0].reshape(-1,2)
    loss_error=abs(float(loss)-expected_loss);gradient_error=float(abs(gradient-expected_gradient).max())
    assert loss_error<2e-14 and gradient_error<2e-13
    native_left=(sampling@flat@a.astype(float).T+b)-(p@a.astype(float).T+b)
    assert np.max(abs(native_left-z/64))<3e-16
    results.update(point_loss_error=loss_error,point_vertex_VJP_error=gradient_error,post_P1_eligible_count=int(eligible.sum()))
    # Scoped selector fixture: exact differentiable minimum at a NONidentity
    # incoming map. Exercises production scheduling/selection, not image benefit.
    with tempfile.TemporaryDirectory(prefix='independent-refreshed-') as folder:
        directory=Path(folder);cfg=configuration(directory,'analytic');cfg.mind_frame='shared_affine'
        with np.load(cfg.affine) as saved:a,b=saved['post_affine_matrix'],saved['post_affine_offset']
        reference=identity(cfg.grid_side);incoming=reference.copy();incoming[1:-1,1:-1,0]+=.007
        path=directory/'incoming.npz';np.savez(path,vertices=incoming[None],boundary_reference=reference[None],post_affine_matrix=a,post_affine_offset=b,interpolation=np.asarray('p1_ac'))
        target=torch.tensor(incoming[None]);seen=[]
        def minimum_at_incoming(self,vertices):
            seen.append(vertices.detach().clone());loss=((vertices-target.to(vertices))**2).sum();zero=loss*0
            return loss,dict(image=loss,strain=zero,shape=zero,oob=zero,outside_fraction=zero)
        with patch.object(application.Evidence,'__call__',minimum_at_incoming):result=application.optimize(cfg,initial_map=path)
        assert torch.equal(seen[0],target) and result['initial']['total']==result['final']['total']==0
        assert result['selected_stage'] is None and result['gradient_steps']==4
        with np.load(cfg.output) as saved:assert np.array_equal(saved['vertices'],incoming[None]) and np.array_equal(saved['boundary_reference'],reference[None])
        results.update(nonidentity_incoming_first_evaluation_exact=True,incoming_retained_as_best_full=True,fixture_new_gradients=result['gradient_steps'])
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
