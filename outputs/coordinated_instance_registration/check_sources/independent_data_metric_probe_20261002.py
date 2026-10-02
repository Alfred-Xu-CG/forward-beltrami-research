"""Literal tiny data-metric/operator/descent checks; no annotations or GPU runs."""
import json
import math
import sys
import time
import copy
import tempfile
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests'),str(Path(__file__).parent)]
from tools.coordinated_real_case import Evidence
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from qcopt.neural_bijection.dense import coordinated_data_metric as module
from independent_joint_pose_probe_20261002 import pixels,corners
torch.set_num_threads(1)

def identity(side=5):
    yy,xx=np.meshgrid(np.arange(side)/(side-1),np.arange(side)/(side-1),indexing='ij')
    return torch.tensor(np.stack((xx,yy),-1)[None],dtype=torch.float64)

def rows(side,query):
    """Literal hand-assembled fine-ac rows, no production evaluator."""
    result=np.zeros((len(query),(side-2)**2));n=side-1
    for i,(x,y) in enumerate(query):
        col=min(math.floor(n*x),n-1);row=min(math.floor(n*y),n-1);u=n*x-col;v=n*y-row
        triples=([(row,col,1-u),(row,col+1,u-v),(row+1,col+1,v)] if v<=u else
                 [(row,col,1-v),(row+1,col+1,u),(row+1,col,v-u)])
        for r,c,w in triples:
            if 0<r<n and 0<c<n:result[i,(r-1)*(side-2)+c-1]+=w
    return result

def fivepoint(side):
    n=side-2;K=np.zeros((n*n,n*n))
    for y in range(n):
        for x in range(n):
            i=y*n+x;K[i,i]=4
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                if 0<=x+dx<n and 0<=y+dy<n:K[i,(y+dy)*n+x+dx]=-1
    return K

def literal_slopes(features,query):
    """Bilinear zero-padding derivative, at actual float32 unnormalized grid.

    Emulates scalar grid arithmetic, not a finite difference through rounding.
    """
    image=np.asarray(features,dtype=np.float32);channels,h,w=image.shape
    normalized=(2*np.asarray(query,dtype=np.float64)-1).astype(np.float32)
    xy=((normalized+np.float32(1))*np.array([w,h],dtype=np.float32)-np.float32(1))/np.float32(2)
    out=np.zeros((len(query),channels,2))
    for i,(x,y) in enumerate(xy):
        ix,iy=math.floor(float(x)),math.floor(float(y));u=float(x)-ix;v=float(y)-iy
        at=lambda a,b:image[:,b,a].astype(float) if 0<=a<w and 0<=b<h else np.zeros(channels)
        a,b,c,d=at(ix,iy),at(ix+1,iy),at(ix,iy+1),at(ix+1,iy+1)
        out[i,:,0]=w*((1-v)*(b-a)+v*(d-c))
        out[i,:,1]=h*((1-u)*(c-a)+u*(d-b))
    return out

def fixture():
    rng=np.random.default_rng(1803);side=9
    fixed=torch.tensor(rng.uniform(.1,.9,(1,1,side,side)),dtype=torch.float32)
    moving=torch.tensor(rng.uniform(.1,.9,(1,1,side,side)),dtype=torch.float32)
    mask=torch.tensor(rng.choice([0.,.25,.5,1.],(1,1,side,side)),dtype=torch.float32)
    a=torch.tensor([[1.31,.12],[-.07,1.14]],dtype=torch.float64);b=torch.tensor([-.14,-.11],dtype=torch.float64)
    q=torch.tensor([[0,0],[1,1],[.5,0],[.13,.27],[.5,.5],[.71,.26],[.99,.43]],dtype=torch.float64)
    p=(q+torch.tensor([.023,-.017])).clamp(0,1);conf=torch.tensor([.1,.3,.2,.3,.4,.1,.2],dtype=torch.float64)
    matches=ImageCorrespondences(q,p,conf,pixel_scale=512,robust_scale=8)
    e=Evidence(fixed,moving,a,b,'mind',3.,1.,1e-4,fixed_mask=mask,interpolation='p1_ac',matches=matches,
        match_weight=.2,strain_model='p1_arap',mind_frame='shared_affine')
    Y=identity();Y[:,1:-1,1:-1,0]+=.012;Y[:,1:-1,1:-1,1]-=.009
    e.prepare_fixed_p1_sampling(5,5,dtype=torch.float64,device='cpu');matches.prepare_fixed_p1_sampling(5,5)
    return Y,e

def sampler_check():
    _,item=fixture();q=np.array([[-.21,.3],[-.01,.41],[0,.5],[1,.5],[1.01,.43],[1.21,.3],[.5,.5],[.5/9,4.5/9],[.13527,.31782]])
    query=torch.tensor(q.reshape(1,len(q),1,2),dtype=torch.float64,requires_grad=True)
    sample=lambda x:F.grid_sample(item.moving_feature,(2*x-1).float(),align_corners=False,padding_mode='zeros',mode='bilinear')
    jac=torch.autograd.functional.jacobian(sample,query).reshape(8,len(q),len(q),2).numpy()
    slopes=np.stack([jac[:,i,i] for i in range(len(q))]);off=jac.copy()
    for i in range(len(q)):off[:,i,i]=0
    assert not np.any(off)
    expected=literal_slopes(item.moving_feature[0].numpy(),q)
    discrepancy=float(abs(slopes-expected).max());assert discrepancy<1e-5
    assert np.array_equal(slopes[[0,5]],np.zeros_like(slopes[[0,5]]))
    assert np.any(slopes[2,:,0]) and np.any(slopes[3,:,0])
    return dict(literal_bilinear_vs_actual_AD_maximum_error=discrepancy,query_jacobian_offdiagonal_exact_zero=True,
        zero_padding_boundary_and_far_outside_checked=True,rounded_grid_cast_and_alignFalse_included=True)

def metric_check():
    Y,item=fixture();results=[]
    for direction in ((1.,0.),(0.,1.)):
        metric=module.DataAwareMetric(item,Y,direction);e=np.array(direction);q=metric.query.numpy().reshape(-1,2)
        slopes=literal_slopes(item.moving_feature[0].numpy(),q)@e
        warped=F.grid_sample(item.moving_feature,(2*metric.query-1).float(),align_corners=False,padding_mode='zeros')
        residual=(item.fixed_feature-warped).abs().numpy()[0].reshape(8,-1).T.astype(float)
        m=item.mask.numpy().reshape(-1).astype(float);Z=float(item.denominator)
        DI=(slopes**2/np.maximum(residual,1e-3)).sum(-1)*m/(8*Z)
        image_error=float(abs(DI-metric.image_diagonal.numpy()).max());assert np.allclose(DI,metric.image_diagonal.numpy(),rtol=3e-6,atol=3e-7)
        query=metric.query.detach().requires_grad_(True)
        full_jac=torch.autograd.functional.jacobian(lambda v:F.grid_sample(item.moving_feature,(2*v-1).float(),align_corners=False,padding_mode='zeros'),query).reshape(8,81,81,2).numpy()
        scalar_slopes=np.stack([full_jac[:,i,i]@e for i in range(81)])
        exact_DI=(scalar_slopes**2/np.maximum(residual,1e-3)).sum(-1)*m/(8*Z)
        ad_diagonal_error=float(abs(exact_DI-metric.image_diagonal.numpy()).max());assert ad_diagonal_error<2e-12
        image_relative_error=float(np.max(abs(DI-metric.image_diagonal.numpy())/np.maximum(1.,abs(metric.image_diagonal.numpy()))))
        A=item.matrix.numpy();b=item.offset.numpy();world=q@A.T+b;ae=A@e
        DO=2*m/Z*(((world<0)|(world>1))*ae**2).sum(-1)
        assert np.array_equal(DO,metric.oob_diagonal.numpy())
        T=rows(5,pixels(9).reshape(-1,2));L=rows(5,item.matches.source.numpy()[0]);n=9
        mapped=item.matches.fixed_p1_evaluator(Y).numpy()[0];z=(mapped-item.matches.target.numpy()[0])@A.T*64;a=64*ae
        norm=np.sqrt(1+(z*z).sum(-1));beta=.2*item.matches.weights.numpy()*((a*a).sum()/norm-((z*a).sum(-1)**2)/(norm**3))
        point_error=float(abs(beta-metric.point_diagonal.numpy()).max());assert np.allclose(beta,metric.point_diagonal.numpy(),atol=5e-13,rtol=3e-14)
        zero=torch.zeros((1,3,3),dtype=torch.float64,requires_grad=True)
        make=lambda c:Y+F.pad(c,(1,1,1,1))[...,None]*Y.new_tensor(direction)
        point_hessian=torch.autograd.functional.hessian(lambda c:.2*item.matches(make(c),item.matrix,'p1_ac'),zero).reshape(n,n).numpy()
        assert np.allclose(point_hessian,L.T@(beta[:,None]*L),atol=2e-12,rtol=2e-13)
        outside_hessian=torch.autograd.functional.hessian(lambda c:item.image_terms(make(c))[1],zero).reshape(n,n).numpy()
        assert np.allclose(outside_hessian,T.T@(DO[:,None]*T),atol=2e-14,rtol=2e-13)
        # Use independently checked AD-based diagonal here to isolate operator error.
        H=3*fivepoint(5)+T.T@(metric.raster_diagonal.numpy()[:,None]*T)+L.T@(metric.point_diagonal.numpy()[:,None]*L)
        eye=torch.eye(n,dtype=torch.float64);actual=torch.stack([metric.apply(v.reshape(1,3,3)).flatten() for v in eye],1).numpy()
        operator_error=float(abs(actual-H).max());assert operator_error<2e-12 and abs(actual-actual.T).max()<2e-12
        eigenvalues=np.linalg.eigvalsh(H);assert eigenvalues[0]>0
        gamma=np.trace(H-3*fivepoint(5))/n;assert abs(metric.gamma-gamma)<2e-12
        assert np.array_equal((L*L).sum(-1)[:3],np.zeros(3))
        assert np.allclose(metric.image_rows.row_norm_squared.numpy(),(T*T).sum(-1),atol=1e-15)
        rng=np.random.default_rng(193);c=rng.normal(size=n);cot=rng.normal(size=len(T))
        forward=metric.image_rows.apply(torch.tensor(c.reshape(1,3,3))).numpy();transpose=metric.image_rows.transpose(torch.tensor(cot)).numpy().reshape(-1)
        assert np.allclose(forward,T@c,atol=2e-15) and np.allclose(transpose,T.T@cot,atol=2e-15)
        assert abs(forward@cot-c@transpose)<5e-14
        rhs=torch.tensor(rng.normal(size=(1,3,3)),dtype=torch.float64)
        precondition_error=float(abs(metric.precondition(rhs).numpy().reshape(-1)-np.linalg.solve(3*fivepoint(5)+gamma*np.eye(n),rhs.numpy().reshape(-1))).max());assert precondition_error<2e-14
        x,status=module.pcg(metric.apply,rhs,metric.precondition,rtol=.1,max_steps=20)
        actual_residual=np.linalg.norm(H@x.numpy().reshape(-1)-rhs.numpy().reshape(-1))/np.linalg.norm(rhs.numpy())
        assert status['converged'] and actual_residual<=.1 and abs(actual_residual-status['relative_residual'])<5e-14
        capped,cstatus=module.pcg(metric.apply,rhs,metric.precondition,rtol=1e-16,max_steps=1)
        assert cstatus['reason']=='iteration_cap' and not cstatus['converged'] and float((-rhs*capped).sum())<0
        results.append(dict(direction=direction,image_diagonal_literal_error=image_error,image_diagonal_scaled_literal_error=image_relative_error,
            image_diagonal_full_Jacobian_AD_error=ad_diagonal_error,point_curvature_error=point_error,
            explicit_H_error=operator_error,minimum_eigenvalue=float(eigenvalues[0]),gamma=gamma,preconditioner_error=precondition_error,
            pcg_iterations=status['iterations'],actual_relative_residual=actual_residual))
    return results

class Ticks:
    def __init__(self):self.now=0.
    def __call__(self):self.now+=.1;return self.now

class RecordEvidence:
    def __init__(self,item,flat=False):self.item=item;self.records=[];self.flat=flat
    def __getattr__(self,name):return getattr(self.item,name)
    def __call__(self,Y):
        if self.flat:
            total=Y[0,2,2,0]*.1+1e20;parts={}
        else:total,parts=self.item(Y)
        self.records.append((Y.detach().clone(),float(total.detach()),torch.is_grad_enabled()))
        return total,parts

def descent_checks():
    Y,item=fixture();result={};norm=16.;floor=.001+.05*(corners(Y.numpy()[0])-.001)
    for name,fn,extra in [('metric',module.optimize_data_metric_fiber,{}),('adam',module.optimize_timed_adam_fiber,{'learning_rate':.00025})]:
        objective=RecordEvidence(item)
        out=fn(Y,objective,seconds=.25,clock=Ticks(),**extra)
        assert out.counts['gradient_steps']==2 and out.stop_reason=='time_budget' and out.elapsed_seconds>=.25
        assert out.final_objective<=out.initial_objective and out.minimum_contracted_slack>0
        actual=corners(out.vertices.numpy()[0]);assert np.all(actual>floor)
        assert torch.equal(out.vertices[:,[0,-1]],Y[:,[0,-1]]) and torch.equal(out.vertices[:,:,[0,-1]],Y[:,:,[0,-1]])
        assert abs(float(item(out.vertices)[0])-out.final_objective)<1e-15
        if name=='metric':
            for step in out.trace:
                assert len(step['trials'])<=13
                for trial in step['trials']:
                    if trial['accepted']:assert trial['total']<step['total'] and trial['total']<=trial['armijo_rhs'] and trial['geometry_valid']
            assert out.counts['descriptor_forwards']==out.counts['metric_refreshes'] and out.counts['descriptor_vjps']==8*out.counts['metric_refreshes']
        assert out.counts['objective_evaluations']==len(objective.records)
        result[name]=dict(counts=out.counts,actual_contracted_slack=float((actual-floor).min()),stop=out.stop_reason)
    rounded=RecordEvidence(item,flat=True);out=module.optimize_data_metric_fiber(Y,rounded,seconds=2.,clock=Ticks())
    assert out.stop_reason=='line_search_exhausted' and out.counts['accepted_steps']==0 and torch.equal(out.vertices,Y)
    assert len(out.trace)==1 and len(out.trace[0]['trials'])==13 and out.counts['backtracks']==12
    assert all(t['total']==out.initial_objective==t['armijo_rhs'] and not t['accepted'] for t in out.trace[0]['trials'])
    assert out.counts['trial_evaluations']==13 and out.counts['objective_evaluations']==15
    result['rounded_equal_objective_rejected']=dict(trials=13,backtracks=12,last_accepted_map_retained=True)
    # Clock injection: deliberate setup consumes the entire nominal two seconds.
    clock=Ticks();real_geometry=module.MetricGeometry
    def slow_geometry(*args,**kwargs):
        geometry=real_geometry(*args,**kwargs);clock.now+=3.;return geometry
    module.MetricGeometry=slow_geometry
    try:timed=module.optimize_data_metric_fiber(Y,item,seconds=2.,clock=clock)
    finally:module.MetricGeometry=real_geometry
    assert timed.counts['gradient_steps']==0 and timed.stop_reason=='time_budget' and timed.elapsed_seconds>3.
    result['setup_charged_before_first_gradient']=dict(elapsed=timed.elapsed_seconds,overrun=timed.time_overrun_seconds)
    at_floor=identity(3);at_floor[...,0]*=.001
    try:module._setup(at_floor,(1.,0.),2.,.001)
    except ValueError:pass
    else:raise AssertionError('incoming equality with strict floor accepted')
    # OOB branch derivative is zero at precisely 0 and 1.
    v=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
    h=torch.autograd.functional.hessian(lambda z:(F.relu(-z)+F.relu(z-1)).square().sum(),v)
    assert torch.equal(h,torch.zeros_like(h))
    def linear(vertices):
        loss=-vertices[:,1:-1,1:-1,0].sum();return loss,dict(image=loss)
    active=module.optimize_timed_adam_fiber(identity(),linear,learning_rate=1.,seconds=.25,clock=Ticks())
    assert active.stop_reason=='time_budget' and active.counts['failed_trials']==0 and active.final_objective<active.initial_objective
    assert abs(active.minimum_contracted_slack)<1e-13 and corners(active.vertices.numpy()[0]).min()>.001
    result['original_adam_active_bound_preserved']=dict(contracted_slack=active.minimum_contracted_slack,
        original_minimum=float(corners(active.vertices.numpy()[0]).min()))
    # Failure after three completed image VJPs must not erase their work counts.
    counts=module._counts();counts['metric_refreshes']=1;real_grad=torch.autograd.grad;calls=0
    def interrupted_grad(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==4:raise RuntimeError('independent injected fourth VJP failure')
        return real_grad(*args,**kwargs)
    torch.autograd.grad=interrupted_grad
    try:
        try:module.DataAwareMetric(item,Y,(1.,0.),work_counts=counts)
        except RuntimeError as error:assert 'injected fourth' in str(error)
        else:raise AssertionError('partial metric failure not exercised')
    finally:torch.autograd.grad=real_grad
    assert counts['descriptor_forwards']==1 and counts['descriptor_vjps']==3 and counts['metric_refreshes']==1
    result['partial_metric_work_retained']=dict(forwards=1,completed_vjps=3,attempted_refreshes=1)
    tick=time.perf_counter();real_time=module.optimize_data_metric_fiber(Y,item,seconds=.02);wall=time.perf_counter()-tick
    assert 0<real_time.elapsed_seconds<=wall+.002 and real_time.time_overrun_seconds==max(0,real_time.elapsed_seconds-.02)
    if real_time.stop_reason=='time_budget':assert real_time.elapsed_seconds>=.02
    result['actual_CPU_clock_smoke']=dict(seconds=.02,elapsed=real_time.elapsed_seconds,external_wall=wall,
        overrun=real_time.time_overrun_seconds,stop=real_time.stop_reason,gradients=real_time.counts['gradient_steps'])
    return result

def wrapper_checks():
    from tests.test_coordinated_shared_affine_evidence import configuration
    from tools import coordinated_real_case as original
    from tools import coordinated_data_metric_application as app
    source=json.loads((ROOT/'outputs/coordinated_instance_registration/match_fusion_all50_t23/fusion/predictions.json').read_text())['rows'][0]['configuration']
    prod=argparse.Namespace(**source);app._validate_configuration(prod)
    for key,value in [('grid_side',129),('image_side',1024),('learning_rate',.1),('shape_weight',.1)]:
        changed=copy.deepcopy(prod);setattr(changed,key,value)
        try:app._validate_configuration(changed)
        except ValueError:pass
        else:raise AssertionError('production final-only configuration changed')
    with tempfile.TemporaryDirectory(prefix='independent_data_metric_',dir=ROOT/'outputs/coordinated_instance_registration') as folder:
        directory=Path(folder);cfg=configuration(directory,'analytic');cfg.mind_frame='shared_affine'
        cfg.levels=[3,3,5,5,cfg.grid_side];cfg.image_levels=[8,8,8,8,cfg.image_side]
        full=[];complete=original.optimize(cfg,lambda y,s,t:full.append((y.detach().clone(),copy.deepcopy(s))))
        old_validate=app._validate_configuration;old_functions={n:getattr(app,n) for n in ('optimize_data_metric_fiber','optimize_timed_adam_fiber')}
        old_call=original.Evidence.__call__;calls=[];stage_anchors=[]
        def counted(self,Y):calls.append(Y.detach().clone());return old_call(self,Y)
        def bounded(name):
            def run(anchor,*args,**kwargs):
                stage_anchors.append(anchor.detach().clone());return old_functions[name](anchor,*args,clock=Ticks(),**kwargs)
            return run
        app._validate_configuration=lambda args:None
        try:
            cfg.output=directory/'prefix.npz';prefix=app.extract_prefix(cfg)
            prefix_report=json.loads(Path(prefix['prefix_report']).read_text())
            with np.load(prefix['prefix_start']) as z:raw=z['vertices'].copy()
            assert np.array_equal(raw,full[7][0].numpy()) and prefix['gradient_steps']==8 and prefix['accepted_stages']==8
            assert prefix_report['stages']==json.loads(json.dumps(complete['stages'][:8]))
            for n in old_functions:setattr(app,n,bounded(n))
            original.Evidence.__call__=counted;reports=[];starts=[]
            for arm in ('adam','data_metric'):
                calls.clear();stage_anchors.clear();cfg.output=directory/(arm+'.npz')
                report=app.optimize_timed_final(cfg,prefix_best=prefix['prefix_best'],prefix_start=prefix['prefix_start'],arm=arm)
                reports.append(report);starts.append(stage_anchors[0].clone())
                assert np.array_equal(stage_anchors[0].numpy(),raw) and len(stage_anchors)==2
                assert report['suffix_objective_evaluations']==len(calls)
                assert report['gradient_steps']==8+sum(s['counts']['gradient_steps'] for s in report['stages'])
                assert report['saved_binary_certificate']['valid'] and not report['landmarks_used']
                assert report['final']['total']==min([report['prefix_best_total']]+[s['accepted_full_total'] for s in report['stages']])
                assert report['terminal_budget']['seconds_per_axis']==2. and report['terminal_budget']['stage_records']==report['stages']
                if arm=='adam':assert all(s['physical_lr']==cfg.learning_rate*(cfg.levels[0]-1)/(cfg.levels[-1]-1) for s in report['stages'])
            assert torch.equal(starts[0],starts[1]) and reports[0]['initial']==reports[1]['initial'] and reports[0]['prefix_best_total']==reports[1]['prefix_best_total']
        finally:
            app._validate_configuration=old_validate;original.Evidence.__call__=old_call
            for n,f in old_functions.items():setattr(app,n,f)
    return dict(production_final257_fixed512_configuration_checked=True,tiny_raw_eighth_stage_exact=True,
        common_numerical_suffix_start_and_initial_E_exact=True,best_prefix_vs_suffix_selector_checked=True,
        all_complete_objective_calls_counted=True,default_Adam_learning_rate_preserved=True,
        tiny_wrapper_scope='small grid with production configuration guard bypassed only in isolated checker; deterministic clocks; no GPU')

if __name__=='__main__':
    started=time.perf_counter();result=dict(sampler=sampler_check(),metric=metric_check(),descent=descent_checks(),wrapper=wrapper_checks())
    result['elapsed_seconds']=time.perf_counter()-started
    print(json.dumps(result,indent=2))
