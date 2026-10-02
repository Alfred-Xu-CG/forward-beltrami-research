"""Independent bounded cost, screened solve and legal-construction checks."""
import sys,json,itertools
from pathlib import Path
import numpy as np
import torch
root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src'),str(Path(__file__).parent)]
from qcopt.neural_bijection.dense import coordinated_coupled_seed as core
from independent_stiffness_probe_20261002 import triangle_data,direct_corners
torch.set_num_threads(1)

def zero_sample(image,q):
    channels,h,w=image.shape;p=q*np.array([w,h])-.5;i=np.floor(p).astype(int);t=p-i
    value=np.zeros(channels)
    for dx,dy in itertools.product((0,1),repeat=2):
        x,y=i+[dx,dy]
        if 0<=x<w and 0<=y<h:value+=image[:,y,x]*(t[0] if dx else 1-t[0])*(t[1] if dy else 1-t[1])
    return value

def cost_check(dtype=torch.float64):
    gen=torch.Generator().manual_seed(9121);side=16
    fixed=torch.rand(1,8,side,side,generator=gen,dtype=dtype)
    moving=torch.rand(1,8,side,side,generator=gen,dtype=dtype)
    mask=torch.rand(1,1,side,side,generator=gen,dtype=dtype);mask[:,:,:7,:7]=0.
    A=torch.tensor([[.1,-1.2],[.9,.2]],dtype=torch.float64);b=torch.tensor([1.1,-.1],dtype=torch.float64)
    volume=core.finite_displacement_cost(fixed,moving,mask,A,b,proposal_side=5,label_radius=2,label_batch=3)
    values=[(0,0)]+[p for p in itertools.product(range(-2,3),repeat=2) if p!=(0,0)]
    assert volume.labels.tolist()==[list(v) for v in values]
    expected=np.zeros_like(volume.costs.numpy());weights=[]
    for i,(y,x) in enumerate(itertools.product(range(1,4),repeat=2)):
        q=np.array([x,y])/4;queries=[q+np.array(o)/side for o in itertools.product((-1,0,1),repeat=2)]
        m=np.array([zero_sample(mask.numpy()[0],s)[0] for s in queries]);weights.append(m.mean())
        for k,label in enumerate(values):
            h=[]
            for s in queries:
                r=s+np.array(label)/side;world=A.numpy()@r+b.numpy()
                h.append(np.abs(zero_sample(fixed.numpy()[0],s)-zero_sample(moving.numpy()[0],r)).mean()+np.square(np.maximum(-world,0)+np.maximum(world-1,0)).sum())
            expected[k,i]=np.dot(m,h)/m.sum() if m.sum() else 0.
    error=float(np.max(np.abs(expected-volume.costs.numpy())))
    tolerance=1e-14 if dtype==torch.float64 else 2e-7
    assert error<tolerance and np.allclose(weights,volume.weights.numpy(),atol=tolerance,rtol=0)
    assert volume.weights[0]==0 and torch.equal(volume.costs[:,0],torch.zeros_like(volume.costs[:,0]))
    assert abs(sum(weights)-float(volume.denominator))<tolerance
    return volume,dict(maximum_cost_error=error,maximum_weight_error=float(np.max(np.abs(np.array(weights)-volume.weights.numpy()))),empty_patch=True,label_order=True)

def coupling_check(volume):
    fine=9;coarse=volume.proposal_side;n=coarse-2
    desired=torch.tensor([[2. if i%2 else -2.,1.] for i in range(n*n)],dtype=torch.float64)
    strong_cost=17*(volume.labels.double()[:,None,:]-desired[None]).square().sum(-1)
    volume=core.NodalCostVolume(strong_cost,volume.labels,volume.weights,volume.denominator,coarse,volume.image_side,{})
    _,_,_,_,K=triangle_data(fine-1)
    xf=torch.arange(1,fine-1,dtype=torch.float64)/(fine-1);xc=torch.arange(1,coarse-1,dtype=torch.float64)/(coarse-1)
    B=(1-(xf[:,None]-xc[None,:]).abs()*(coarse-1)).clamp_min(0);P=torch.kron(B,B);G=(P.T@K@P).numpy()
    costs=volume.costs.numpy();labels=volume.labels.numpy();weights=volume.weights.numpy();Z=float(volume.denominator);side=volume.image_side
    index=np.argmin(costs*weights,axis=0);z=labels[index].T;I=np.eye(n*n)
    solve=lambda z,c:np.linalg.solve(2*c*I+3*Z/side**2*G,2*c*z.T).T
    u=solve(z,.003);rows=[]
    def energy(index,u,c):
        return float((sum(weights[i]*costs[index[i],i] for i in range(n*n))+c*np.square(labels[index].T-u).sum())/Z+3/(2*side**2)*sum(a@G@a for a in u))
    for c in core.COUPLING_SCHEDULE:
        for iteration in range(2):
            before=energy(index,u,c)
            objectives=costs*weights+c*np.square(labels[:,None,:]-u.T[None]).sum(-1)
            index=np.argmin(objectives,axis=0);mid=energy(index,u,c);u=solve(labels[index].T,c);after=energy(index,u,c)
            assert mid<=before+1e-13 and after<=mid+1e-13
            rows.append([before,mid,after])
    actual=core.couple_cost_volume(volume,fine_side=fine)
    assert float(actual.displacement.abs().max())>.1
    error=float(np.max(np.abs(actual.displacement.numpy().reshape(2,-1)-u)))
    recorded=np.array([[b[k]['total'] for k in ['before','after_labels','after_continuous']] for b in actual.diagnostics['blocks']])
    assert error<1e-13 and np.array_equal(actual.selected_labels.numpy().reshape(-1),index)
    assert np.max(np.abs(recorded-np.array(rows)))<1e-13
    assert actual.diagnostics['two_component_screened_solves']==13 and actual.diagnostics['label_minimizations']==13
    zero=core.NodalCostVolume(torch.zeros_like(volume.costs),volume.labels,volume.weights,volume.denominator,volume.proposal_side,volume.image_side,{})
    tie=core.couple_cost_volume(zero,fine_side=fine)
    assert torch.equal(tie.displacement,torch.zeros_like(tie.displacement)) and not bool(tie.selected_labels.any())
    return dict(dense_screened_maximum_error=error,entire_alternating_trace_energy_error=float(np.max(np.abs(recorded-np.array(rows)))),all_fixed_c_blocks_decrease=True,zero_tie=True)

def construction_check():
    x,_,_,_,_=triangle_data(16);reference=x.reshape(1,17,17,2)
    # A strongly alternating raw coefficient target deliberately folds many cells.
    raw=torch.tensor([[[8.,-8.,8.],[-8.,8.,-8.],[8.,-8.,8.]],[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]],dtype=torch.float64)
    original=core.CoordinatedQ1Update.forward;observed=[];targets=[]
    def watch(self,vertices,proposal,**kwargs):
        component=0 if self.direction[0] else 1
        targets.append((component,(vertices[...,component]+proposal).clone()))
        result=original(self,vertices,proposal,**kwargs);observed.append(result.vertices.clone());return result
    core.CoordinatedQ1Update.forward=watch
    try:result=core.construct_safe_seed(reference,raw,image_side=16)
    finally:core.CoordinatedQ1Update.forward=original
    assert len(observed)==16 and result.diagnostics['raw_target_nonpositive_corner_count']>0
    minima=[]
    for v in observed:
        minimum=float(direct_corners(v).min());minima.append(minimum);assert minimum>.001
        for a,b in [(v[:,0],reference[:,0]),(v[:,-1],reference[:,-1]),(v[:,:,0],reference[:,:,0]),(v[:,:,-1],reference[:,:,-1])]:assert torch.equal(a,b)
    for component in (0,1):
        cases=[t for k,t in targets if k==component]
        assert all(torch.allclose(t,cases[0],atol=2e-16,rtol=0) for t in cases)
    assert torch.equal(result.vertices,observed[-1])
    bad=raw.clone();bad[1]=raw[0]*7/8
    try:core.construct_safe_seed(reference,bad,image_side=16)
    except RuntimeError as error:
        assert 'rounded candidate failed strict margins' in str(error)
    else:raise AssertionError('expected explicit numerical construction failure was hidden')
    return dict(raw_folded_corners=result.diagnostics['raw_target_nonpositive_corner_count'],constructed_steps=16,minimum_intermediate_corner=min(minima),frozen_target=True,all_boundaries_exact=True,extreme_two_axis_roundoff_failure_explicit=True)

def integration_check():
    import tempfile
    sys.path.insert(0,str(root/'tests'))
    from test_coordinated_shared_affine_evidence import configuration
    from tools import coordinated_real_case as app
    path=Path(tempfile.mkdtemp(prefix='coupled_independent_',dir=root/'outputs/coordinated_instance_registration'))
    cfg=configuration(path,'analytic')
    cfg.grid_side=257;cfg.image_side=512;cfg.levels=[17,257];cfg.image_levels=[128,512]
    cfg.mind_frame='shared_affine';cfg.strain_weight=3.;cfg.seed_initializer='coupled_mind'
    original_builder=core.build_coupled_mind_seed;original_call=app.Evidence.__call__
    calls=[];supplied=[]
    def builder(f,m,mask,A,b,reference):
        assert f.shape==m.shape==(1,8,128,128) and mask.shape==(1,1,128,128)
        seed=reference.clone();profile=torch.sin(torch.pi*reference[...,0])*torch.sin(torch.pi*reference[...,1])
        profile[:,0]=profile[:,-1]=0.;profile[:,:,0]=profile[:,:,-1]=0.
        seed[...,0]+=.16*profile;seed[...,1]-=.08*profile
        assert float(direct_corners(seed).min())>.001
        supplied.append(seed)
        return core.CoupledSeedResult(seed,reference.new_zeros((1,65,65,2)),{'independent_mock_initializer':True})
    def observe(self,vertices):
        calls.append((self.fixed.shape[-1],vertices.detach().clone()))
        return original_call(self,vertices)
    core.build_coupled_mind_seed=builder;app.Evidence.__call__=observe
    try:report=app.optimize(cfg)
    finally:core.build_coupled_mind_seed=original_builder;app.Evidence.__call__=original_call
    assert len(supplied)==1 and len(calls)==report['objective_evaluations']
    assert report['seed_record']['original_E_increased'] and not report['seed_record']['original_E_seed_rejection']
    # The first coarse-stage objective sees the actual high-energy seed, not identity.
    first=next(vertices for side,vertices in calls if side==128)
    assert torch.equal(first,supplied[0])
    best=min([report['initial']['total'],report['seed_record']['original_full_objective']['total']]+[s['accepted_full_total'] for s in report['stages']])
    assert report['final']['total']==best
    assert report['seed_objective_evaluations']==1 and report['gradient_steps']==4
    with np.load(cfg.output) as saved:
        assert np.array_equal(saved['initializer_vertices'],supplied[0].numpy())
    return dict(mock_seed_higher_E=True,coarse_refinement_really_starts_from_seed=True,identity_seed_prefix_selector=True,observed_objective_calls=len(calls),reported_objective_calls=report['objective_evaluations'],gradient_steps=report['gradient_steps'],extra_seed_objective_calls=1)

if __name__=='__main__':
    if '--postrun' in sys.argv:
        from independent_stiffness_probe_20261002 import postrun_checks
        print(json.dumps(postrun_checks(('coupled_seed',))))
    else:
        volume,cost=cost_check()
        _,mixed=cost_check(torch.float32)
        print(json.dumps(dict(cost=cost,mixed_float32_cost=mixed,coupling=coupling_check(volume),construction=construction_check(),integration=integration_check())))
