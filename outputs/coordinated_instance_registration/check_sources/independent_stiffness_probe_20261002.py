"""Independent triangle assembly/ARAP majorizer and tensor-Galerkin check."""
import sys,json,math
from pathlib import Path
import numpy as np
import torch
root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src')]
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.coordinated_stiffness import DirichletGalerkinStiffness,optimize_stiffness_fiber
torch.set_num_threads(1)

def triangle_data(n):
    yy,xx=np.meshgrid(np.arange(n+1)/n,np.arange(n+1)/n,indexing='ij')
    x=np.stack((xx,yy),-1).reshape(-1,2)
    faces=[]
    for y in range(n):
        for c in range(n):
            a=y*(n+1)+c;b=a+1;d=a+n+1;cc=d+1
            faces.extend(((a,b,cc),(a,cc,d)))
    f=np.array(faces);edges=(x[f[:,1:]]-x[f[:,:1]]).transpose(0,2,1)
    inv=np.linalg.inv(edges);area=np.linalg.det(edges)/2
    grads=inv.transpose(0,2,1)@np.array([[-1.,1.,0.],[-1.,0.,1.]])
    K=np.zeros((len(x),len(x)))
    for ids,g,w in zip(f,grads,area):K[np.ix_(ids,ids)]+=w*g.T@g
    inside=np.array([y*(n+1)+c for y in range(1,n) for c in range(1,n)])
    return torch.tensor(x),torch.tensor(f),torch.tensor(inv),torch.tensor(inside),torch.tensor(K[np.ix_(inside,inside)])

def jac(y,faces,inv):
    edge=(y[faces[:,1:]]-y[faces[:,:1]]).transpose(-1,-2)
    return edge@inv

def frozen_checks():
    errors=[]
    for n in [2,3,4,8]:
        x,f,inv,inside,K=triangle_data(n)
        y=x.clone();y[inside]+=torch.randn(len(inside),2,generator=torch.Generator().manual_seed(n),dtype=torch.float64)*(.025/n)
        J=jac(y,f,inv);U,S,V=torch.linalg.svd(J);rot=(U@V).detach()
        v=y.reshape(1,n+1,n+1,2).requires_grad_()
        frozen=.5*(jac(v.reshape(-1,2),f,inv)-rot).square().sum((-1,-2)).mean()
        true=p1_arap_energy(v)
        a=torch.autograd.grad(frozen,v,retain_graph=True)[0];b=torch.autograd.grad(true,v)[0]
        assert torch.allclose(frozen,true,atol=1e-15,rtol=1e-12)
        assert torch.allclose(a,b,atol=1e-14,rtol=1e-12)
        def energy(z):
            change=torch.zeros_like(y).index_add(0,inside,torch.stack((z,torch.zeros_like(z)),-1))
            return .5*(jac(y+change,f,inv)-rot).square().sum((-1,-2)).mean()
        H=torch.autograd.functional.hessian(energy,torch.zeros(len(inside),dtype=torch.float64))
        L=torch.diag(torch.full((n-1,),2.,dtype=torch.float64))-torch.diag(torch.ones(n-2,dtype=torch.float64),1)-torch.diag(torch.ones(n-2,dtype=torch.float64),-1)
        five=torch.kron(L,torch.eye(n-1,dtype=torch.float64))+torch.kron(torch.eye(n-1,dtype=torch.float64),L)
        assert torch.allclose(H,K,atol=1e-13,rtol=1e-13)
        assert torch.allclose(K,five,atol=1e-13,rtol=1e-13)
        errors.append(dict(fine_intervals=n,touching_value=float(abs(frozen-true)),touching_gradient=float((a-b).abs().max()),frozen_hessian_vs_FE=float((H-K).abs().max()),FE_vs_fivepoint=float((K-five).abs().max())))
    return errors

def galerkin_checks():
    rows=[]
    for nc in [2,3,4]:
        for r in [1,2,4]:
            nf=nc*r
            _,_,_,_,K=triangle_data(nf)
            xf=torch.arange(1,nf,dtype=torch.float64)/nf;xc=torch.arange(1,nc,dtype=torch.float64)/nc
            B=(1-(xf[:,None]-xc[None,:]).abs()*nc).clamp_min(0)
            P=torch.kron(B,B);actual=P.T@K@P
            k=torch.arange(1,nc,dtype=torch.float64);theta=math.pi*k/nc
            sine=(2/nc)**.5*torch.sin(k[:,None]*k[None,:]*math.pi/nc)
            S=torch.kron(sine,sine)
            stiff=(2-2*torch.cos(theta))/r
            mass=(2*r*r+1+(r*r-1)*torch.cos(theta))/(3*r)
            eigen=(stiff[:,None]*mass[None,:]+mass[:,None]*stiff[None,:]).reshape(-1)
            expected=S@torch.diag(eigen)@S.T
            rhs=torch.randn((nc-1)**2,generator=torch.Generator().manual_seed(nc*10+r),dtype=torch.float64)
            spectral=S@((S.T@rhs)/(3*eigen));direct=torch.linalg.solve(3*actual,rhs)
            production=DirichletGalerkinStiffness(nf+1,nc+1)
            field=rhs.reshape(1,nc-1,nc-1)
            production_solution=production.solve(field).reshape(-1)
            production_action=production.apply(field).reshape(-1)
            production_prolong=production.prolong(field)[:,1:-1,1:-1].reshape(-1)
            assert torch.allclose(actual,expected,atol=2e-14,rtol=2e-14)
            assert torch.allclose(spectral,direct,atol=2e-14,rtol=2e-14)
            assert torch.allclose(production_solution,direct,atol=2e-14,rtol=2e-14)
            assert torch.allclose(production_action,3*actual@rhs,atol=2e-14,rtol=2e-14)
            assert torch.allclose(production_prolong,P@rhs,atol=2e-14,rtol=2e-14)
            rows.append(dict(coarse_intervals=nc,refinement=r,galerkin_max=float((actual-expected).abs().max()),inverse_max=float((spectral-direct).abs().max()),production_inverse_max=float((production_solution-direct).abs().max()),production_action_max=float((production_action-3*actual@rhs).abs().max()),production_prolong_max=float((production_prolong-P@rhs).abs().max()),gradient_direction_pairing=float(-rhs@spectral)))
    return rows

def direct_corners(y):
    a,b,c,d=y[:,:-1,:-1],y[:,:-1,1:],y[:,1:,1:],y[:,1:,:-1]
    determinant=lambda u,v:torch.linalg.det(torch.stack((u,v),-1))
    return torch.stack((determinant(b-a,d-a),determinant(b-a,c-a),determinant(b-d,c-d),determinant(c-a,d-a)),-1)*(y.shape[1]-1)**2

def fiber_checks():
    n=8;x,_,_,_,_=triangle_data(n);anchor=x.reshape(1,n+1,n+1,2)
    anchor=anchor.clone();anchor[...,1]+=.02*torch.sin(math.pi*anchor[...,0])*torch.sin(math.pi*anchor[...,1])
    # Independent bilinear hat with a single coarse-interior coefficient.
    hat=(1-(torch.arange(n+1,dtype=torch.float64)/n-.5).abs()*2).clamp_min(0)
    P=hat[:,None]*hat[None,:];direction=anchor.new_tensor([1.,0.])
    target=anchor+.6*P[None,...,None]*direction
    objective=lambda y:((y-target).square().sum(),{'data':(y-target).square().sum()})
    result=optimize_stiffness_fiber(anchor,objective,coefficient_side=3,maximum_gradients=8)
    initial=direct_corners(anchor);floor=.001+.05*(initial-.001)
    actual=direct_corners(result.vertices)
    assert bool((actual>floor).all())
    assert torch.equal(result.vertices[:,0],anchor[:,0])
    assert torch.equal(result.vertices[:,-1],anchor[:,-1])
    assert torch.equal(result.vertices[:,:,0],anchor[:,:,0])
    assert torch.equal(result.vertices[:,:,-1],anchor[:,:,-1])
    assert torch.allclose(result.vertices,anchor+result.coefficients.item()*P[None,...,None]*direction,atol=1e-15,rtol=1e-15)
    gradient=-1.2*P.square().sum();_,_,_,_,K=triangle_data(n)
    rhs=P[1:-1,1:-1].reshape(-1);H=3*(rhs@K@rhs)
    d=-gradient/H
    delta=direct_corners(anchor+d*P[None,...,None]*direction)-initial
    bound=float(((initial-floor)[delta<0]/(-delta[delta<0])).min())
    assert abs(bound-result.trace[0]['alpha_max'])<1e-13
    assert abs(result.trace[0]['initial_alpha']-min(1.,.99*bound))<1e-13
    assert all(t['accepted_total']<=t['total']+1e-4*t['accepted_alpha']*t['directional_derivative'] for t in result.trace if t['accepted'])
    # High curvature and a nearby optimum make the first stiffness step overshoot;
    # with no backtracking allowance it must preserve the original coefficients.
    near=anchor+.01*P[None,...,None]*direction
    steep=lambda y:(1e5*(y-near).square().sum(),{})
    rejected=optimize_stiffness_fiber(anchor,steep,coefficient_side=3,maximum_gradients=3,max_backtracks=0)
    assert rejected.stop_reason=='line_search_exhausted'
    assert rejected.counts['accepted_steps']==0 and torch.equal(rejected.vertices,anchor)
    return dict(active_bound=bound,reported_bound=result.trace[0]['alpha_max'],minimum_original_contracted_slack=float((actual-floor).min()),counts=result.counts,stop=result.stop_reason,rejection_stop=rejected.stop_reason,rejection_preserved_anchor=True)

def application_checks():
    import tempfile,copy
    sys.path.insert(0,str(root/'tests'))
    from test_coordinated_shared_affine_evidence import configuration
    from tools import coordinated_real_case as original_app
    from tools import coordinated_stiffness_application as new_app
    path=Path(tempfile.mkdtemp(prefix='stiffness_independent_',dir=root/'outputs/coordinated_instance_registration'))
    cfg=configuration(path,'analytic');cfg.mind_frame='shared_affine';cfg.image_objective='continuation'
    # Both implementations use the same tiny input and objective setup.
    cfg.grid_side=5;cfg.levels=[3,5];cfg.strain_weight=3.;cfg.precision='float64'
    old_class=original_app.Evidence;constructed=[];calls={}
    class Recorder(old_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);constructed.append(self)
        def __call__(self,vertices):
            calls[id(self)]=calls.get(id(self),0)+1
            return super().__call__(vertices)
    original_app.Evidence=new_app.Evidence=Recorder
    try:
        old=original_app.optimize(cfg);previous=list(constructed);constructed.clear();calls.clear()
        cfg=copy.deepcopy(cfg);cfg.output=path/'stiffness.npz'
        new=new_app.optimize_stiffness(cfg);current=list(constructed)
    finally:original_app.Evidence=new_app.Evidence=old_class
    observed=sum(calls.values())
    assert observed==new['objective_evaluations']
    errors=[]
    x,_,_,_,_=triangle_data(4);v=x.reshape(1,5,5,2).clone();v[:,1:-1,1:-1]+=.003
    for a,b in zip(previous,current,strict=True):
        assert torch.equal(a.fixed,b.fixed) and torch.equal(a.moving,b.moving)
        assert torch.equal(a.mask,b.mask) and torch.equal(a.matrix,b.matrix) and torch.equal(a.offset,b.offset)
        w=v.clone().requires_grad_();ea,pa=a(w);eb,pb=b(w)
        ga=torch.autograd.grad(ea,w)[0];gb=torch.autograd.grad(eb,w)[0]
        assert torch.equal(ea,eb) and torch.equal(ga,gb)
        assert all(torch.equal(pa[key],pb[key]) for key in pa)
        errors.append(dict(side=a.fixed.shape[-1],scalar_error=float(abs(ea-eb)),gradient_error=float((ga-gb).abs().max())))
    totals=[new['initial']['total']]+[s['accepted_full_total'] for s in new['stages']]
    assert new['final']['total']==min(totals)==new['best_accepted_full_total']
    assert new['saved_binary_certificate']['valid']
    return dict(per_scale_comparison=errors,reported_objective_evaluations=new['objective_evaluations'],observed_objective_evaluations=observed,gradient_steps=new['gradient_steps'],accepted_steps=new['accepted_steps'],backtracks=new['backtracks'],selected_full_total=new['final']['total'],certificate=True)

def postrun_checks(arms=('adam900','stiffness300')):
    import csv
    source=Path('D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')
    results=[]
    def read_points(section):
        with (source/str(section)/'landmarks'/f'{section:02}.csv').open(encoding='utf-8-sig') as stream:
            return {r['label']:np.array([float(r['x']),float(r['y'])]) for r in csv.DictReader(stream)}
    def unit(p,l):return ((p+.5)*np.array(l['effective_original_to_canvas_scale_xy'])+l['padding_xy'])/512
    def literal_p1(v,q):
        n=v.shape[0]-1;cell=np.minimum(np.floor(q*n).astype(int),n-1);t=q*n-cell
        answer=[]
        for (ix,iy),uv in zip(cell,t):
            # Solve barycentric coordinates on the explicitly chosen ac triangle.
            nodes=[(0,0),(1,0),(1,1)] if uv[1]<=uv[0] else [(0,0),(1,1),(0,1)]
            bary=np.linalg.solve(np.vstack((np.array(nodes).T,np.ones(3))),np.r_[uv,1.])
            answer.append(sum(w*v[iy+y,ix+x] for w,(x,y) in zip(bary,nodes)))
        return np.array(answer)
    for arm in arms:
        directory=root/'outputs/coordinated_instance_registration'/f'miit_{arm}_t20'
        manifest=json.loads((directory/'predictions.json').read_text())
        score=json.loads((directory/('landmark_scores_complete.json' if arm=='stiffness300' else 'landmark_scores.json')).read_text())
        assert manifest['prediction_complete'] and not manifest['annotations_read']
        rows=[]
        for row,scores in zip(manifest['rows'],score['rows'],strict=True):
            name=row['name'];report=json.loads((directory/row['methods']['analytic']['report']).read_text())
            cfg=report['configuration'];layout=json.loads((directory/row['layout']).resolve().read_text())
            assert name==scores['name'] and not report['landmarks_used']
            assert all(not any(word in cfg[k].lower() for word in ['landmark','annotation','.csv']) for k in ['fixed','moving','affine','matches'])
            with np.load(directory/row['methods']['analytic']['output']) as ar:
                v=ar['vertices'][0];ref=ar['boundary_reference'][0];A=ar['post_affine_matrix'].astype(float);b=ar['post_affine_offset'].astype(float)
                if arm=='coupled_seed':seed=ar['initializer_vertices'][0];coefficients=ar['initializer_coefficients_128px'][0]
            assert np.isfinite(v).all() and v.dtype==np.float64
            assert all(np.array_equal(a,c) for a,c in [(v[0],ref[0]),(v[-1],ref[-1]),(v[:,0],ref[:,0]),(v[:,-1],ref[:,-1])])
            assert np.linalg.det(A)>0
            q=direct_corners(torch.from_numpy(v)[None]).numpy();minimum=float(q.min())
            assert minimum>.001
            fp,mp=read_points(row['fixed_section']),read_points(row['moving_section'])
            assert len(fp)==len(mp)==124 and set(fp)==set(mp)
            ids=sorted(k for k in fp if np.isfinite(fp[k]).all() and np.isfinite(mp[k]).all())
            assert ids==scores['available_pair_labels']
            fixed=np.array([fp[k] for k in ids]);target=np.array([mp[k] for k in ids]);query=unit(fixed,layout['fixed'])
            mapped=literal_p1(v@A.T+b,query)
            canvas_errors=np.linalg.norm((mapped-unit(target,layout['moving']))*512,axis=1)
            native=(mapped*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5
            native_errors=np.linalg.norm(native-target,axis=1)
            discrepancies=[]
            for errors,key in [(canvas_errors,'canvas_pixels'),(native_errors,'native_moving_pixels')]:
                recorded=scores['methods']['analytic']['metrics'][key]
                discrepancy=max(abs(errors[i]-recorded['per_label'][label]) for i,label in enumerate(ids))
                assert discrepancy<1e-10
                assert abs(errors.mean()-recorded['mean'])<1e-10
                assert abs(np.percentile(errors,90)-recorded['p90'])<1e-10
                discrepancies.append(discrepancy)
            expected=900 if arm=='adam900' else 300
            assert report['gradient_steps']==sum(s['inner_steps'] for s in report['stages'])==expected
            assert report['failed_trials']==0 and len(report['stages'])==10
            selector=[report['initial']['total']]+[s['accepted_full_total'] for s in report['stages']]
            if arm=='coupled_seed':selector.append(report['seed_record']['original_full_objective']['total'])
            assert report['final']['total']==min(selector)
            extra={}
            if arm=='stiffness300':
                trace=report['trace'];assert len(trace)==300 and report['accepted_steps']==300
                assert all(t['initial_alpha']==1. and t['accepted'] for t in trace)
                backtracks=sum(len(t['trials'])-1 for t in trace)
                assert backtracks==report['backtracks']
                assert sum(len(t['trials']) for t in trace)==report['trial_evaluations']
                assert report['objective_evaluations']==300+report['trial_evaluations']+12
                for t in trace:
                    for i,trial in enumerate(t['trials']):
                        assert trial['geometry_valid'] and trial['alpha']==.5**i
                        assert trial['accepted']==(trial['total']<=trial['armijo_rhs'])==(i==len(t['trials'])-1)
                finest=[t for t in trace if t['level']==257]
                ratios=[2*(t['accepted_total']-t['total']-t['accepted_alpha']*t['directional_derivative'])/(t['accepted_alpha']**2*(-t['directional_derivative'])) for t in finest]
                extra=dict(initial_alpha_all_one=True,backtracks=backtracks,complete_objective_calls=report['objective_evaluations'],finest_median_directional_secant_ratio=float(np.median(ratios)),finest_median_alpha=float(np.median([t['accepted_alpha'] for t in finest])))
            elif arm=='adam900':
                assert len(report['trace'])==910 and all(s['inner_steps']==90 for s in report['stages'])
            elif arm=='coupled_seed':
                record=report['seed_record']
                assert len(report['trace'])==310 and report['objective_evaluations']==333
                assert report['seed_objective_evaluations']==1 and report['selected_stage']==9
                assert not record['original_E_seed_rejection']
                assert record['coordinate_steps']==16 and all(s['scale']==1. for s in record['steps'])
                assert record['raw_target_nonpositive_corner_count']==0
                assert coefficients.shape==(65,65,2) and seed.shape==v.shape
                assert not np.any(coefficients[0]) and not np.any(coefficients[-1]) and not np.any(coefficients[:,0]) and not np.any(coefficients[:,-1])
                axis=np.arange(257)/256;coarse=np.arange(65)/64
                B=np.maximum(1-np.abs(axis[:,None]-coarse[None,:])*64,0)
                raw=np.stack([B@coefficients[...,k]@B.T for k in range(2)],axis=-1)/128
                reconstruction=float(np.max(np.abs(seed-ref-raw)))
                assert reconstruction<3e-16
                assert all(np.array_equal(a,c) for a,c in [(seed[0],ref[0]),(seed[-1],ref[-1]),(seed[:,0],ref[:,0]),(seed[:,-1],ref[:,-1])])
                seed_minimum=float(direct_corners(torch.from_numpy(seed)[None]).min());assert seed_minimum>.001
                raised=record['original_full_objective']['total']>report['initial']['total']
                assert raised==record['original_E_increased']
                from tools.coordinated_real_case import Evidence,load_registration_evidence,load_image_matches
                fp_path=(directory/row['fixed']).resolve();mp_path=(directory/row['moving']).resolve()
                fixed_image,moving_image,mask,_=load_registration_evidence(fp_path,mp_path,512)
                matches,_=load_image_matches((directory/row['raw_matches']['path']).resolve(),A.astype(np.float32),b.astype(np.float32),fixed_path=fp_path,moving_path=mp_path,image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
                totals={}
                for side in [32,512]:
                    reduce=lambda image:torch.nn.functional.interpolate(image,size=(side,side),mode='area')
                    ev=Evidence(reduce(fixed_image),reduce(moving_image),torch.tensor(A),torch.tensor(b),'mind',3.,1.,1e-4,fixed_mask=reduce(mask),interpolation='p1_ac',matches=matches,match_weight=.1,strain_model='p1_arap',mind_frame='shared_affine')
                    ev.prepare_fixed_p1_sampling(257,257,dtype=torch.float64,device='cpu')
                    with torch.no_grad():totals[side]=float(ev(torch.from_numpy(seed)[None])[0])
                assert abs(totals[32]-report['stages'][0]['anchor_total'])<1e-7
                assert abs(totals[32]-report['trace'][0]['total'])<1e-7
                assert abs(totals[512]-record['original_full_objective']['total'])<1e-7
                extra=dict(seed_higher_E_than_identity=raised,seed_reconstruction_error=reconstruction,seed_minimum_corner=seed_minimum,all_16_construction_scales_one=True,complete_objective_calls=333,selected_stage=9,recomputed_seed_full_E_error=abs(totals[512]-record['original_full_objective']['total']),recomputed_first_stage_seed_anchor_E_error=abs(totals[32]-report['stages'][0]['anchor_total']))
            rows.append(dict(name=name,landmarks=len(ids),mean_canvas=float(canvas_errors.mean()),p90_canvas=float(np.percentile(canvas_errors,90)),minimum_corner=minimum,boundary_exact=True,maximum_canvas_score_difference=discrepancies[0],maximum_native_score_difference=discrepancies[1],gradient_steps=report['gradient_steps'],**extra))
        assert sum(r['landmarks'] for r in rows)==328
        mean=float(np.mean([r['mean_canvas'] for r in rows]))
        assert abs(mean-score['aggregate']['analytic']['all_three']['canvas_pixels']['mean_pair_mean'])<1e-12
        results.append(dict(arm=arm,mean_pair_mean=mean,mean_pair_p90=float(np.mean([r['p90_canvas'] for r in rows])),rows=rows))
    return results

if __name__=='__main__':
    print(json.dumps(postrun_checks() if '--postrun' in sys.argv else dict(frozen_ARAP=frozen_checks(),galerkin=galerkin_checks(),physical_fiber=fiber_checks(),application=application_checks())))
