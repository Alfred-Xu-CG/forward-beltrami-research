import sys,json,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src'),str(root/'tests')]
from tools.coordinated_real_case import Evidence,optimize
from tools.coordinated_multiscale_evidence import SimultaneousImageEvidence
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from tools.coordinated_real_case import corner_symmetric_dirichlet
from test_coordinated_shared_affine_evidence import configuration
torch.set_num_threads(1)
gen=torch.Generator().manual_seed(8002)
fixed=torch.rand(1,1,32,32,generator=gen)*.8+.1
moving=torch.rand(1,1,32,32,generator=gen)*.8+.1
mask=torch.rand(1,1,32,32,generator=gen);mask[:,:,:9]=0.;mask[:,:,:,4:7]=.125
A=torch.tensor([[1.12,.08],[-.03,1.02]],dtype=torch.float64);b=torch.tensor([-.07,.02],dtype=torch.float64)
ps=torch.tensor([[.19,.37],[.48,.61],[.83,.75]],dtype=torch.float64)
matches=ImageCorrespondences(ps,ps+torch.tensor([.02,-.015]),torch.tensor([.5,.8,.9],dtype=torch.float64))
objects={}
for s in [8,12,16,24,32]:
    r=lambda x:F.interpolate(x,(s,s),mode='area')
    objects[s]=Evidence(r(fixed),r(moving),A,b,'mind',3.,1.,1e-4,fixed_mask=r(mask),interpolation='p1_ac',matches=matches,match_weight=.1,strain_model='p1_arap',mind_frame='shared_affine')
    objects[s].prepare_fixed_p1_sampling(7,9,dtype=torch.float64,device='cpu')
yy,xx=torch.meshgrid(torch.linspace(0,1,7,dtype=torch.float64),torch.linspace(0,1,9,dtype=torch.float64),indexing='ij')
v=torch.stack((xx,yy),-1)[None];v[:,1:-1,1:-1]+=torch.randn(1,5,7,2,generator=gen,dtype=torch.float64)*.009;v.requires_grad_()
def literal_p1(vertices,s):
    yy,xx=torch.meshgrid((torch.arange(s,dtype=torch.float64)+.5)/s,(torch.arange(s,dtype=torch.float64)+.5)/s,indexing='ij')
    rx,ry=xx*8,yy*6;x=rx.floor().long().clamp(max=7);y=ry.floor().long().clamp(max=5);a,bx,c,d=vertices[:,y,x],vertices[:,y,x+1],vertices[:,y+1,x+1],vertices[:,y+1,x]
    tx,ty=(rx-x)[None,...,None],(ry-y)[None,...,None]
    return torch.where(ty<=tx,(1-tx)*a+(tx-ty)*bx+ty*c,(1-ty)*a+tx*c+(ty-tx)*d)
images=[];terms=[]
for s,obj in objects.items():
    q=literal_p1(v,s);warped=F.grid_sample(obj.moving_feature,(2*q-1).float(),align_corners=False,mode='bilinear',padding_mode='zeros')
    image=((warped-obj.fixed_feature).abs().mean(1,keepdim=True)*obj.mask).sum()/obj.mask.sum()
    images.append(image)
    if s==32:
        world=q@A.T+b
        oob=(((F.relu(-world)+F.relu(world-1)).square().sum(-1)[:,None])*obj.mask).sum()/obj.mask.sum()
prior=3*p1_arap_energy(v,diagonal='ac',validate=False)+1e-4*corner_symmetric_dirichlet(v)+.1*matches(v,A,'p1_ac')+oob
expected=sum(.2*im for im in images)+prior
actual,parts=SimultaneousImageEvidence(objects)(v)
g=torch.autograd.grad(actual,v,retain_graph=True)[0]
ge=torch.autograd.grad(expected,v,retain_graph=True)[0]
separate=sum(torch.autograd.grad(.2*im,v,retain_graph=True)[0] for im in images)+torch.autograd.grad(prior,v)[0]
print(json.dumps(dict(actual=float(actual),literal=float(expected),value_difference=float(abs(actual-expected)),max_vertex_gradient_difference=float((g-ge).abs().max()),max_separately_summed_gradient_difference=float((g-separate).abs().max()),oob=float(oob),prior=float(prior),parts={k:float(x) for k,x in parts.items()},image_dtype=str(images[0].dtype),geometry_dtype=str(v.dtype))))
assert torch.allclose(actual,expected,rtol=0,atol=2e-9)
assert torch.allclose(g,ge,rtol=0,atol=5e-7)
assert torch.allclose(g,separate,rtol=0,atol=5e-7)

# Count actual raster evaluations in the complete tiny application path.
results={}
for mode in ['continuation','simultaneous_multiscale']:
    path=Path(tempfile.mkdtemp(prefix='ms_independent_',dir=root/'outputs/coordinated_instance_registration'))
    cfg=configuration(path,'analytic');cfg.mind_frame='shared_affine';cfg.image_objective=mode
    counts={};original=Evidence.image_terms
    def counted(self,vertices):
        side=self.fixed.shape[-1];counts[side]=counts.get(side,0)+1
        return original(self,vertices)
    Evidence.image_terms=counted
    try:report=optimize(cfg)
    finally:Evidence.image_terms=original
    results[mode]=dict(counts=counts,J=report['objective_evaluations'],gradient_steps=report['gradient_steps'],stage_pairs=[[s['accepted_total'],s['accepted_full_total']] for s in report['stages']],final=report['final']['total'],best=report['best_accepted_full_total'])
print(json.dumps(results))
assert results['continuation']['counts']=={16:12,8:6}
assert results['simultaneous_multiscale']['counts']=={16:18,8:18}
assert all(a==b for a,b in results['simultaneous_multiscale']['stage_pairs'])
