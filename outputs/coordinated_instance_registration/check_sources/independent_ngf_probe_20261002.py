"""Independent NGF check: literal matrix stencil, explicit transpose and image-map chain."""
import sys,json
from pathlib import Path
import torch
import torch.nn.functional as F
root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src')]
torch.set_num_threads(1)

def matrix(n,dtype=torch.float64):
    d=torch.zeros(n,n,dtype=dtype)
    d[0,0],d[0,1]=-n,n;d[-1,-2],d[-1,-1]=-n,n
    for i in range(1,n-1):d[i,i-1],d[i,i+1]=-n/2,n/2
    return d

def G(x):
    dx,dy=matrix(x.shape[-1],x.dtype),matrix(x.shape[-2],x.dtype)
    return torch.cat((x@dx.T,dy@x),1)

def GT(z):
    dx,dy=matrix(z.shape[-1],z.dtype),matrix(z.shape[-2],z.dtype)
    return z[:,:1]@dx+dy.T@z[:,1:]

def identity(rows,columns):
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),torch.linspace(0,1,columns,dtype=torch.float64),indexing='ij')
    return torch.stack((x,y),-1)[None]

def P1(v,s):
    h,w=v.shape[1:3]
    y,x=torch.meshgrid((torch.arange(s,dtype=torch.float64)+.5)/s,(torch.arange(s,dtype=torch.float64)+.5)/s,indexing='ij')
    rx,ry=x*(w-1),y*(h-1);ix,iy=rx.floor().long().clamp(max=w-2),ry.floor().long().clamp(max=h-2)
    a,b,c,d=v[:,iy,ix],v[:,iy,ix+1],v[:,iy+1,ix+1],v[:,iy+1,ix]
    u,t=(rx-ix)[None,...,None],(ry-iy)[None,...,None]
    return torch.where(t<=u,(1-u)*a+(u-t)*b+t*c,(1-t)*a+u*c+(t-u)*d)

def main():
    from tools.coordinated_ngf import unit_canvas_gradient,FrozenPostwarpNGF
    from tools.coordinated_real_case import Evidence
    seed=torch.Generator().manual_seed(22961)
    # Rectangular/endpoints, including the two-point case where both stencils overlap.
    adjoint=[]
    for h,w in [(2,2),(2,7),(5,2),(7,11)]:
        x=torch.randn(1,1,h,w,generator=seed,dtype=torch.float64,requires_grad=True)
        z=torch.randn(1,2,h,w,generator=seed,dtype=torch.float64)
        actual=unit_canvas_gradient(x);expected=G(x)
        dx=torch.autograd.grad((actual*z).sum(),x)[0]
        torch.testing.assert_close(actual,expected,atol=2e-14,rtol=2e-15)
        torch.testing.assert_close(dx,GT(z),atol=2e-14,rtol=2e-15)
        adjoint.append(float(((G(x)*z).sum()-(x*GT(z)).sum()).abs()))
    s=32
    fixed=torch.rand(1,1,s,s,generator=seed,dtype=torch.float64)
    moving=torch.rand(1,1,s,s,generator=seed,dtype=torch.float64)
    mask=torch.rand(1,1,s,s,generator=seed,dtype=torch.float64);mask[...,:2,:]=0;mask[...,:,-3:]=.125
    A=torch.tensor([[1.04,.035],[-.02,.97]],dtype=torch.float64);b=torch.tensor([-.011,.025],dtype=torch.float64)
    obj=Evidence(fixed,moving,A,b,'ngf',3.,1.,1e-4,fixed_mask=mask,interpolation='p1_ac',strain_model='p1_arap')
    obj.prepare_fixed_p1_sampling(9,11,dtype=torch.float64,device='cpu')
    v=identity(9,11);v[:,1:-1,1:-1]+=torch.randn(1,7,9,2,generator=seed,dtype=torch.float64)*.003;v.requires_grad_()
    q=P1(v,s);world=q@A.T+b
    warped=F.grid_sample(moving,2*world-1,align_corners=False,padding_mode='zeros',mode='bilinear')
    ai,bi=G(fixed),G(warped);ef,em=obj.ngf.epsilon_fixed,obj.ngf.epsilon_moving
    AA,BB=ai.square().sum(1,keepdim=True)+ef.square(),bi.square().sum(1,keepdim=True)+em.square()
    c=(ai*bi).sum(1,keepdim=True);weighted=mask/mask.sum()
    literal=(weighted*(1-c.square()/(AA*BB))).sum()
    explicit_b=-2*weighted*(c*ai/(AA*BB)-c.square()*bi/(AA*BB.square()))
    explicit_w=GT(explicit_b)
    explicit_vertices=torch.autograd.grad(warped,v,explicit_w,retain_graph=True)[0]
    actual=obj.image_terms(v)[0]
    actual_vertices=torch.autograd.grad(actual,v,retain_graph=True)[0]
    direction=torch.randn(v.shape,generator=seed,dtype=torch.float64)*.04
    delta=1e-7
    numerical=(obj.image_terms(v+delta*direction)[0]-obj.image_terms(v-delta*direction)[0])/(2*delta)
    analytic=(actual_vertices*direction).sum()
    torch.testing.assert_close(actual,literal,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(actual_vertices,explicit_vertices,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(analytic,numerical,atol=2e-8,rtol=2e-6)
    # Independently calibrate from initial COMPLETE affine on raw moving intensity.
    initial_q=P1(identity(9,11),s)@A.T+b
    initial=F.grid_sample(moving,2*initial_q-1,align_corners=False,padding_mode='zeros')
    epsilon=lambda z:max(s/255.,float((weighted*G(z).square().sum(1,keepdim=True).sqrt()).sum())*.1)
    assert abs(float(ef)-epsilon(fixed))<1e-14 and abs(float(em)-epsilon(initial))<1e-14
    frozen=(float(ef),float(em))
    obj.image_terms(v+direction*.05)
    assert frozen==(float(obj.ngf.epsilon_fixed),float(obj.ngf.epsilon_moving))
    assert not ef.requires_grad and not em.requires_grad
    # Fixed eps sign/constant symmetry, no false stationary identity requirement.
    normal=obj.ngf.errors(warped);reversed_contrast=obj.ngf.errors(1-warped)
    torch.testing.assert_close(normal,reversed_contrast,atol=2e-14,rtol=2e-14)
    flat=torch.full_like(warped,.37,requires_grad=True)
    flat_error=obj.ngf.errors(flat);flat_grad=torch.autograd.grad(flat_error.sum(),flat)[0]
    assert torch.equal(flat_error,torch.ones_like(flat_error)) and torch.count_nonzero(flat_grad)==0
    # Nonidentity spatial map: postwarp G differs from transporting moving G.
    yy,xx=torch.meshgrid((torch.arange(s,dtype=torch.float64)+.5)/s,(torch.arange(s,dtype=torch.float64)+.5)/s,indexing='ij')
    ramp=(.2+.2*xx+.1*yy)[None,None]
    deform=identity(9,11);deform[...,0]+=.06*torch.sin(torch.pi*deform[...,0])*torch.sin(torch.pi*deform[...,1])
    dq=P1(deform,s)
    intensity=F.grid_sample(ramp,2*dq-1,align_corners=False,padding_mode='zeros')
    transported=F.grid_sample(G(ramp),2*dq-1,align_corners=False,padding_mode='zeros')
    post=G(intensity);difference=float((post-transported)[...,3:-3,3:-3].abs().max())
    assert difference>.025
    print(json.dumps(dict(stencil_adjoint_max_error=max(adjoint),epsilon_fixed=float(ef),epsilon_moving=float(em),value_difference=float(abs(actual-literal)),explicit_vertex_vjp_max_difference=float((actual_vertices-explicit_vertices).abs().max()),directional_gradient=float(analytic),directional_finite_difference=float(numerical),transport_counterexample_max_gradient_difference=difference,eps_frozen_after_new_candidate=True,flat_loss_one_gradient_zero=True)))

if __name__=='__main__':main()
