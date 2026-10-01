"""Independent literal descriptor oracle, not a clinical correspondence test."""
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_real_case import Evidence
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences


OFFSETS=((2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2))


def literal_descriptor(image,return_distances=False):
    """Explicit replicate neighbors and valid clipped 3x3 patch averages."""
    height,width=image.shape[-2:]
    channels=[]
    for dx,dy in OFFSETS:
        shifted=torch.stack([torch.stack([image[...,min(max(r+dy,0),height-1),
            min(max(c+dx,0),width-1)] for c in range(width)],-1) for r in range(height)],-2)
        squared=(image-shifted).square()
        channel=torch.stack([torch.stack([squared[...,max(r-1,0):min(r+2,height),
            max(c-1,0):min(c+2,width)].mean((-1,-2)) for c in range(width)],-1)
            for r in range(height)],-2)
        channels.append(channel)
    distance=torch.cat(channels,1)
    minimum=distance.amin(1,keepdim=True)
    scale=distance.mean(1,keepdim=True)
    feature=torch.exp(-(distance-minimum)/(scale+1e-4))
    return (feature,distance) if return_distances else feature


def grid(side,dtype=torch.float64,batch=1):
    yy,xx=torch.meshgrid(torch.arange(side,dtype=dtype)/(side-1),
        torch.arange(side,dtype=dtype)/(side-1),indexing="ij")
    return torch.stack((xx,yy),-1)[None].expand(batch,-1,-1,-1).clone()


def literal_queries(vertices,height,width,diagonal):
    """Generic source-triangle edge solve for each fixed pixel center."""
    batch,rows,columns,_=vertices.shape
    source=grid(rows,vertices.dtype)[0]
    result=[]
    for r in range(height):
        line=[]
        for c in range(width):
            q=vertices.new_tensor([(c+.5)/width,(r+.5)/height])
            column=min(int(float(q[0])*(columns-1)),columns-2)
            row=min(int(float(q[1])*(rows-1)),rows-2)
            u,v=float(q[0])*(columns-1)-column,float(q[1])*(rows-1)-row
            corners=((row,column),(row,column+1),(row+1,column+1),(row+1,column))
            triangle=((0,1,2) if v<=u else (0,2,3)) if diagonal=="ac" else ((0,1,3) if u+v<=1 else (1,2,3))
            ids=[corners[i] for i in triangle]
            edges=torch.stack([source[ids[i]]-source[ids[0]] for i in (1,2)],1)
            bary=torch.linalg.solve(edges,q-source[ids[0]])
            mapped=vertices[:,*ids[0]]+bary[0]*(vertices[:,*ids[1]]-vertices[:,*ids[0]])+bary[1]*(vertices[:,*ids[2]]-vertices[:,*ids[0]])
            line.append(mapped)
        result.append(torch.stack(line,1))
    return torch.stack(result,1)


def literal_image_loss(fixed,moving,vertices,matrix,offset,mask,diagonal,order):
    query=literal_queries(vertices,*fixed.shape[-2:],diagonal)@matrix.T+offset
    source=moving if order=="after_warp" else literal_descriptor(moving)
    warped=F.grid_sample(source,2*query-1,mode="bilinear",padding_mode="zeros",align_corners=False)
    feature=literal_descriptor(warped) if order=="after_warp" else warped
    return ((literal_descriptor(fixed)-feature).abs().mean(1,keepdim=True)*mask).sum()/mask.sum()


def fixture(batch=2,height=5,width=7):
    generator=torch.Generator().manual_seed(114)
    fixed=.15+.7*torch.rand(batch,1,height,width,generator=generator,dtype=torch.float64)
    moving=.15+.7*torch.rand(batch,1,height,width,generator=generator,dtype=torch.float64)
    vertices=.81*grid(3,batch=batch)+torch.tensor([.071,.093],dtype=torch.float64)
    vertices+=.003*torch.randn(vertices.shape,generator=generator,dtype=torch.float64)
    mask=.2+.8*torch.rand(fixed.shape,generator=generator,dtype=torch.float64)
    return fixed,moving,vertices,mask


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("cached",[False,True])
def test_literal_after_warp_value_full_gradient_and_fd(diagonal,cached):
    fixed,moving,vertices,mask=fixture()
    vertices.requires_grad_();matrix=torch.eye(2,dtype=vertices.dtype);offset=vertices.new_zeros(2)
    evidence=Evidence(fixed,moving,matrix,offset,"mind",0.,0.,fixed_mask=mask,
        interpolation="p1_"+diagonal,mind_order="after_warp")
    if cached:evidence.prepare_fixed_p1_sampling(3,3,dtype=vertices.dtype,device=vertices.device)
    query=literal_queries(vertices,*fixed.shape[-2:],diagonal)
    moving_pixel=query*query.new_tensor([moving.shape[-1],moving.shape[-2]])-.5
    assert (moving_pixel-moving_pixel.round()).abs().min()>1e-3
    # Stable minima and L1 signs under a deterministic small full-vertex probe.
    # Exact zero L1 channels at shared descriptor minima remain identically zero.
    feature_fixed=literal_descriptor(fixed)
    def branch_signature(value):
        q=literal_queries(value,*fixed.shape[-2:],diagonal)
        raw=F.grid_sample(moving,2*q-1,mode="bilinear",padding_mode="zeros",align_corners=False)
        feature,distance=literal_descriptor(raw,return_distances=True)
        return distance.argmin(1),(feature_fixed-feature).sign()
    tangent=torch.linspace(-.7,.9,vertices.numel(),dtype=vertices.dtype).reshape_as(vertices)
    signatures=[branch_signature(vertices.detach()+step*tangent) for step in (-2e-7,0.,2e-7)]
    for signature in signatures:
        assert torch.equal(signature[0],signatures[1][0])
        assert torch.equal(signature[1],signatures[1][1])
    actual,parts=evidence(vertices)
    oracle=literal_image_loss(fixed,moving,vertices,matrix,offset,mask,diagonal,"after_warp")
    torch.testing.assert_close(actual,oracle,rtol=2e-13,atol=2e-14)
    actual_gradient,=torch.autograd.grad(actual,vertices)
    oracle_gradient,=torch.autograd.grad(oracle,vertices)
    torch.testing.assert_close(actual_gradient,oracle_gradient,rtol=2e-10,atol=2e-11)
    assert torch.isfinite(actual_gradient).all()
    # Every vertex component, independent literal scalar FD; no held-out truth.
    h=2e-7;fd=torch.empty_like(vertices)
    for index in range(vertices.numel()):
        plus=vertices.detach().clone();minus=plus.clone()
        plus.reshape(-1)[index]+=h;minus.reshape(-1)[index]-=h
        fd.reshape(-1)[index]=(literal_image_loss(fixed,moving,plus,matrix,offset,mask,diagonal,"after_warp")-
            literal_image_loss(fixed,moving,minus,matrix,offset,mask,diagonal,"after_warp"))/(2*h)
    torch.testing.assert_close(actual_gradient,fd,rtol=3e-5,atol=2e-7)
    torch.testing.assert_close(parts["image"],actual,rtol=0,atol=0)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_default_transport_bit_path_and_noncommutation(diagonal):
    fixed,moving,vertices,mask=fixture(batch=1)
    vertices.requires_grad_()
    matrix=torch.eye(2,dtype=vertices.dtype);offset=vertices.new_zeros(2)
    kwargs=dict(fixed_mask=mask,interpolation="p1_"+diagonal)
    default=Evidence(fixed,moving,matrix,offset,"mind",.2,.7,**kwargs)
    explicit=Evidence(fixed,moving,matrix,offset,"mind",.2,.7,mind_order="transport",**kwargs)
    after=Evidence(fixed,moving,matrix,offset,"mind",.2,.7,mind_order="after_warp",**kwargs)
    old,old_parts=default(vertices);same,same_parts=explicit(vertices);new,new_parts=after(vertices)
    assert torch.equal(old,same)
    assert all(torch.equal(old_parts[k],same_parts[k]) for k in old_parts)
    default_gradient,=torch.autograd.grad(old,vertices)
    explicit_gradient,=torch.autograd.grad(same,vertices)
    assert torch.equal(default_gradient,explicit_gradient)
    reference=literal_image_loss(fixed,moving,vertices,matrix,offset,mask,diagonal,"transport")
    torch.testing.assert_close(old_parts["image"],reference,rtol=2e-13,atol=2e-14)
    assert abs(float(new_parts["image"]-old_parts["image"]))>1e-3


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_same_raster_identity_after_warp_and_immutable_raw(diagonal):
    fixed,_,_,mask=fixture(batch=2)
    identity=grid(3,batch=2);before=fixed.clone()
    evidence=Evidence(fixed,fixed,torch.eye(2,dtype=fixed.dtype),fixed.new_zeros(2),"mind",0.,0.,
        fixed_mask=mask,interpolation="p1_"+diagonal,mind_order="after_warp")
    value,_=evidence(identity)
    assert abs(float(value))<2e-14
    assert torch.equal(fixed,before)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_only_image_changes_mask_oob_match_shape_strain_identical(diagonal):
    fixed,moving,vertices,mask=fixture(batch=1)
    matrix=torch.tensor([[1.13,.03],[.02,1.1]],dtype=vertices.dtype)
    offset=torch.tensor([.15,-.17],dtype=vertices.dtype)
    points=torch.tensor([[.13,.21],[.32,.47],[.71,.82]],dtype=vertices.dtype)
    matches=ImageCorrespondences(points,points+.01,torch.tensor([.3,.5,.8],dtype=vertices.dtype),pixel_scale=7,robust_scale=2.)
    kwargs=dict(shape_weight=.0001,fixed_mask=mask,interpolation="p1_"+diagonal,matches=matches,match_weight=.1)
    before=moving.clone()
    transport=Evidence(fixed,moving,matrix,offset,"mind",3.,1.,mind_order="transport",**kwargs)
    after=Evidence(fixed,moving,matrix,offset,"mind",3.,1.,mind_order="after_warp",**kwargs)
    old,op=transport(vertices);new,np=after(vertices)
    assert torch.equal(transport.denominator,mask.sum()) and torch.equal(after.denominator,mask.sum())
    for term in ("strain","shape","match","oob","outside_fraction"):
        torch.testing.assert_close(op[term],np[term],rtol=0,atol=0)
    assert op["oob"]>0 and op["outside_fraction"]>0
    torch.testing.assert_close(new-old,np["image"]-op["image"],rtol=2e-13,atol=2e-14)
    assert torch.equal(moving,before)


@pytest.mark.parametrize("loss,order",[("local_ncc","after_warp"),("mind","other")])
def test_unsupported_order_guards(loss,order):
    fixed,moving,_,_=fixture(batch=1)
    with pytest.raises(ValueError):
        Evidence(fixed,moving,torch.eye(2,dtype=fixed.dtype),fixed.new_zeros(2),loss,0.,0.,mind_order=order)
