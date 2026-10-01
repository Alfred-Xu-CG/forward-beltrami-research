"""Literal quarter sum, independent triangle mapping and zero-padded bilinear."""
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_real_case import Evidence
from tools.coordinated_pixel_quadrature import FourQuarterMindEvidence,pixel_quarter_queries,_normalize_evidence_device
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences


def grid(rows,columns=None,batch=1,dtype=torch.float64):
    columns=rows if columns is None else columns
    yy,xx=torch.meshgrid(torch.arange(rows,dtype=dtype)/(rows-1),
                        torch.arange(columns,dtype=dtype)/(columns-1),indexing="ij")
    return torch.stack((xx,yy),-1)[None].repeat(batch,1,1,1)


def fixture(diagonal="ac",batch=1,image_dtype=torch.float64):
    generator=torch.Generator().manual_seed(531)
    fixed=torch.rand(batch,1,5,7,generator=generator,dtype=image_dtype)*.7+.1
    moving=torch.rand(batch,1,5,7,generator=generator,dtype=image_dtype)*.7+.1
    mask=torch.linspace(0,1,35,dtype=image_dtype).reshape(1,1,5,7).repeat(batch,1,1,1)
    matrix=torch.tensor([[.87,.06],[-.04,.91]],dtype=torch.float64)
    offset=torch.tensor([.047,.029],dtype=torch.float64)
    base=Evidence(fixed,moving,matrix,offset,"mind",3.,1.,.0001,mask,
                  interpolation="p1_"+diagonal,strain_model="p1_arap")
    vertices=grid(3,4,batch)
    vertices[:,1,1:3]+=.013*torch.randn(batch,2,2,generator=generator,dtype=vertices.dtype)
    return base,vertices


def literal_p1(vertices,query,diagonal):
    rows,columns=vertices.shape[1:3]
    source=grid(rows,columns)[0]
    xr,yr=float(query[0])*(columns-1),float(query[1])*(rows-1)
    col,row=min(int(xr),columns-2),min(int(yr),rows-2)
    u,v=xr-col,yr-row
    corners=((row,col),(row,col+1),(row+1,col+1),(row+1,col))
    face=((0,1,2) if v<=u else (0,2,3)) if diagonal=="ac" else ((0,1,3) if u+v<=1 else (1,2,3))
    p,q,r=[corners[i] for i in face]
    edges=torch.stack((source[q]-source[p],source[r]-source[p]),1)
    weights=torch.linalg.solve(edges,query-source[p])
    return vertices[:,*p]+weights[0]*(vertices[:,*q]-vertices[:,*p])+weights[1]*(vertices[:,*r]-vertices[:,*p])


def literal_bilinear_zero(field,queries):
    """Explicit four neighbor contributions, original alignfalse pixel centers."""
    batch,channels,height,width=field.shape
    result=[]
    for n in range(batch):
        x,y=queries[n,0]*width-.5,queries[n,1]*height-.5
        ix,iy=int(torch.floor(x.detach())),int(torch.floor(y.detach()))
        value=field.new_zeros(channels)
        for row in (iy,iy+1):
            for col in (ix,ix+1):
                wx=(1-(x-col).abs());wy=(1-(y-row).abs())
                if 0<=row<height and 0<=col<width:value=value+wx*wy*field[n,:,row,col]
        result.append(value)
    return torch.stack(result)


def literal_four_sum(base,vertices,diagonal):
    height,width=base.fixed.shape[-2:]
    numerator=vertices.new_zeros(())
    for row in range(height):
        for col in range(width):
            for dy,dx in ((.25,.25),(.25,.75),(.75,.25),(.75,.75)):
                source=vertices.new_tensor([(col+dx)/width,(row+dy)/height])
                fixed=literal_bilinear_zero(base.fixed_feature,source[None].expand(vertices.shape[0],-1))
                mapped=literal_p1(vertices,source,diagonal)@base.matrix.T+base.offset
                moving=literal_bilinear_zero(base.moving_feature,mapped)
                numerator=numerator+((fixed-moving).abs().mean(1)*base.mask[:,0,row,col]).sum()
    return numerator/(4*base.mask.sum())


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_b2_nonsquare_literal_four_sum_full_vertex_ad_and_fd(diagonal):
    base,vertices=fixture(diagonal,batch=2)
    vertices.requires_grad_()
    evidence=FourQuarterMindEvidence(base,3,4)
    _,parts=evidence(vertices)
    literal=literal_four_sum(base,vertices,diagonal)
    torch.testing.assert_close(parts["image"],literal,rtol=2e-13,atol=2e-14)
    actual,=torch.autograd.grad(parts["image"],vertices,retain_graph=True)
    expected,=torch.autograd.grad(literal,vertices)
    torch.testing.assert_close(actual,expected,rtol=3e-12,atol=3e-13)
    # EVERY vertex component, no control-boundary exclusion from the gradient test.
    h=2e-7
    height,width=base.fixed.shape[-2:]
    # Fixed queries have fixed source branches; the moving raster and L1
    # branches must remain separated under every component's +/-h probe.
    maximum_l1_change=h*float((base.matrix.abs()*base.matrix.new_tensor([width,height])[:,None]).sum(0).max())
    for evaluator,fixed in zip(evidence.quarter_evaluators,evidence.fixed_quarter_features):
        q=evaluator(vertices)@base.matrix.T+base.offset
        pixels=q*q.new_tensor([width,height])-.5
        assert float((pixels-pixels.round()).abs().min())>5*h*max(height,width)
        warped=F.grid_sample(base.moving_feature,2*q-1,mode="bilinear",padding_mode="zeros",align_corners=False)
        assert float((fixed-warped).abs().min())>5*maximum_l1_change
    for index in range(vertices.numel()):
        tangent=torch.zeros_like(vertices);tangent.flatten()[index]=1
        plus=evidence(vertices.detach()+h*tangent)[1]["image"]
        minus=evidence(vertices.detach()-h*tangent)[1]["image"]
        torch.testing.assert_close((plus-minus)/(2*h),actual.flatten()[index],rtol=3e-6,atol=2e-9)


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("frozen",[False,True])
def test_original_center_oob_strain_shape_match_values_and_gradients_bit_unchanged(diagonal,frozen):
    base,vertices=fixture(diagonal)
    base.offset=torch.tensor([-.19,.11],dtype=vertices.dtype)
    base.matches=ImageCorrespondences(torch.tensor([[.2,.3],[.7,.8]],dtype=vertices.dtype),
        torch.tensor([[.23,.31],[.66,.79]],dtype=vertices.dtype),torch.tensor([.4,.9],dtype=vertices.dtype))
    base.match_weight=.1
    if frozen:base.prepare_fixed_p1_sampling(3,4,dtype=vertices.dtype,device=vertices.device)
    vertices.requires_grad_()
    old_total,old=base(vertices)
    wrapper=FourQuarterMindEvidence(base,3,4)
    new_total,new=wrapper(vertices)
    for key in ("strain","shape","match","oob","outside_fraction"):
        torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        if old[key].requires_grad:
            og,=torch.autograd.grad(old[key],vertices,retain_graph=True)
            ng,=torch.autograd.grad(new[key],vertices,retain_graph=True)
            torch.testing.assert_close(og,ng,rtol=0,atol=0)
    torch.testing.assert_close(new_total-new["image"],old_total-old["image"],rtol=2e-14,atol=2e-14)
    assert wrapper.metadata["full_base_evidence_calls_per_call"]==0
    assert wrapper.metadata["denominator_factor"]==4
    assert float(base.denominator)==float(base.mask.sum())


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_same_fields_identity_and_original_pixel_mask_quarter_positions(diagonal):
    base,_=fixture(diagonal)
    base.moving=base.fixed
    base.moving_feature=base.fixed_feature
    base.matrix=torch.eye(2,dtype=base.matrix.dtype);base.offset=torch.zeros_like(base.offset)
    wrapper=FourQuarterMindEvidence(base,3,4)
    total,parts=wrapper(grid(3,4))
    assert float(parts["image"])<2e-15
    arrays=pixel_quarter_queries(5,7)
    assert len(arrays)==4
    torch.testing.assert_close(arrays[0][0,0,0],torch.tensor([.25/7,.25/5],dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(arrays[-1][0,-1,-1],torch.tensor([6.75/7,4.75/5],dtype=torch.float64),rtol=0,atol=0)
    assert wrapper.resident_constant_bytes>0
    assert wrapper.metadata["fixed_descriptor_interpolations_setup"]==4
    assert wrapper.metadata["moving_descriptor_interpolations_per_call"]==4
    assert wrapper.metadata["feature_extractions_setup"]==0


def test_no_feature_reextraction_or_full_base_call_mixed_precision_and_nested_prior(monkeypatch):
    base,vertices=fixture(image_dtype=torch.float32)
    # Factor1 nested path as used by the final-stage suffix, actual square grid.
    base=base.coarse_nested_evidence(3,3,dtype=vertices.dtype,device=vertices.device)
    vertices=grid(3)
    old_total,old=base(vertices)
    def forbidden(*args,**kwargs):raise AssertionError("must not recompute features/fullbase")
    monkeypatch.setattr("tools.coordinated_real_case.self_similarity",forbidden)
    wrapper=FourQuarterMindEvidence(base,3)
    monkeypatch.setattr(Evidence,"__call__",forbidden)
    _,parts=wrapper(vertices.requires_grad_())
    assert parts["image"].dtype==torch.float32
    for key in ("strain","shape","oob","outside_fraction"):
        torch.testing.assert_close(parts[key],old[key],rtol=0,atol=0)
    grad,=torch.autograd.grad(parts["image"],vertices)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("case",["loss","order","q1","field_grad","field_nan","affine_grad","affine_negative",
    "bad_mask","denominator","grid","dtype","center_grid","vertices_batch","vertices_precision",
    "strain_model","missing_matches"])
def test_unsupported_configs_and_nonfinite_fields_guard(case):
    base,vertices=fixture()
    args=(3,4);kwargs={}
    if case=="loss":base.loss="local_ncc"
    elif case=="order":base.mind_order="after_warp"
    elif case=="q1":base.interpolation="q1"
    elif case=="field_grad":base.fixed_feature.requires_grad_()
    elif case=="field_nan":base.moving_feature[0,0,0,0]=float("nan")
    elif case=="affine_grad":base.matrix.requires_grad_()
    elif case=="affine_negative":base.matrix[0,0]=-1
    elif case=="bad_mask":base.mask[0,0,0,0]=-1
    elif case=="denominator":base.denominator=base.denominator+1
    elif case=="grid":args=(True,4)
    elif case=="dtype":kwargs["dtype"]=torch.float16
    elif case=="center_grid":base.prepare_fixed_p1_sampling(5,5,dtype=vertices.dtype,device=vertices.device)
    elif case=="strain_model":base.strain_model="bad"
    elif case=="missing_matches":base.match_weight=.1
    with pytest.raises(ValueError):
        wrapper=FourQuarterMindEvidence(base,*args,**kwargs)
        if case=="vertices_batch":wrapper(vertices.repeat(2,1,1,1))
        elif case=="vertices_precision":wrapper(vertices.float())


@pytest.mark.parametrize("height,width,dtype",[(0,7,torch.float64),(5,False,torch.float64),(5,7,torch.float16)])
def test_query_size_precision_guards(height,width,dtype):
    with pytest.raises(ValueError):pixel_quarter_queries(height,width,dtype=dtype)


def test_nonfinite_geometry_never_becomes_a_finite_complete_trial():
    base,vertices=fixture()
    wrapper=FourQuarterMindEvidence(base,3,4)
    vertices[0,1,1,0]=float("nan")
    total,parts=wrapper(vertices)
    assert not torch.isfinite(total)
    # Geometry validity is owned by the existing safe layer/check, not sampler.


def test_existing_displacement_gradient_prior_unchanged_and_rectangular_scope_guard():
    base,_=fixture()
    base.strain_model="displacement_gradient"
    vertices=grid(3)
    vertices[:,1,1]+=.01
    vertices.requires_grad_()
    old=base(vertices)[1]
    new=FourQuarterMindEvidence(base,3)(vertices)[1]
    torch.testing.assert_close(old["strain"],new["strain"],rtol=0,atol=0)
    og,=torch.autograd.grad(old["strain"],vertices)
    ng,=torch.autograd.grad(new["strain"],vertices)
    torch.testing.assert_close(og,ng,rtol=0,atol=0)
    # The legacy membrane function builds a square reference: do not invent a
    # rectangular replacement and call it the unchanged original functional.
    with pytest.raises(ValueError,match="square"):
        FourQuarterMindEvidence(base,3,4)


def test_device_alias_normalization_deterministic_cpu_and_cuda_indices(monkeypatch):
    base,vertices=fixture()
    normal=FourQuarterMindEvidence(base,3,4,device="cpu")
    alias=FourQuarterMindEvidence(base,3,4,device="cpu:0")
    assert alias.device==base.matrix.device
    torch.testing.assert_close(normal(vertices)[0],alias(vertices)[0],rtol=0,atol=0)
    # No CUDA tensor allocation needed to check current-index alias semantics.
    monkeypatch.setattr(torch.cuda,"current_device",lambda:2)
    assert _normalize_evidence_device("cuda",torch.device("cuda:2"))==torch.device("cuda:2")
    assert _normalize_evidence_device(None,torch.device("cuda:1"))==torch.device("cuda:1")
    with pytest.raises(ValueError,match="index"):
        _normalize_evidence_device("cuda",torch.device("cuda:1"))
    with pytest.raises(ValueError,match="index"):
        _normalize_evidence_device("cuda:1",torch.device("cuda:2"))
    with pytest.raises(ValueError,match="device"):
        _normalize_evidence_device("cuda",torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA fixture requires an available device")
def test_cuda_unindexed_evidence_device_values_and_vertex_gradient():
    base,vertices=fixture()
    device=torch.device("cuda",torch.cuda.current_device())
    for key in ("fixed","moving","fixed_feature","moving_feature","matrix","offset","mask","denominator"):
        setattr(base,key,getattr(base,key).to(device))
    vertices=vertices.to(device).requires_grad_()
    indexed=FourQuarterMindEvidence(base,3,4,device=device)
    unindexed=FourQuarterMindEvidence(base,3,4,device="cuda")
    assert unindexed.device==device
    loss_i=indexed(vertices)[0];loss_u=unindexed(vertices)[0]
    torch.testing.assert_close(loss_i,loss_u,rtol=0,atol=0)
    gi,=torch.autograd.grad(loss_i,vertices);gu,=torch.autograd.grad(loss_u,vertices)
    # CUDA raster/P1 scatter backward can differ in atomic addition order even
    # for identical operators; no bitwise CUDA-gradient claim is made.
    torch.testing.assert_close(gi,gu,rtol=2e-12,atol=2e-12)
    # Alias normalization does not disable finite/immutable field guards.
    base.moving_feature[0,0,0,0]=float("nan")
    with pytest.raises(ValueError,match="finite immutable"):
        FourQuarterMindEvidence(base,3,4,device="cuda")
