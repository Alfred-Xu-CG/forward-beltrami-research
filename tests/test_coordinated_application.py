"""Independent geometry recomputation and image-evidence convention checks."""
import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update, single_direction_corner_change
from qcopt.neural_bijection.dense.coordinated_patches import CoordinatedPatchQ1Pass
from tools.coordinated_real_case import Evidence,corner_symmetric_dirichlet


def triangles_numpy(vertices):
    # Generic homogeneous 3x3 determinants: independent of production cross
    # differences and production Q1 determinant helper.
    rows, columns = vertices.shape[:2]
    answer = np.empty((rows-1, columns-1, 4))
    for i in range(rows-1):
        for j in range(columns-1):
            a,b,c,d = vertices[i,j],vertices[i,j+1],vertices[i+1,j+1],vertices[i+1,j]
            for k, points in enumerate(((a,b,d),(a,b,c),(d,b,c),(a,c,d))):
                matrix = np.concatenate((np.asarray(points), np.ones((3,1))), axis=1)
                answer[i,j,k] = np.linalg.det(matrix)
    return answer


@pytest.mark.parametrize("mode", ["radial", "analytic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_separate_triangle_path(mode, dtype):
    gen = np.random.default_rng(403)
    y,x = np.meshgrid(np.linspace(0,1,7), np.linspace(0,1,9), indexing="ij")
    anchor = np.stack((x,y),-1)
    anchor[1:-1,1:-1] += gen.normal(0,.004,(5,7,2))
    proposal = gen.normal(0,.2,(7,9))
    source = torch.tensor(anchor[None],dtype=dtype)
    raw = torch.tensor(proposal[None],dtype=dtype)
    e = np.array([.8,.6])
    predicted = single_direction_corner_change(source.double(), raw.double(), tuple(e))[0].numpy()
    direct = triangles_numpy(source.double()[0].numpy()+raw.double()[0].numpy()[...,None]*e)-triangles_numpy(source.double()[0].numpy())
    np.testing.assert_allclose(predicted,direct,atol=3e-16,rtol=1e-12)
    result = CoordinatedQ1Update(tuple(e),mode=mode,minimum_jacobian=.001)(source,raw)
    corners = triangles_numpy(result.vertices.double()[0].numpy())
    assert corners.min() > .001/48
    assert np.array_equal(result.vertices[0,0].numpy(), source[0,0].numpy())


@pytest.mark.parametrize("loss", ["mind", "local_ncc"])
def test_original_evidence_identity_and_oob_fixed_denominator(loss):
    torch.manual_seed(904)
    image = torch.rand(1,1,32,32)
    y,x = torch.meshgrid(torch.linspace(0,1,9),torch.linspace(0,1,9),indexing="ij")
    vertices = torch.stack((x,y),-1)[None].requires_grad_()
    evidence = Evidence(image,image,torch.eye(2),torch.zeros(2),loss,.05,1.)
    initial,parts = evidence(vertices)
    if loss == "mind":
        assert float(parts["image"]) < 2e-6
    assert float(parts["outside_fraction"]) == 0
    initial.backward()
    assert bool(torch.isfinite(vertices.grad).all())
    denominator = float(evidence.denominator)
    shifted = vertices.detach()+torch.tensor([2.,0.])
    worse, shifted_parts = evidence(shifted)
    assert float(evidence.denominator) == denominator
    assert float(shifted_parts["outside_fraction"]) == 1
    assert float(shifted_parts["oob"]) > 1
    assert float(worse) > float(initial)


def test_independent_tiny_inactive_analytic_gradient():
    y,x = torch.meshgrid(torch.linspace(0,1,4,dtype=torch.float64),
                         torch.linspace(0,1,4,dtype=torch.float64),indexing="ij")
    vertices = torch.stack((x,y),-1)[None].requires_grad_()
    proposal = torch.zeros(1,4,4,dtype=torch.float64)
    proposal[0,1,1] = 1e-305
    proposal.requires_grad_()
    result = CoordinatedQ1Update(mode="analytic")(vertices,proposal)
    assert float(result.scale) == 1
    g_vertices,g_proposal = torch.autograd.grad(result.vertices.sum(),(vertices,proposal))
    assert bool(torch.isfinite(g_vertices).all() and torch.isfinite(g_proposal).all())
    assert float(g_proposal[0,1,1]) == 1


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_regional_separate_reconstruction(mode):
    y,x = torch.meshgrid(torch.linspace(0,1,13,dtype=torch.float64),
                         torch.linspace(0,1,18,dtype=torch.float64),indexing="ij")
    reference = torch.stack((x,y),-1)[None]
    current = reference.clone()
    current[0,5,5,0] -= .997/17
    raw = torch.full((1,13,18),.12,dtype=torch.float64,requires_grad=True)
    layer = CoordinatedPatchQ1Pass(13,18,patch_cells=4,offset_row=1,offset_column=1,mode=mode)
    result = layer(current,raw,reference=reference)
    # Recompute all grid triangles, not just gathered patch subarrays.
    q = triangles_numpy(result.vertices[0].detach().numpy())
    assert q.min()*12*17 > .001
    assert torch.equal(result.vertices[:,0],reference[:,0])
    assert torch.equal(result.vertices[:,:,-1],reference[:,:,-1])
    assert result.covered_cells + result.uncovered_cells == 12*17
    actual = (result.vertices-current)[...,0]
    torch.testing.assert_close(actual,result.amplitude,rtol=0,atol=8e-17)
    result.vertices.square().sum().backward()
    assert bool(torch.isfinite(raw.grad).all())


@pytest.mark.parametrize("matrix", [[[1.,0.],[0.,1.]],[[0.,-1.],[1.,0.]],
                                    [[.02,0.],[0.,1.]],[[1.2,.3],[.1,.8]]])
def test_corner_shape_independent_inverse(matrix):
    a=np.asarray(matrix,dtype=np.float64)
    y,x=torch.meshgrid(torch.linspace(0,1,7,dtype=torch.float64),
                       torch.linspace(0,1,9,dtype=torch.float64),indexing="ij")
    vertices=torch.stack((x,y),-1)[None]@torch.from_numpy(a).T
    expected=np.linalg.norm(a,"fro")**2+np.linalg.norm(np.linalg.inv(a),"fro")**2-4
    assert float(corner_symmetric_dirichlet(vertices))==pytest.approx(expected,abs=2e-10)


def test_corner_shape_directional_derivative():
    torch.manual_seed(631)
    y,x=torch.meshgrid(torch.linspace(0,1,9,dtype=torch.float64),
                       torch.linspace(0,1,9,dtype=torch.float64),indexing="ij")
    vertices=(torch.stack((x,y),-1)[None]+.003*torch.randn(1,9,9,2,dtype=torch.float64)).requires_grad_()
    direction=torch.randn_like(vertices)*.01
    grad,=torch.autograd.grad(corner_symmetric_dirichlet(vertices),vertices)
    step=1e-6
    finite=(corner_symmetric_dirichlet(vertices+step*direction)-corner_symmetric_dirichlet(vertices-step*direction))/(2*step)
    torch.testing.assert_close((grad*direction).sum(),finite,rtol=1e-6,atol=2e-9)


def test_mixed_evidence_keeps_double_geometry_gradient():
    torch.manual_seed(919)
    image=torch.rand(1,1,32,32,dtype=torch.float64)
    y,x=torch.meshgrid(torch.linspace(0,1,9,dtype=torch.float64),
                       torch.linspace(0,1,9,dtype=torch.float64),indexing="ij")
    vertices=(torch.stack((x,y),-1)[None]+.001*torch.randn(1,9,9,2,dtype=torch.float64)).requires_grad_()
    matrix,offset=torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64)
    full=Evidence(image,image,matrix,offset,"mind",.05,1.,1e-4)
    mixed=Evidence(image.float(),image.float(),matrix,offset,"mind",.05,1.,1e-4)
    full_value=full(vertices)[0];mixed_value=mixed(vertices)[0]
    g_full,=torch.autograd.grad(full_value,vertices)
    g_mixed,=torch.autograd.grad(mixed_value,vertices)
    assert g_mixed.dtype==torch.float64 and bool(torch.isfinite(g_mixed).all())
    torch.testing.assert_close(mixed_value,full_value,rtol=2e-5,atol=1e-6)
    torch.testing.assert_close(g_mixed,g_full,rtol=2e-3,atol=1e-5)


def test_pyramid_fixed_mask_conservation():
    image=torch.full((1,1,32,32),.5)
    mask=torch.zeros_like(image);mask[...,3:29,6:27]=1
    reduced=torch.nn.functional.interpolate(mask,size=(8,8),mode="area")
    evidence=Evidence(torch.full((1,1,8,8),.5),torch.full((1,1,8,8),.5),
                      torch.eye(2),torch.zeros(2),"mind",.05,1.,fixed_mask=reduced)
    assert float(evidence.denominator)*16==float(mask.sum())
    assert bool(((evidence.mask>0)&(evidence.mask<1)).any())


@pytest.mark.parametrize("interpolation",["q1","p1_ac","p1_bd"])
@pytest.mark.parametrize("selection",["last","best_full"])
def test_tiny_image_continuation_stage_acceptance(tmp_path,interpolation,selection):
    import argparse
    from PIL import Image
    from tools.coordinated_real_case import optimize
    image=(np.random.default_rng(418).uniform(30,210,(16,16))).astype(np.uint8)
    Image.fromarray(image).save(tmp_path/"fixed.png")
    Image.fromarray(np.roll(image,1,axis=1)).save(tmp_path/"moving.png")
    np.savez(tmp_path/"affine.npz",post_affine_matrix=np.eye(2),post_affine_offset=np.zeros(2))
    args=argparse.Namespace(output=tmp_path/"map.npz",fixed=tmp_path/"fixed.png",moving=tmp_path/"moving.png",
        affine=tmp_path/"affine.npz",grid_side=9,image_side=16,image_levels=[8,16],levels=[5,9],
        inner_steps=1,cycles=1,learning_rate=.001,device="cpu",threads=2,precision="float64",
        image_precision="float32",loss="mind",strain_weight=.05,shape_weight=.0001,oob_weight=1.,
        method="radial",minimum_jacobian=.001,lr_calibration="edge",interpolation=interpolation,
        output_selection=selection)
    observed=[]
    def observe(snapshot,metadata,elapsed):
        assert not snapshot.requires_grad and elapsed>=0
        observed.append(metadata)
        snapshot.zero_()  # This clone must not alter the optimizer's accepted map.
    report=optimize(args,accepted_stage_callback=observe)
    assert report["gradient_steps"]==4 and report["evaluations"]==8
    assert [s["image_side"] for s in report["stages"]]==[8,8,16,16]
    assert all(s["accepted_total"]<=s["anchor_total"] for s in report["stages"])
    assert report["saved_binary_certificate"]["valid"]
    assert report["end_to_end_seconds"]>=report["optimize_seconds"]
    assert report["objective_evaluations"]==18 and len(observed)==4
    if selection=="best_full":
        expected=min([report["initial"]["total"]]+[s["accepted_full_total"] for s in report["stages"]])
        assert report["final"]["total"]==pytest.approx(expected,abs=1e-8)
        assert report["final"]["total"]<=report["terminal_full_total"]+1e-8
    with np.load(args.output) as archive:
        assert str(archive["interpolation"].item())==interpolation


def test_frozen_preprocessing_preserves_original_foreground(tmp_path,monkeypatch):
    from PIL import Image
    from tools.coordinated_real_case import load_registration_evidence
    import tools.coordinated_native_evidence as native
    image=np.full((16,16),255,dtype=np.uint8)
    image[3:11,5:13]=100
    fixed,moving=tmp_path/"fixed.png",tmp_path/"moving.png"
    Image.fromarray(image).save(fixed);Image.fromarray(image).save(moving)
    original,_,mask,_=load_registration_evidence(fixed,moving,16,dtype=torch.float64)
    assert float(mask.sum())==64 and torch.equal(mask,(original>.04).double())
    calls=[]
    def frozen(f,m,**kwargs):
        calls.append((f,m,kwargs))
        return torch.ones(1,1,16,16),torch.zeros(1,1,16,16),{"test_capture":True}
    monkeypatch.setattr(native,"load_native_preprocessed_pair",frozen)
    f,m,native_mask,metadata=load_registration_evidence(fixed,moving,16,
        preprocessing="native_dhr",dtype=torch.float64)
    assert torch.equal(native_mask,mask) and f.dtype==m.dtype==torch.float64
    assert not f.requires_grad and not m.requires_grad
    assert metadata["native_capture"]["test_capture"] and len(calls)==1
    assert calls[0][2]==dict(expected_side=16,device="cpu")
    with pytest.raises(ValueError,match="declare"):
        load_registration_evidence(fixed,moving,16,preprocessing="undeclared")


def test_default_frozen_preprocessing_is_old_evidence(tmp_path):
    from PIL import Image
    from tools.coordinated_real_case import load_registration_evidence
    from tools.digital_q1_real_optimize import _read_gray_thumbnail
    image=np.random.default_rng(463).integers(20,240,(16,16),dtype=np.uint8)
    fixed,moving=tmp_path/"fixed.png",tmp_path/"moving.png"
    Image.fromarray(image).save(fixed);Image.fromarray(image[::-1]).save(moving)
    f,m,mask,metadata=load_registration_evidence(fixed,moving,16)
    torch.testing.assert_close(f,_read_gray_thumbnail(fixed,16)[0],rtol=0,atol=0)
    torch.testing.assert_close(m,_read_gray_thumbnail(moving,16)[0],rtol=0,atol=0)
    assert torch.equal(mask,(f>.04).float()) and metadata["name"]=="raw_inverted"


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_prepared_p1_evidence_same_loss_and_vertex_gradient(diagonal):
    torch.manual_seed(614)
    image=torch.rand(1,1,32,32)
    y,x=torch.meshgrid(torch.linspace(0,1,9,dtype=torch.float64),
                       torch.linspace(0,1,9,dtype=torch.float64),indexing="ij")
    vertices=(torch.stack((x,y),-1)[None]+.002*torch.randn(1,9,9,2,dtype=torch.float64)).requires_grad_()
    evidence=Evidence(image,image,torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        "mind",3.,1.,1e-4,interpolation="p1_"+diagonal)
    old=evidence(vertices)[0];g_old,=torch.autograd.grad(old,vertices)
    evidence.prepare_fixed_p1_sampling(9,9,dtype=vertices.dtype,device=vertices.device)
    new=evidence(vertices)[0];g_new,=torch.autograd.grad(new,vertices)
    torch.testing.assert_close(old,new,rtol=0,atol=1e-14)
    torch.testing.assert_close(g_old,g_new,rtol=1e-12,atol=1e-12)
    assert sum(b.numel()*b.element_size() for b in evidence.fixed_p1_evaluator.buffers())==32*32*48
    q1=Evidence(image,image,torch.eye(2),torch.zeros(2),"mind",.05,1.)
    with pytest.raises(ValueError,match="P1"):
        q1.prepare_fixed_p1_sampling(9,9,dtype=torch.float32,device="cpu")
