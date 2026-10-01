import argparse

import numpy as np
from PIL import Image
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_nested_refinement import FrozenNestedP1Refinement
from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools.coordinated_real_case import Evidence,optimize,require_nested_fine_margin


def configuration(tmp_path,**changes):
    raster=np.random.default_rng(21).integers(20,220,(16,16),dtype=np.uint8)
    Image.fromarray(raster).save(tmp_path/"fixed.png")
    Image.fromarray(np.roll(raster,1,axis=1)).save(tmp_path/"moving.png")
    np.savez(tmp_path/"affine.npz",post_affine_matrix=np.eye(2),post_affine_offset=np.zeros(2))
    options=dict(output=tmp_path/"map.npz",fixed=tmp_path/"fixed.png",moving=tmp_path/"moving.png",
        affine=tmp_path/"affine.npz",grid_side=9,image_side=16,image_levels=[8,16],levels=[5,9],
        inner_steps=2,cycles=1,learning_rate=.001,device="cpu",threads=2,precision="float64",
        image_precision="float32",loss="mind",strain_weight=3.,shape_weight=.0001,oob_weight=1.,
        method="analytic",minimum_jacobian=.001,lr_calibration="edge",interpolation="p1_ac",
        p1_sampling="frozen",geometry_backend="stage_cache",output_selection="best_full",
        control_hierarchy="nested_p1")
    options.update(changes)
    return argparse.Namespace(**options)


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("selection",["last","best_full"])
@pytest.mark.parametrize("mode",["radial","analytic"])
def test_real_tiny_nested_pipeline(tmp_path,diagonal,selection,mode):
    args=configuration(tmp_path,interpolation="p1_"+diagonal,output_selection=selection,method=mode)
    observed=[]
    def capture(vertices,metadata,elapsed):
        assert vertices.shape==(1,9,9,2) and not vertices.requires_grad
        observed.append((vertices.clone(),metadata,elapsed))
    report=optimize(args,accepted_stage_callback=capture)
    assert report["gradient_steps"]==8 and report["failed_trials"]==0
    assert report["control_sizes"]==[5,9] and report["nested_p1_cache_bytes"]==9*9*48
    assert [s["control_side"] for s in report["stages"]]==[5,5,9,9]
    assert all(t["margin"]>0 and t["coarse_margin"]>0 for t in report["trace"])
    assert report["saved_binary_certificate"]["valid"] and len(observed)==4
    with np.load(args.output) as data:
        vertices=data["vertices"]
        assert vertices.shape==(1,9,9,2)
        index=report["selected_stage"]
        if index is not None:
            np.testing.assert_array_equal(vertices,observed[index][0].numpy())
    if selection=="best_full":
        expected=min([report["initial"]["total"]]+[s["accepted_full_total"] for s in report["stages"]])
        assert report["final"]["total"]==pytest.approx(expected,abs=1e-8)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_all_fine_objective_terms_and_complete_gradient(diagonal):
    torch.manual_seed(55)
    y,x=torch.meshgrid(torch.linspace(0,1,5,dtype=torch.float64),
        torch.linspace(0,1,5,dtype=torch.float64),indexing="ij")
    vertices=torch.stack((x,y),-1)[None]
    vertices[:,1:-1,1:-1]+=.005*torch.randn(1,3,3,2,dtype=vertices.dtype)
    vertices.requires_grad_()
    fixed,moving=torch.rand(1,1,16,16,dtype=vertices.dtype),torch.rand(1,1,16,16,dtype=vertices.dtype)
    source=torch.tensor([[.27,.43],[.66,.81]],dtype=vertices.dtype)
    points=ImageCorrespondences(source,source+.002,torch.ones(2,dtype=vertices.dtype),pixel_scale=16.)
    evidence=Evidence(fixed,moving,torch.eye(2,dtype=vertices.dtype),torch.zeros(2,dtype=vertices.dtype),
        "local_ncc",3.,1.,.0001,interpolation="p1_"+diagonal,matches=points,match_weight=.1)
    evidence.prepare_fixed_p1_sampling(9,9,dtype=vertices.dtype,device="cpu")
    layer=FrozenNestedP1Refinement(5,9,diagonal=diagonal)
    a,parts_a=evidence(layer(vertices));b,parts_b=evidence(refine_p1_vertices(vertices,2,diagonal))
    for key in parts_a:
        torch.testing.assert_close(parts_a[key],parts_b[key],rtol=0,atol=0)
    ga,=torch.autograd.grad(a,vertices);gb,=torch.autograd.grad(b,vertices)
    torch.testing.assert_close(ga,gb,rtol=2e-12,atol=2e-12)
    tangent=torch.randn_like(vertices);tangent[:,0]=0;tangent[:,-1]=0;tangent[:,:,0]=0;tangent[:,:,-1]=0
    eps=1e-7
    fd=(evidence(layer(vertices.detach()+eps*tangent))[0]-evidence(layer(vertices.detach()-eps*tangent))[0])/(2*eps)
    torch.testing.assert_close(fd,(ga*tangent).sum(),rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize("changes",[dict(cycles=2),dict(method="f1"),dict(interpolation="q1"),
    dict(levels=[5,8]),dict(levels=[5]),dict(levels=[9,5])])
def test_reject_incompatible_hierarchies_before_execution(tmp_path,changes):
    with pytest.raises(ValueError):
        optimize(configuration(tmp_path,**changes))


def test_strict_fine_floor_rejects_coarse_safe_rounding_fixture():
    from tools.digital_q1_dhr_distill import identity_vertices
    from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants,validate_q1_map
    coarse=identity_vertices(3,device="cpu");coarse[0,1,1,0]=torch.nextafter(torch.tensor(.25),torch.tensor(1.))
    fine=FrozenNestedP1Refinement(3,5,dtype=torch.float32)(coarse)
    coarse_ref=q1_corner_determinants(identity_vertices(3,device="cpu").double())
    fine_ref=q1_corner_determinants(identity_vertices(5,device="cpu").double())
    assert float((q1_corner_determinants(coarse.double())/coarse_ref).amin())>.5
    assert float((q1_corner_determinants(fine.double())/fine_ref).amin())==.5
    assert validate_q1_map(fine,identity_vertices(5,device="cpu"))["valid"]
    with pytest.raises(RuntimeError,match="accepted anchor/fallback"):
        require_nested_fine_margin(fine,fine_ref,.5,"accepted anchor/fallback")


def test_earlier_coarse_stage_winner_exports_exact_final_grid(tmp_path,monkeypatch):
    # Protocol fixture, NOT an image-registration accuracy experiment.
    from tools.digital_q1_dhr_distill import identity_vertices
    target=identity_vertices(5,device="cpu").double()
    target[:,1:-1,1:-1,0]+=.001
    fine_target=refine_p1_vertices(target,2,"ac")
    def objective(self,vertices):
        assert vertices.shape==fine_target.shape
        # Flat exact optimum makes later zero proposals unable to win by roundoff.
        value=torch.relu((vertices-fine_target).abs()-1e-6).square().sum()
        zero=value*0
        return value,dict(image=value,strain=zero,oob=zero,outside_fraction=zero,shape=zero)
    monkeypatch.setattr(Evidence,"__call__",objective)
    args=configuration(tmp_path,inner_steps=1)
    snapshots=[]
    report=optimize(args,accepted_stage_callback=lambda v,m,t:snapshots.append(v.clone()))
    assert report["selected_stage"]==0 and report["final"]["total"]==0
    with np.load(args.output) as archive:
        np.testing.assert_array_equal(archive["vertices"],snapshots[0].numpy())
        assert archive["vertices"].shape==(1,9,9,2)
