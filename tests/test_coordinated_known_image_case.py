import numpy as np
import pytest
import torch

from tools.coordinated_geometry_pilot import reference_grid, target_map
from tools.coordinated_known_image_case import (
    generate_fixed, generic_corner_ratio, held_out_map_metrics, sample_q1_queries,
    objective_call_count,
)


def test_identity_pixel_centers_preserve_rectangular_texture():
    moving = torch.arange(99, dtype=torch.float64).reshape(1, 1, 9, 11)/100
    axis_y, axis_x = torch.linspace(0, 1, 5, dtype=torch.float64), torch.linspace(0, 1, 7, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    vertices = torch.stack((xx, yy), -1)[None].double()
    assert torch.allclose(generate_fixed(moving, vertices), moving, atol=1e-14)


def test_independent_query_interpolator_recovers_affine_non_square_map():
    y, x = torch.meshgrid(torch.linspace(0, 1, 5, dtype=torch.float64),
                          torch.linspace(0, 1, 7, dtype=torch.float64), indexing="ij")
    vertices = torch.stack((2*x+3*y+.1, -x+4*y-.2), -1)[None]
    points = torch.tensor([[.13, .27], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    expected = torch.stack((2*points[:, 0]+3*points[:, 1]+.1,
                            -points[:, 0]+4*points[:, 1]-.2), -1)
    assert torch.allclose(sample_q1_queries(vertices, points), expected, atol=1e-14)


@pytest.mark.parametrize("name", ["wide_shear", "local_rotation", "coarse_fine"])
def test_actual_257_targets_have_all_corners_and_exact_boundary(name):
    reference = reference_grid(257)
    target = target_map(reference, name)
    assert generic_corner_ratio(target, reference) > .001
    for edge in ((slice(None), 0), (slice(None), -1)):
        assert torch.equal(target[edge], reference[edge])
    assert torch.equal(target[:, :, 0], reference[:, :, 0])
    assert torch.equal(target[:, :, -1], reference[:, :, -1])


def test_held_out_queries_are_evaluation_only_and_units_are_explicit():
    target = reference_grid(9)
    estimate = target+torch.tensor([.01, -.02], dtype=torch.float64)
    report = held_out_map_metrics(estimate, target, 512, count=31)
    assert report["euclidean_query_rmse_normalized"] == pytest.approx(np.sqrt(.0005))
    assert report["euclidean_query_rmse_canvas_pixels"] == pytest.approx(512*np.sqrt(.0005))
    assert report["query_count"] == 31
    assert report["queries_used_for_optimization"] is False


def test_objective_call_count_distinguishes_old_and_new_stage_reports():
    old=dict(evaluations=120,stages=[{} for _ in range(20)])
    new=dict(evaluations=120,stages=[{"accepted_full_total":.1} for _ in range(20)])
    assert objective_call_count(old)==142
    assert objective_call_count(new)==162
    assert objective_call_count({**new,"objective_evaluations_total":163})==163
    assert objective_call_count({**new,"objective_evaluations":164})==164


def _independent_p1(vertices,points,diagonal):
    rows,columns=vertices.shape[:2]
    yy,xx=np.meshgrid(np.arange(rows)/(rows-1),np.arange(columns)/(columns-1),indexing="ij")
    source=np.stack((xx,yy),-1).reshape(-1,2)
    table=vertices.reshape(-1,2)
    triangles=[]
    for r in range(rows-1):
        for c in range(columns-1):
            a=r*columns+c;b=a+1;d=a+columns;ne=d+1
            triangles.extend(((a,b,ne),(a,ne,d)) if diagonal=="ac" else ((a,b,d),(b,ne,d)))
    values=[]
    for point in points:
        for ids in triangles:
            matrix=np.vstack((source[list(ids)].T,np.ones(3)))
            weights=np.linalg.solve(matrix,np.r_[point,1.])
            if weights.min()>=-1e-12:
                values.append(weights@table[list(ids)])
                break
        else:
            raise AssertionError("independent triangle search found no containing source face")
    return np.array(values)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_root_p1_sampler_matches_independent_face_search_and_endpoints(diagonal):
    from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries
    rng=np.random.default_rng(187)
    vertices=torch.from_numpy(rng.normal(size=(2,4,7,2)))
    points=np.concatenate((rng.uniform(0,1,(31,2)),[[0,0],[1,0],[0,1],[1,1]]))
    actual=p1_map_at_queries(vertices,torch.from_numpy(points)[None],diagonal).numpy()
    expected=np.stack([_independent_p1(v.numpy(),points,diagonal) for v in vertices])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-14)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_root_p1_sampler_vertex_and_query_vjp_off_active_edges(diagonal):
    from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries
    torch.manual_seed(317)
    vertices=torch.randn(2,4,7,2,dtype=torch.float64,requires_grad=True)
    # Local coordinates stay away from BOTH diagonal choices and cell boundaries.
    local=torch.tensor([[.23,.37],[.71,.19],[.64,.82]],dtype=torch.float64)
    cell=torch.tensor([[1.,1.],[3.,2.],[4.,0.]],dtype=torch.float64)
    queries=((cell+local)/torch.tensor([6.,3.],dtype=torch.float64))[None].requires_grad_()
    weight=torch.randn(2,3,2,dtype=torch.float64)
    vd,qd=torch.randn_like(vertices),.1*torch.randn_like(queries)
    loss=lambda v,q:(p1_map_at_queries(v,q,diagonal)*weight).sum()
    vg,qg=torch.autograd.grad(loss(vertices,queries),(vertices,queries))
    step=1e-6
    finite=(loss(vertices+step*vd,queries+step*qd)-loss(vertices-step*vd,queries-step*qd))/(2*step)
    analytic=(vg*vd).sum()+(qg*qd).sum()
    assert float((finite-analytic).abs())<2e-8


def test_root_p1_and_q1_are_distinct_on_non_affine_quad():
    from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries
    vertices=torch.tensor([[[[0.,0.],[1.,0.]],[[0.,1.],[1.2,1.]]]],dtype=torch.float64)
    points=torch.tensor([[.3,.4]],dtype=torch.float64)
    assert float(sample_q1_queries(vertices,points)[0,0])==pytest.approx(.324)
    assert float(p1_map_at_queries(vertices,points[None],"ac")[0,0,0])==pytest.approx(.36)
    assert float(p1_map_at_queries(vertices,points[None],"bd")[0,0,0])==pytest.approx(.3)


def test_common_shape_weight_and_target_free_optimizer_interface(tmp_path,monkeypatch):
    from PIL import Image
    from types import SimpleNamespace
    import tools.coordinated_real_case as app
    from tools.coordinated_known_image_case import run
    from tools.digital_q1_real_optimize import _read_gray_thumbnail
    raster=np.arange(256,dtype=np.uint8).reshape(16,16)
    for name in ("fixed.png","moving.png"):
        Image.fromarray(raster).save(tmp_path/name)
    target=target_map(reference_grid(9),"wide_shear")
    np.savez(tmp_path/"target.npz",vertices=target.numpy())
    manifest=dict(moving="moving.png",identity_affine="identity.npz",
                  cases=[dict(target="wide_shear",fixed="fixed.png",target_archive="target.npz")])
    args=SimpleNamespace(output=tmp_path/"report.json",inputs_from=None,image_side=16,grid_side=9,
        targets=["wide_shear"],methods=["radial"],loss="mind",strain_weight=.05,shape_weight=1e-4,
        levels=[5],inner_steps=1,cycles=1,learning_rate=.004,minimum_jacobian=.001,
        image_levels=None,device="cpu",threads=2,record_stages=False,query_thresholds=[1.,5.,10.])
    def image_only_optimizer(opt):
        assert opt.shape_weight==args.shape_weight
        assert not any(word in key for key in vars(opt) for word in ("target","landmark","truth"))
        assert opt.fixed.name=="fixed.png" and opt.moving.name=="moving.png"
        assert opt.affine.name=="identity.npz"
        np.savez(opt.output,vertices=reference_grid(9).numpy())
        return dict(optimize_seconds=0.,gradient_steps=0,evaluations=2,objective_evaluations=4,
                    stages=[],failed_trials=0,median_forward_objective_seconds=0.,median_vjp_seconds=0.,
                    peak_allocated_bytes=None,initial={},final={},saved_binary_certificate={"valid":True})
    monkeypatch.setattr(app,"optimize",image_only_optimizer)
    row=run(args,manifest)["results"][0]
    fixed,_=_read_gray_thumbnail(tmp_path/"fixed.png",16)
    baseline=app.Evidence(fixed.double(),fixed.double(),torch.eye(2,dtype=torch.float64),
                          torch.zeros(2,dtype=torch.float64),"mind",.05,1.)(target)[0]
    assert row["target_objective_total"]==pytest.approx(float(baseline)+1e-4*row["target_corner_shape"])
    assert row["target_objective_parts"]["shape"]==pytest.approx(row["target_corner_shape"])


def test_native_posthoc_constant_translation_matches_physical_units():
    from tools.coordinated_known_image_case import native_dhr_query_metrics
    field=np.zeros((2,16,16),dtype=np.float32)
    field[0]=2.;field[1]=-1.
    params=dict(source_resample_ratio=1.,target_resample_ratio=1.,initial_resample_ratio=1.,
                pad_1=[[0,0],[0,0]],pad_2=[[0,0],[0,0]])
    target=reference_grid(9)+torch.tensor([2/16,-1/16],dtype=torch.float64)
    report=native_dhr_query_metrics(field,params,target,16,count=67)
    assert report["euclidean_query_rmse_canvas_pixels"]<2e-6
    assert report["query_seed"]==20261001 and report["denominator"]==67
    assert report["queries_dropped"]==0 and report["out_of_unit_square_predictions"]>0
