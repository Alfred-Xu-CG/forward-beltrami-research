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


@pytest.mark.parametrize("interpolation,strain_model",[("q1","displacement_gradient"),
    ("p1_ac","displacement_gradient"),("p1_bd","displacement_gradient"),
    ("p1_ac","p1_arap"),("p1_bd","p1_arap")])
@pytest.mark.parametrize("coordinate_mode",["alternating","joint"])
def test_common_shape_weight_and_target_free_optimizer_interface(tmp_path,monkeypatch,interpolation,strain_model,coordinate_mode):
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
    args.interpolation=interpolation
    args.strain_model=strain_model
    args.mind_order="after_warp"
    args.coordinate_mode=coordinate_mode;args.joint_backend="cached_manual"
    args.geometry_backend="existing";args.output_selection="best_full"
    args.p1_sampling="existing" if interpolation=="q1" else "frozen"
    def image_only_optimizer(opt):
        assert opt.shape_weight==args.shape_weight
        assert opt.interpolation==interpolation and opt.p1_sampling==args.p1_sampling
        assert opt.coordinate_mode==coordinate_mode and opt.joint_backend=="cached_manual"
        assert opt.geometry_backend=="existing" and opt.output_selection=="best_full"
        assert opt.strain_model==strain_model
        assert opt.mind_order=="after_warp"
        assert opt.capture_prefix=="none"
        assert not any(word in key for key in vars(opt) for word in ("target","landmark","truth"))
        assert opt.fixed.name=="fixed.png" and opt.moving.name=="moving.png"
        assert opt.affine.name=="identity.npz"
        np.savez(opt.output,vertices=reference_grid(9).numpy())
        return dict(optimize_seconds=0.,gradient_steps=0,evaluations=2,objective_evaluations=4,
                    stages=[],failed_trials=0,median_forward_objective_seconds=0.,median_vjp_seconds=0.,
                    peak_allocated_bytes=None,initial={},final={},saved_binary_certificate={"valid":True})
    monkeypatch.setattr(app,"optimize",image_only_optimizer)
    row=run(args,manifest)["results"][0]
    assert row["estimate_interpolation"]==interpolation
    assert row["target_interpolation"]=="q1"
    fixed,_=_read_gray_thumbnail(tmp_path/"fixed.png",16)
    baseline=app.Evidence(fixed.double(),fixed.double(),torch.eye(2,dtype=torch.float64),
                          torch.zeros(2,dtype=torch.float64),"mind",.05,1.,
                          interpolation=interpolation if strain_model=="p1_arap" else "q1",strain_model=strain_model,
                          mind_order="after_warp")(target)[0]
    assert row["target_objective_total"]==pytest.approx(float(baseline)+1e-4*row["target_corner_shape"])
    assert row["target_objective_parts"]["shape"]==pytest.approx(row["target_corner_shape"])
    assert row["strain_model"]==strain_model and row["target_declared_objective_interpolation"]==interpolation
    assert row["mind_order"]=="after_warp"
    assert row["capture_prefix"]=="none" and row["capture_record"] is None
    assert row["capture_objective_evaluations"]==row["capture_geometry_attempts"]==0


def _capture_args(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(output=tmp_path/"report.json",inputs_from=None,image_side=128,grid_side=257,
        targets=["wide_shear"],methods=["analytic"],loss="mind",strain_weight=3.,strain_model="p1_arap",
        shape_weight=1e-4,levels=[17,33,65,129,257],inner_steps=30,cycles=1,learning_rate=.004,
        minimum_jacobian=.001,image_levels=[32,64,128,128,128],device="cpu",threads=2,
        record_stages=False,query_thresholds=[1.,5.,10.],interpolation="p1_ac",mind_order="transport",
        coordinate_mode="alternating",geometry_backend="stage_cache",p1_sampling="frozen",
        output_selection="best_full",capture_prefix="mind_discrete")


def test_capture_prefix_forwards_configuration_and_preserves_authoritative_call_count(tmp_path,monkeypatch):
    from PIL import Image
    import tools.coordinated_real_case as app
    from tools.coordinated_known_image_case import run
    torch.set_num_threads(2)
    args=_capture_args(tmp_path)
    raster=np.arange(128*128,dtype=np.uint8).reshape(128,128)
    for name in ("fixed.png","moving.png"):
        Image.fromarray(raster).save(tmp_path/name)
    target=target_map(reference_grid(257),"wide_shear")
    np.savez(tmp_path/"target.npz",vertices=target.numpy())
    manifest=dict(moving="moving.png",identity_affine="identity.npz",
        cases=[dict(target="wide_shear",fixed="fixed.png",target_archive="target.npz")])
    capture=dict(mode="mind_discrete",geometry_attempts=2,objective_evaluations=2,accepted_axes=1,
        search_seconds=.11,complete_prefix_seconds=.14,search_diagnostics={"labels":81},attempts=[])
    calls=[]
    def image_only_optimizer(opt):
        calls.append(opt)
        assert opt.capture_prefix=="mind_discrete"
        assert opt.precision==opt.image_precision=="float64"
        assert opt.strain_weight==3. and opt.strain_model=="p1_arap" and opt.shape_weight==1e-4
        assert opt.cycles==1 and opt.inner_steps==30 and opt.levels==args.levels
        assert opt.interpolation=="p1_ac" and opt.output_selection=="best_full"
        assert not any(word in key for key in vars(opt) for word in ("target","landmark","truth"))
        np.savez(opt.output,vertices=reference_grid(257).numpy())
        return dict(optimize_seconds=.8,gradient_steps=300,evaluations=310,objective_evaluations=334,
            stages=[{"accepted_full_total":.1} for _ in range(10)],failed_trials=0,
            median_forward_objective_seconds=.001,median_vjp_seconds=.002,peak_allocated_bytes=None,
            initial={},final={},saved_binary_certificate={"valid":True},capture_prefix="mind_discrete",
            capture_record=capture,capture_objective_evaluations=2)
    monkeypatch.setattr(app,"optimize",image_only_optimizer)
    payload=run(args,manifest)
    row=payload["results"][0]
    assert len(calls)==1 and payload["configuration"]["capture_prefix"]=="mind_discrete"
    assert row["capture_record"]==capture and row["capture_objective_evaluations"]==2
    assert row["capture_geometry_attempts"]==2
    assert row["capture_search_seconds"]==.11 and row["capture_complete_prefix_seconds"]==.14
    assert row["optimize_seconds"]==.8 and row["gradient_steps"]==300
    assert row["objective_evaluations_including_stage_anchors"]==334  # NOT 336: already includes prefix.
    assert row["target_interpolation"]=="q1" and row["landmarks_or_target_used_by_optimizer"] is False


@pytest.mark.parametrize("change",[
    {"methods":["analytic","f2"]},{"grid_side":513},{"interpolation":"p1_bd"},
    {"coordinate_mode":"joint"},{"loss":"local_ncc"},{"mind_order":"after_warp"},
    {"levels":[17,65,129,257]},{"image_levels":[32,64,256,512,512]},
    {"capture_prefix":"unknown"},
])
def test_unsupported_capture_matrix_refused_before_any_input_or_optimizer(tmp_path,monkeypatch,change):
    import tools.coordinated_real_case as app
    import tools.digital_q1_real_optimize as images
    from tools.coordinated_known_image_case import run
    args=_capture_args(tmp_path)
    for key,value in change.items():
        setattr(args,key,value)
    monkeypatch.setattr(app,"optimize",lambda *a,**k:pytest.fail("optimizer must not start"))
    monkeypatch.setattr(images,"_read_gray_thumbnail",lambda *a,**k:pytest.fail("input must not be opened"))
    with pytest.raises(ValueError,match="capture"):
        run(args,{})
    assert not args.output.exists()


def test_capture_cli_guard_runs_before_preparation(tmp_path,monkeypatch):
    import sys
    import tools.coordinated_known_image_case as known
    monkeypatch.setattr(known,"prepare",lambda *a:pytest.fail("preparation must not start"))
    monkeypatch.setattr(sys,"argv",["known","--moving",str(tmp_path/"moving.png"),
        "--output",str(tmp_path/"report.json"),"--capture-prefix","mind_discrete",
        "--interpolation","p1_ac","--image-levels","32","64","128","256","512"])
    with pytest.raises(SystemExit) as error:
        known.main()  # Default four-method matrix is incompatible, even before prepare-only.
    assert error.value.code==2


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


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_declared_estimate_metrics_use_actual_p1_not_q1(diagonal):
    from tools.coordinated_known_image_case import sample_declared_queries
    rng=np.random.default_rng(661)
    vertices=torch.from_numpy(rng.normal(size=(1,4,7,2)))
    points=rng.uniform(0,1,(23,2))
    actual=sample_declared_queries(vertices,torch.from_numpy(points),"p1_"+diagonal)
    np.testing.assert_allclose(actual.numpy(),_independent_p1(vertices[0].numpy(),points,diagonal),atol=2e-14,rtol=0)
    report=held_out_map_metrics(vertices,vertices,512,count=31,estimate_interpolation="p1_"+diagonal)
    assert report["euclidean_query_rmse_canvas_pixels"]>1
    assert report["estimate_interpolation"]=="p1_"+diagonal
    assert report["target_interpolation"]=="q1"


@pytest.mark.parametrize("interpolation",["p1_ac","p1_bd"])
def test_declared_p1_raster_identity_preserves_non_square_texture(interpolation):
    moving=torch.arange(99,dtype=torch.float64).reshape(1,1,9,11)/100
    y,x=torch.meshgrid(torch.linspace(0,1,5,dtype=torch.float64),torch.linspace(0,1,7,dtype=torch.float64),indexing="ij")
    vertices=torch.stack((x,y),-1)[None]
    torch.testing.assert_close(generate_fixed(moving,vertices,interpolation),moving,atol=1e-14,rtol=0)
