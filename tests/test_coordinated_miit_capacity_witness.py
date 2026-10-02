import argparse
import json

import torch
import numpy as np
import pytest

from tools import coordinated_miit_capacity_witness as module
from test_coordinated_miit_trajectory_score import cohort
from test_coordinated_miit_score import manifest


def grid(rows=9,columns=9):
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),
        torch.linspace(0,1,columns,dtype=torch.float64),indexing="ij")
    return torch.stack((x,y),-1)[None]


def oracle(v,q,target,a=None,b=None):
    return module.OracleObjective(v,torch.eye(2,dtype=v.dtype) if a is None else a,
        torch.zeros(2,dtype=v.dtype) if b is None else b,q,target)


def test_physical_units_full_vertex_gradient_and_affine_exactly_once():
    v=grid();q=torch.tensor([[.5,.5]],dtype=v.dtype);a=torch.tensor([[2.,.1],[0.,1.]],dtype=v.dtype)
    b=torch.tensor([.03,-.02],dtype=v.dtype);target=q@a.T+b
    objective=oracle(v,q,target,a,b)
    current=v.clone();current[0,4,4,0]+=.01;current.requires_grad_()
    value=objective(current);gradient,=torch.autograd.grad(value,current)
    assert value.item()==pytest.approx((512*2*.01)**2,rel=1e-13)
    assert gradient[0,4,4,0].item()==pytest.approx(2*512**2*4*.01,rel=1e-13)
    direction=torch.randn(v.shape,dtype=v.dtype,generator=torch.Generator().manual_seed(88))*.1
    h=1e-6;fd=(objective(current.detach()+h*direction)-objective(current.detach()-h*direction))/(2*h)
    torch.testing.assert_close((gradient*direction).sum(),fd,rtol=1e-9,atol=1e-7)


def test_identity_already_exact_stays_identity_budget_counts_and_boundary():
    v=grid(9,11);q=torch.tensor([[.31,.42],[.6,.7]],dtype=v.dtype)
    objective=oracle(v,q,q)
    result=module.fit_oracle(v,v,objective,levels=(3,5),inner_steps=2)
    assert result["final_oracle_loss"]<1e-20
    assert torch.equal(result["vertices"],v)
    assert result["counts"]["gradient_steps"]==8
    assert result["counts"]["decoder_trials"]==12
    assert result["counts"]["oracle_objective_evaluations"]==18  # 1+4anchors+12trials+1
    assert result["counts"]["failed_trials"]==0 and result["budget_complete"]
    assert result["final_geometry"]["exact_boundary"]
    assert all(s["accepted_oracle_loss"]<=s["anchor_oracle_loss"] for s in result["stages"])


def test_small_legal_shift_target_progress_no_boundary_repair():
    v=grid();q=torch.tensor([[.5,.5],[.25,.375],[.625,.625]],dtype=v.dtype)
    target=q+torch.tensor([.003,-.002],dtype=v.dtype)
    result=module.fit_oracle(v,v,oracle(v,q,target),levels=(3,5,9),inner_steps=3,learning_rate=.001)
    assert result["final_oracle_loss"]<result["initial_oracle_loss"]*.15
    assert result["final_geometry"]["valid"] and result["counts"]["gradient_steps"]==18
    for index in ((slice(None),0,slice(None)),(slice(None),-1,slice(None)),(slice(None),slice(None),0),(slice(None),slice(None),-1)):
        assert torch.equal(result["vertices"][index],v[index])


@pytest.mark.parametrize("bad",["objective","gradient","geometry"])
def test_failed_trial_retains_valid_anchor_and_actual_incomplete_counts(monkeypatch,bad):
    v=grid();q=torch.tensor([[.5,.5]],dtype=v.dtype);base=oracle(v,q,q+.01)
    if bad=="objective":
        def objective(y):return base(y)*float("inf") if y.requires_grad else base(y)
        objective.errors=base.errors
    else:objective=base
    original=module.FrozenAnchorCoordinatedUpdate.__call__
    def corrupt(self,proposal,**kw):
        result=original(self,proposal,**kw)
        if bad=="gradient":result.vertices.register_hook(lambda g:torch.full_like(g,float("nan")))
        elif bad=="geometry":
            result.vertices=result.vertices.clone();result.vertices[:,4,4,0]=float("inf")
        return result
    monkeypatch.setattr(module.FrozenAnchorCoordinatedUpdate,"__call__",corrupt)
    result=module.fit_oracle(v,v,objective,levels=(3,),inner_steps=2)
    assert result["counts"]["gradient_steps"]==0 and result["counts"]["failed_trials"]==2
    assert not result["budget_complete"] and result["final_geometry"]["valid"]
    assert torch.equal(result["vertices"],v)
    json.dumps(module._json_safe(result["failures"]),allow_nan=False)


@pytest.mark.parametrize("bad",["trainable","zero_reference","steps","query","target"])
def test_preconditions_not_silently_dropped(bad):
    v=grid();q=torch.tensor([[.5,.5]],dtype=v.dtype);ref=v.clone();steps=1
    if bad=="trainable":v.requires_grad_()
    elif bad=="zero_reference":ref.zero_()
    elif bad=="steps":steps=True
    elif bad=="query":q.requires_grad_()
    else:q[0,0]=float("nan")
    with pytest.raises(ValueError):module.fit_oracle(v,ref,oracle(v,q,q),levels=(3,),inner_steps=steps)


@pytest.mark.parametrize("identity",[False,True])
def test_complete_synthetic_production_recipe_explicit_oracle_export(cohort,monkeypatch,identity):
    path,native,_=cohort;calls=[]
    if identity:
        source=json.loads(path.read_text())["rows"][1]
        matrix=np.eye(2,dtype=np.float32);offset=np.zeros(2,dtype=np.float32)
        np.savez(path.parent/source["affine"],post_affine_matrix=matrix,post_affine_offset=offset)
        for record in source["methods"].values():
            if "output" not in record:continue
            saved_path=path.parent/record["output"]
            with np.load(saved_path) as saved:reference=saved["boundary_reference"].copy()
            np.savez(saved_path,vertices=reference,boundary_reference=reference,
                post_affine_matrix=matrix,post_affine_offset=offset,interpolation=np.asarray("p1_ac"))
    class OriginalE1:
        def __call__(self,y):calls.append(y.clone());return y.new_tensor(.5),dict(image=y.new_tensor(.4),strain=y.new_tensor(.1))
    def evidence(source,directory,config,a,b,device):
        assert source["name"]=="miit_7_to_8" and config.get("mind_frame","original")=="original"
        return OriginalE1(),dict(name="mock original E1, no speed/evidence claim")
    monkeypatch.setattr(module,"_original_evidence",evidence)
    output=path.parent/"explicit_label_oracle.npz"
    result=module.run(argparse.Namespace(predictions=path,source_data=native,output=output,device="cpu",threads=1))
    assert result["label_oracle"] and result["landmarks_used"] and result["not_registration_initializer_or_training_teacher"]
    assert len(result["nominal_ids"])==124 and len(result["available_ids"])==107
    assert len(calls)==result["original_e1_evaluations"]==2
    assert result["counts"]["gradient_steps"]==300 and result["counts"]["decoder_trials"]==310
    assert result["counts"]["oracle_objective_evaluations"]==322
    assert result["saved_binary_certificate"]["valid"] and result["budget_complete"]
    assert len(result["stages"])==10 and result["stages"][0]["physical_lr"]==.004
    assert result["stages"][-1]["physical_lr"]==.004*16/256
    assert len(result["end_errors"]["canvas_pixels"]["per_label"])==107
    assert result["counts"]["accepted_metric_queries"]==10
    assert result["counts"]["export_metric_queries"]==2 and result["counts"]["nonobjective_metric_queries"]==12
    assert result["counts"]["geometry_passes"]==310
    if identity:assert result["first_certified_max_error_le_one"] is not None
    if result["first_certified_max_error_le_one"] is not None:
        threshold_path=module.Path(result["first_threshold_output"])
        assert threshold_path.exists() and module.certify_q1_binary_map(threshold_path)["valid"]
        with np.load(threshold_path) as saved:
            assert saved["label_oracle"].item() and saved["landmarks_used"].item()
            assert saved["interpolation"].item()=="p1_ac"
            if identity:
                np.testing.assert_array_equal(saved["vertices"],saved["boundary_reference"])
                np.testing.assert_array_equal(saved["post_affine_matrix"],matrix)
                np.testing.assert_array_equal(saved["post_affine_offset"],offset)
    with np.load(output) as saved:
        assert saved["label_oracle"].item() and saved["landmarks_used"].item()
    assert json.loads(path.read_text())["annotations_read"] is False  # No production-manifest edits.


@pytest.mark.parametrize("bad",["missing","wrong_id","pair"])
def test_annotation_identity_count_and_pair_guards(cohort,bad):
    path,native,_=cohort;file=native/"8"/"landmarks"/"08.csv"
    if bad=="missing":file.write_text(file.read_text().replace("id123,inf,inf\n",""))
    elif bad=="wrong_id":file.write_text(file.read_text().replace("id123,inf,inf","OTHER,inf,inf"))
    args=argparse.Namespace(predictions=path,source_data=native,output=path.parent/"new.npz",device="cpu")
    if bad=="pair":args.pair_name="miit_2_to_3"
    with pytest.raises(ValueError):module.run(args)


def test_unfinished_manifest_precedes_label_access(tmp_path,monkeypatch):
    value=manifest();value["prediction_complete"]=False
    path=tmp_path/"predictions.json";path.write_text(json.dumps(value))
    monkeypatch.setattr(module,"read_points",lambda *a,**kw:pytest.fail("labels read"))
    with pytest.raises(ValueError):module.run(argparse.Namespace(predictions=path,source_data=tmp_path,output=tmp_path/"new.npz",device="cpu"))


def test_corrected_f2_exact_controls_budget_cycles_passes_and_certified_threshold(monkeypatch):
    v=grid();q=torch.tensor([[.5,.5],[.3,.4]],dtype=v.dtype)
    objective=oracle(v,q,q+.0005);construct=module.StaggeredPatchQ1Layer;calls=[]
    def factory(*args,**kwargs):
        layer=construct(*args,**kwargs);calls.append((args,kwargs,layer));return layer
    monkeypatch.setattr(module,"StaggeredPatchQ1Layer",factory)
    result=module.fit_oracle(v,v,objective,levels=(3,5),inner_steps=2,learning_rate=.0005,method="f2",patch_cells=2)
    assert result["budget_complete"] and result["counts"]["gradient_steps"]==8
    assert result["counts"]["decoder_trials"]==12 and result["counts"]["oracle_objective_evaluations"]==18
    assert result["counts"]["geometry_passes"]==48 and result["counts"]["accepted_metric_queries"]==4
    assert [(s["cycle"],s["level"],s["direction"]) for s in result["stages"]]==[(0,3,None),(0,5,None),(1,3,None),(1,5,None)]
    assert all(s["raw_parameter_channels"]==2 and s["scalar_parameter_count"]==2*(s["level"]-2)**2 for s in result["stages"])
    assert all(kwargs==dict(proposal_mode="fixed_h",raw_span=.5,safety_fraction=.75,minimum_jacobian=.001,
        accepted_gain=1.,floor_safety_fraction=.95) for _,kwargs,_ in calls)
    assert all(p.floor_safety_fraction==.95 and p.accepted_gain==1. for _,_,l in calls for p in l.passes)
    threshold=result["first_certified_max_error_le_one"]
    assert threshold["stage"]==0 and threshold["cumulative_gradients"]==2
    assert threshold["saved_binary_certificate"]["valid"] and threshold["errors"]["maximum"]<=1.
    assert threshold["elapsed_fit_seconds"]>=threshold["candidate_metric_elapsed_fit_seconds"]
    assert result["threshold_vertex_snapshot_bytes"]==v.numel()*v.element_size()


def test_f2_default_calibrated_decode_gradient_and_strict_floor():
    v=grid();q=torch.tensor([[.45,.53]],dtype=v.dtype);objective=oracle(v,q,q+.01)
    layer=module.StaggeredPatchQ1Layer(9,2,proposal_mode="fixed_h",raw_span=.5,safety_fraction=.75,
        minimum_jacobian=.001,accepted_gain=1.,floor_safety_fraction=.95)
    coverage=module.f2_coverage(layer,9,v.dtype)
    coefficient=torch.randn(1,2,3,3,generator=torch.Generator().manual_seed(711),dtype=v.dtype)*.001
    coefficient.requires_grad_();candidate=module.decode_control(layer,"f2",v,coefficient,coverage)
    loss=objective(candidate);gradient,=torch.autograd.grad(loss,coefficient)
    direction=torch.randn(coefficient.shape,generator=torch.Generator().manual_seed(712),dtype=v.dtype)
    h=1e-7
    fd=(objective(module.decode_control(layer,"f2",v,coefficient.detach()+h*direction,coverage))-
        objective(module.decode_control(layer,"f2",v,coefficient.detach()-h*direction,coverage)))/(2*h)
    torch.testing.assert_close((gradient*direction).sum(),fd,rtol=2e-8,atol=2e-7)
    assert module._geometry(candidate.detach(),v,.001)["valid"]


def test_no_early_stop_or_threshold_artifact_when_not_attained():
    v=grid();q=torch.tensor([[.5,.5]],dtype=v.dtype)
    result=module.fit_oracle(v,v,oracle(v,q,q+.2),levels=(3,),inner_steps=1,learning_rate=1e-6,method="f2",patch_cells=2)
    assert result["counts"]["gradient_steps"]==2 and result["first_certified_max_error_le_one"] is None
    assert result["first_threshold_vertices"] is None and result["counts"]["threshold_certification_calls"]==0


@pytest.mark.parametrize("bad",["method","patch","rectangle","reference"])
def test_f2_scope_guards(bad):
    v=grid(9,11) if bad=="rectangle" else grid();ref=v.clone();q=torch.tensor([[.5,.5]],dtype=v.dtype)
    if bad=="reference":ref[:,4,4,0]+=.001
    with pytest.raises(ValueError):module.fit_oracle(v,ref,oracle(v,q,q),levels=(3,),inner_steps=1,
        method="bad" if bad=="method" else "f2",patch_cells=3 if bad=="patch" else 2)
