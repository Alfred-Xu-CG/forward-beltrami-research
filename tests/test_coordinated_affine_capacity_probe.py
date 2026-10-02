"""Label-free convex-regression/frame fixtures, not an actual MIIT experiment."""
import json

import torch
import numpy as np
import pytest

from tools import coordinated_affine_capacity_probe as module
from test_coordinated_miit_score import manifest


def samples():
    q=np.array([(x,y) for y in (.1,.3,.6,.9) for x in (.1,.3,.6,.9)])
    return q,np.linspace(.4,1.,len(q))


@pytest.mark.parametrize("model",["similarity","affine"])
def test_exact_similarity_fit_and_stationarity(model):
    q,w=samples();a=np.array([[.8,-.1],[.1,.8]]);b=np.array([.1,.02]);y=q@a.T+b
    fit=module.fit_robust_affine(q,y,w,model=model)
    assert fit["status"]=="ok" and fit["converged"] and fit["gradient_inf"]<=1e-8
    np.testing.assert_allclose(fit["matrix"],a,atol=2e-14,rtol=0)
    np.testing.assert_allclose(fit["offset"],b,atol=2e-14,rtol=0)
    assert fit["determinant_float64_positive"] and fit["determinant_float32_positive"]
    assert fit["offset_float64_finite"] and fit["offset_float32_finite"]
    assert len(fit["singular_values"])==2 and len(fit["design_singular_values"])==(4 if model=="similarity" else 6)
    assert fit["training"]["count"]==len(q) and fit["training"]["canvas_pixels"]["maximum"]<1e-10


def test_exact_full_affine_and_heldout_not_training_rule():
    q,w=samples();y=q@np.array([[.8,.12],[-.02,.7]]).T+[.04,.08]
    result=module.compare_folds(q,y,w)
    assert result["pass"]
    for fold in result["folds"]:
        assert fold["heldout"]["affine"]["robust_loss"]<fold["heldout"]["similarity"]["robust_loss"]
        assert fold["models"]["affine"]["gradient_inf"]<=1e-8
        assert fold["training_ids"]==fold["models"]["affine"]["point_ids"]==fold["models"]["similarity"]["point_ids"]


def test_robust_outlier_stationarity_and_analytic_gradient_fd():
    q,w=samples();y=q@np.array([[.8,.1],[-.05,.75]]).T+[.03,.05];y[-1]+=[.08,-.06]
    fit=module.fit_robust_affine(q,y,w,model="affine")
    assert fit["status"]=="ok" and fit["iterations"]>0
    design=module.design_matrix(q,"affine");beta=np.array(fit["coefficients"])
    loss,g=module.loss_gradient(beta,design,y,w)
    direction=np.arange(1,7,dtype=float)/6;h=1e-7
    fd=(module.loss_gradient(beta+h*direction,design,y,w)[0]-module.loss_gradient(beta-h*direction,design,y,w)[0])/(2*h)
    assert g@direction==pytest.approx(fd,abs=2e-8)
    tensor=torch.tensor(beta,requires_grad=True);d=torch.tensor(design);target=torch.tensor(y);weights=torch.tensor(w)
    s2=(64*(d@tensor-target.ravel())).reshape(-1,2).square().sum(-1)
    oracle=(weights*s2/(torch.sqrt(1+s2)+1)).sum()/weights.sum()
    oracle.backward();np.testing.assert_allclose(g,tensor.grad.numpy(),atol=1e-12,rtol=1e-12)
    assert fit["maximum_irls_loss_increase"]<1e-12


def test_rank_negative_determinant_and_iteration_cap_fail_without_fallback():
    q,w=samples()
    deficient=module.fit_robust_affine(np.repeat([[.3,.4]],5,axis=0),np.repeat([[.5,.6]],5,axis=0),np.ones(5),model="affine")
    assert deficient["status"]=="failed" and deficient["rank"]<6
    negative=module.fit_robust_affine(q,q@np.diag([-.6,.7])+[.7,.1],w,model="affine")
    assert negative["status"]=="failed" and not negative["determinant_float64_positive"]
    y=q*.8+[.03,.07];y[-1]+=[.1,-.08]
    capped=module.fit_robust_affine(q,y,w,model="affine",max_iterations=1)
    assert capped["status"]=="failed" and not capped["converged"] and capped["iterations"]==1


def test_float32_storage_can_fail_without_projection():
    # Positive float64 determinant rounds to zero when stored as float32.
    a=np.array([[1.,1.],[1.,1.+1e-9]])
    result=module.positive_determinants(a)
    assert result["determinant_float64_positive"] and not result["determinant_float32_positive"]
    translation=module.positive_determinants(np.eye(2),np.array([1e39,0.]))
    assert translation["offset_float64_finite"] and not translation["offset_float32_finite"]


def test_fixed_fold_endpoint_and_zero_weight_empty_guards():
    q=np.array([[0.,0.],[1.,1.],[1.,.4],[.25,0.]])
    np.testing.assert_array_equal(module.spatial_parity(q),[0,0,0,1])
    with pytest.raises(ValueError):module.fit_robust_affine(q,q,np.zeros(4))
    with pytest.raises(ValueError):module.spatial_parity(q+2)
    with pytest.raises(ValueError):module.fit_robust_affine(q,q,np.ones(4),model="rigid")


def test_zero_confidence_heldout_is_explicit_failed_fold_not_nan():
    q,w=samples();parity=module.spatial_parity(q);w[parity==1]=0
    result=module.compare_folds(q,q,w)
    assert not result["pass"]
    for model in ("similarity","affine"):
        assert result["folds"][0]["models"][model]["status"]=="failed"
        assert "positive confidence mass" in result["folds"][0]["models"][model]["error"]
    json.dumps(result,allow_nan=False)


@pytest.fixture
def frozen_input(tmp_path):
    value=manifest();q,w=samples();a=np.array([[.8,-.1],[.1,.8]],dtype=np.float32);b=np.array([.1,.15],dtype=np.float32)
    y=q@np.array([[.65,.07],[-.03,.6]]).T+[.12,.2]
    p=(y-b.astype(float))@np.linalg.inv(a.astype(float)).T
    # Include one static-world excluded and one zero-confidence row.
    q=np.vstack((q,[[.2,.3],[.4,.5]]));p=np.vstack((p,[[1.,1.],[.5,.5]]));w=np.r_[w,.7,0.]
    for row in value["rows"]:
        row.update(fixed=row["name"]+"_fixed512.png",moving=row["name"]+"_moving512.png")
        row["affine"]=row["name"]+"_affine.npz"
        np.savez(tmp_path/row["affine"],post_affine_matrix=a,post_affine_offset=b)
        row["raw_matches"]={"status":"ok","path":row["name"]+"_raw.json"}
        raw=dict(status="ok",image_side=512,fixed=row["fixed"],moving=row["moving"],
            post_affine_matrix=a.tolist(),post_affine_offset=b.tolist(),source_points_unit=q.tolist(),target_points_unit=p.tolist(),
            confidence=w.tolist(),targets_manual_landmarks_or_dense_teacher_loaded=False,global_geometric_ransac_used=False)
        (tmp_path/row["raw_matches"]["path"]).write_text(json.dumps(raw))
    path=tmp_path/"predictions.json";path.write_text(json.dumps(value));return path


def test_frozen_world_once_eligibility_confidence_and_run_protocol(frozen_input,tmp_path):
    original=frozen_input.read_bytes();manifest_value=json.loads(original)
    q,y,w,metadata=module.load_frozen_matches(manifest_value["rows"][0],tmp_path)
    assert len(q)==17 and metadata["raw_matches"]==18 and metadata["static_world_excluded"]==1 and metadata["zero_confidence_eligible"]==1
    np.testing.assert_allclose(y[:-1],q[:-1]@np.array([[.65,.07],[-.03,.6]]).T+[.12,.2],atol=2e-16)
    assert w[-1]==0
    output=tmp_path/"probe.json";result=module.run(frozen_input,output)
    assert result["probe_complete"] and result["pass"] and result["production_refits_eligible"]
    assert len(result["rows"])==3 and all(len(row["folds"])==2 for row in result["rows"])
    assert result["annotations_read"] is False and not result["maps_written"]
    assert result["previously_viewed_label_informed_development"] is True
    assert "NOT untouched/blind" in result["development_scope"]
    assert frozen_input.read_bytes()==original
    assert all(row["all_point_fits"][m]["status"]=="ok" for row in result["rows"] for m in ("similarity","affine"))
    for row in result["rows"]:
        assert sum(len(f["heldout_ids"]) for f in row["folds"])==17
        assert any(17 in f["heldout_ids"] for f in row["folds"])
        assert all(row["all_point_fits"][m]["training"]["count"]==17 for m in ("similarity","affine"))
    with pytest.raises(FileExistsError):module.run(frozen_input,output)


def test_bad_raw_provenance_or_affine_preserves_all_attempts(frozen_input,tmp_path):
    value=json.loads(frozen_input.read_text());path=tmp_path/value["rows"][0]["raw_matches"]["path"]
    raw=json.loads(path.read_text());raw["targets_manual_landmarks_or_dense_teacher_loaded"]=True;path.write_text(json.dumps(raw))
    result=module.run(frozen_input,tmp_path/"failed.json")
    assert result["probe_complete"] and not result["pass"] and len(result["rows"])==3
    assert result["rows"][0]["status"]=="failed" and all(f["models"][m]["status"]=="failed" for f in result["rows"][0]["folds"] for m in ("similarity","affine"))
    assert all(row["all_point_fits"] is None for row in result["rows"])
    raw["targets_manual_landmarks_or_dense_teacher_loaded"]=False;raw["post_affine_matrix"][0][0]+=.01;path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):module.load_frozen_matches(value["rows"][0],tmp_path)


def test_manifest_guard_before_any_match_read(frozen_input,tmp_path,monkeypatch):
    value=json.loads(frozen_input.read_text());value["prediction_complete"]=False;frozen_input.write_text(json.dumps(value))
    monkeypatch.setattr(module,"load_frozen_matches",lambda *a:pytest.fail("match loaded before manifest validated"))
    with pytest.raises(ValueError):module.run(frozen_input,tmp_path/"new.json")


def test_all_pending_folds_persisted_before_first_fit(frozen_input,tmp_path,monkeypatch):
    output=tmp_path/"probe.json";original=module.fit_robust_affine;seen=[]
    def fit(*args,**kwargs):
        if not seen:
            pending=json.loads(output.read_text())
            assert not pending["probe_complete"] and len(pending["rows"])==3
            assert all(f["models"][m]["status"]=="pending" for row in pending["rows"] for f in row["folds"] for m in ("similarity","affine"))
        seen.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,"fit_robust_affine",fit)
    result=module.run(frozen_input,output)
    assert result["probe_complete"] and len(seen)==18  # Twelve fold fits then six gated all-point fits.
