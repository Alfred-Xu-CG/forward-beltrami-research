"""Paired phase/budget/export routing; no joint-optimizer implementation oracle."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import coordinated_joint_pose_batch as batch
from tests.test_coordinated_match_fusion import fixtures, original


def test_configs_keep_fusion_evidence_and_disclose_distinct_budgets(tmp_path):
    source=original(tmp_path);source["match_weight"]=.2;saved=copy.deepcopy(source)
    for arm in ("frozen250","joint300"):
        cfg=vars(batch.configuration(source,arm,"new.npz"))
        norm=lambda k,v:str(Path(v)) if k in ("fixed","moving","affine","matches","output") else v
        delta={k for k,v in cfg.items() if norm(k,v)!=norm(k,source.get(k))}
        expected={"output","inner_steps","inner_steps_by_level","pose_mode"}
        if arm=="joint300":expected.add("pose_steps_per_level")
        assert delta==expected and cfg["match_weight"]==.2
        assert cfg["inner_steps"]==25 and cfg["inner_steps_by_level"]==[25]*5
        assert cfg["pose_mode"]==("joint_positive_affine" if arm=="joint300" else "frozen_affine")
    assert source==saved
    with pytest.raises(ValueError):batch.configuration({**source,"match_weight":.1},"joint300","new.npz")


def test_all50_terminal_and_separate_export_validation(tmp_path,monkeypatch):
    cases,_=fixtures(tmp_path)
    for name in ("fixed.png","moving.png"):(tmp_path/name).touch()
    for case in cases:
        cfg=case["original_configuration"]
        cfg.update(match_weight=.2,fixed=str(tmp_path/"fixed.png"),moving=str(tmp_path/"moving.png"))
    monkeypatch.setattr(batch,"source_cases",lambda *_:copy.deepcopy(cases))
    output=tmp_path/"run";calls=[];validations=[]
    def check(cfg,arm):
        for name in batch.ARMS:
            assert not (output/name/"miit_predictions.json").exists()
            assert not json.loads((output/name/"predictions.json").read_text())["prediction_complete"]
        calls.append((arm,cfg.output.name))
        assert cfg.inner_steps==25 and cfg.match_weight==.2
    def frozen(cfg):
        check(cfg,"frozen250")
        if cfg.output.name.startswith(cases[0]["name"]):raise ValueError("first frozen failure")
        return dict(gradient_steps=250,failed_trials=0)
    def joint(cfg):
        check(cfg,"joint300")
        if cfg.output.name.startswith(cases[-1]["name"]):return dict(gradient_steps=300,failed_trials=0,residual_gradient_steps=249,pose_gradient_steps=51)
        return dict(gradient_steps=300,failed_trials=0,residual_gradient_steps=250,pose_gradient_steps=50)
    def frozen_validator(cfg,result):
        validations.append("frozen")
        assert result["gradient_steps"]==250
        return dict(valid=True,original_affine_normalized_minimum_corner_ratio=.2)
    def joint_validator(path,minimum_jacobian):
        validations.append("joint")
        assert isinstance(path,Path) and minimum_jacobian==.001
        return dict(valid=True,original_affine_normalized_minimum_corner_ratio=.15)
    result=batch.run("unused",output,frozen_optimizer=frozen,joint_optimizer=joint,
        frozen_validator=frozen_validator,joint_validator=joint_validator)
    assert len(calls)==50 and validations.count("frozen")==validations.count("joint")==24
    assert result["prediction_complete"]
    for arm,budget in batch.ARMS.items():
        assert result["arms"][arm]["failed"]==1
        manifest=json.loads((output/arm/"predictions.json").read_text())
        assert manifest["all50_terminal"] and manifest["expected_gradient_steps"]==budget
        for name in ("miit","existing"):
            subgroup=json.loads((output/arm/(name+"_predictions.json")).read_text())
            assert subgroup["all50_terminal"] and subgroup["expected_gradient_steps"]==budget


@pytest.mark.parametrize("budget",[0,-1,1.5,True])
def test_scorer_rejects_invalid_budget_before_reading_files(budget):
    from tools.coordinated_existing_shared_frame_score import score
    with pytest.raises(ValueError,match="positive integer"):score("does-not-exist","unused",budget)


def test_scorer_default300_preserved_explicit250_scores(tmp_path):
    from tests.test_coordinated_dhr_existing_score import _complete_fixture
    from tools.coordinated_existing_shared_frame_score import score
    path,root,value=_complete_fixture(tmp_path)
    y,x=np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3),indexing="ij")
    identity=np.stack((x,y),-1)[None]
    np.savez(tmp_path/"map.npz",vertices=identity,boundary_reference=identity,
        post_affine_matrix=np.eye(2),post_affine_offset=np.zeros(2),interpolation=np.asarray("p1_ac"))
    value.update(cohort_size=22)
    for row in value["rows"]:
        row.update(output="map.npz",configuration=dict(mind_frame="shared_affine"),gradient_steps=300,failed_trials=0)
    path.write_text(json.dumps(value))
    assert score(path,root)["scored_pairs"]==22
    for row in value["rows"]:row["gradient_steps"]=250
    path.write_text(json.dumps(value))
    default=score(path,root)
    assert default["scored_pairs"]==0 and default["expected_gradient_steps"]==300
    explicit=score(path,root,expected_gradient_steps=250)
    assert explicit["scored_pairs"]==22 and explicit["expected_gradient_steps"]==250
    assert explicit["equal_specimen_canvas"]["mean_pair_mean"]<1e-12
