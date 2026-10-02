import copy
import json
from pathlib import Path

import pytest

from tools.coordinated_existing_shared_frame import configuration, source_cases


def recipe():
    return dict(method="analytic", loss="mind", output_selection="best_full", grid_side=257,
                image_side=512, image_levels=[32,64,128,256,512], levels=[17,33,65,129,257],
                inner_steps=30, cycles=1, strain_weight=3., strain_model="p1_arap", shape_weight=1e-4,
                match_weight=.1, fixed="data/f.png", moving="data/m.png", affine="data/a.npz",
                matches="results/raw.json", output="results/old.npz", device="cuda", threads=2,
                precision="float64", image_precision="float32", interpolation="p1_ac")


def test_configuration_changes_only_frame_and_output():
    original = recipe()
    frozen = copy.deepcopy(original)
    result = vars(configuration(original, Path("new.npz")))
    assert original == frozen
    plain = {k: v.as_posix() if isinstance(v,Path) else v for k,v in result.items()}
    assert {k:v for k,v in plain.items() if k not in ("mind_frame","output")} == {k:v for k,v in original.items() if k != "output"}
    assert plain["mind_frame"] == "shared_affine"
    assert "preconditioner" not in plain and "coupled_seed" not in plain


@pytest.mark.parametrize("key,value", [("cycles",2), ("method","f2"), ("mind_frame","shared_affine"),
    ("strain_weight",.05), ("image_objective","simultaneous_multiscale"), ("capture_prefix","mind_discrete")])
def test_no_other_experimental_recipe_is_accepted(key,value):
    source = recipe(); source[key]=value
    with pytest.raises(ValueError): configuration(source, Path("new.npz"))


def test_original22_order_and_archived_relative_analytic_reports(tmp_path):
    from tools.coordinated_dhr_existing_inputs import existing_rows
    rows=[]
    originals=tmp_path/"originals"; originals.mkdir()
    cohort=tmp_path/"cohort"; cohort.mkdir()
    for row in existing_rows(tmp_path)[:20]:
        report=originals/(row["name"]+".json")
        report.write_text(json.dumps(dict(configuration=recipe(),gradient_steps=300,failed_trials=0)))
        rows.append(dict(name=row["name"],methods=dict(analytic=dict(status="ok",report="../originals/"+report.name))))
    (cohort/"predictions.json").write_text(json.dumps(dict(prediction_complete=True,annotations_read=False,cohort_size=20,rows=rows)))
    development=tmp_path/"development"; development.mkdir()
    for case in ("histo","rat_kidney"):
        (development/(case+"_analytic_mind_edge257.json")).write_text(json.dumps(dict(configuration=recipe(),gradient_steps=300,failed_trials=0)))
    cases=source_cases(cohort/"predictions.json",development)
    assert len(cases)==22
    assert [r["name"] for r in cases]==[r["name"] for r in existing_rows(tmp_path)]
    assert cases[0]["source_report"]==str((originals/"he_to_cc10.json").resolve())


def test_shared_frame_scorer_all22_end_to_end_and_terminal_gate(tmp_path):
    import numpy as np
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
    value["prediction_complete"]=False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="terminal"):score(path,root)
    value["prediction_complete"]=True
    path.write_text(json.dumps(value))
    result=score(path,root)
    assert result["scored_pairs"]==22
    assert result["equal_specimen_canvas"]["mean_pair_mean"]<1e-12
    assert [r["scored_landmarks"] for r in result["rows"]]==[80]*20+[77,69]
    assert result["rows"][-1]["fixed_only_ids"]==["70","71"]
