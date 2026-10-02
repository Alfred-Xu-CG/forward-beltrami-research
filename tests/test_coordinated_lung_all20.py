"""Known-specimen cohort stays fixed; no annotation enters prediction."""
import argparse
import importlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from tools.coordinated_real_case import optimize
from tools.digital_q1_dhr_distill import identity_vertices


def module():return importlib.import_module("tools.coordinated_lung_all20")


def assets(tmp_path):
    canvas=tmp_path/"canvas";canvas.mkdir()
    affines=tmp_path/"affines";affines.mkdir()
    stains=("he","cc10","cd31","ki67","prospc")
    raster=np.random.default_rng(31).integers(20,235,(16,16),dtype=np.uint8)
    for index,stain in enumerate(stains):
        Image.fromarray(np.roll(raster,index,axis=1)).save(canvas/(
            "cc10_fixed512.png" if stain=="he" else stain+"_moving512.png"))
    rows=[]
    for fixed,moving in itertools.permutations(stains,2):
        name=fixed+"_to_"+moving
        np.savez(affines/(name+"_affine.npz"),post_affine_matrix=np.eye(2,dtype=np.float32),
            post_affine_offset=np.zeros(2,dtype=np.float32),affine_source="direct_image_superglue")
        rows.append(dict(name=name,status="ok"))
    (affines/"affines.json").write_text(json.dumps(dict(rows=rows,
        landmarks_initial_DHR_affine_full_DHR_field_loaded=False)))
    return argparse.Namespace(canvas=canvas,affines_from=affines,output=tmp_path/"predictions",device="cpu",threads=1)


def fake_match(args):
    points=np.array(list(itertools.product((.2,.5,.8),repeat=2)))
    with np.load(args.affine,allow_pickle=False) as saved:
        matrix=saved["post_affine_matrix"];offset=saved["post_affine_offset"]
    record=dict(status="ok",raw_matches=len(points),fixed=str(args.fixed),moving=str(args.moving),
        image_side=args.image_side,source_points_unit=points.tolist(),target_points_unit=points.tolist(),
        confidence=[.8]*len(points),post_affine_matrix=matrix.tolist(),post_affine_offset=offset.tolist(),
        targets_manual_landmarks_or_dense_teacher_loaded=False,global_geometric_ransac_used=False,
        setup_seconds=.001,inference_seconds=.002,peak_allocated_bytes=None)
    args.output.write_text(json.dumps(record))
    return record


def fake_optimize(args):
    with np.load(args.affine,allow_pickle=False) as saved:
        matrix=saved["post_affine_matrix"];offset=saved["post_affine_offset"]
    vertices=identity_vertices(args.grid_side,device="cpu").double().numpy()
    np.savez(args.output,vertices=vertices,boundary_reference=vertices,post_affine_matrix=matrix,
        post_affine_offset=offset,interpolation="p1_ac")
    gradients=args.cycles*len(args.levels)*args.inner_steps*(2 if args.method=="analytic" else 1)
    report=dict(configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        gradient_steps=gradients,evaluations=gradients+args.cycles*len(args.levels)*(2 if args.method=="analytic" else 1),
        objective_evaluations=gradients+12,failed_trials=0,saved_binary_certificate=dict(valid=True),
        peak_allocated_bytes=None,initial=dict(total=.3),final=dict(total=.2),
        optimize_seconds=.003,loading_seconds=.001,feature_seconds=.001,serialization_seconds=.001,
        certification_seconds=.001,end_to_end_seconds=.01,landmarks_used=False)
    args.output.with_suffix(".json").write_text(json.dumps(report))
    return report


def fake_dhr(args):
    args.output.mkdir()
    final=args.output/"common_affine_dhr/Results_Final";final.mkdir(parents=True)
    (final/"displacement_field.mha").write_text("mock-heavy-call-output")
    (final/"postprocessing_params.json").write_text("{}")
    (args.output/"config.json").write_text("{}")
    with np.load(args.affine,allow_pickle=False) as saved:
        matrix=saved["post_affine_matrix"];offset=saved["post_affine_offset"]
    (args.output/"runtime.json").write_text(json.dumps(dict(runtime_seconds=.003,
        nonrigid_seconds=.002,preprocessing_seconds=.001,post_affine_matrix=matrix.tolist(),
        post_affine_offset=offset.tolist())))


def test_exact_production_recipe_and_equal_gradient_budget(tmp_path):
    runner=module();args=assets(tmp_path)
    pair=runner.pairs(args)[0]
    configs=[runner.make_configuration(pair,method,args) for method in ("analytic","f2")]
    for cfg in configs:
        assert cfg.affine==pair["affine"]
        assert cfg.grid_side==257 and cfg.image_side==512
        assert cfg.levels==[17,33,65,129,257] and cfg.image_levels==[32,64,128,256,512]
        assert (cfg.strain_model,cfg.strain_weight,cfg.shape_weight)==("p1_arap",3.,.0001)
        assert (cfg.match_weight,cfg.match_robust_scale,cfg.learning_rate)==(.1,8.,.004)
        assert (cfg.precision,cfg.image_precision,cfg.interpolation)==("float64","float32","p1_ac")
        assert cfg.lr_calibration=="edge" and cfg.output_selection=="best_full"
        assert cfg.control_hierarchy=="fixed" and cfg.nested_evaluation=="full_fine"
        assert cfg.inner_steps*cfg.cycles*len(cfg.levels)*(2 if cfg.method=="analytic" else 1)==300
    assert configs[0].cycles==1 and configs[1].cycles==2
    assert configs[0].geometry_backend=="stage_cache" and configs[1].geometry_backend=="existing"


def test_all20_raw_matching_precedes_predictions_and_cohort_is_written_upfront(tmp_path):
    runner=module();args=assets(tmp_path);events=[]
    def matcher(config):
        before=json.loads((args.output/"predictions.json").read_text())
        assert len(before["rows"])==20 and not before["prediction_complete"]
        events.append("match");return fake_match(config)
    def predictor(config):
        assert events[:20]==["match"]*20
        events.append(config.method);return fake_optimize(config)
    report=runner.run(args,production=False,match_extractor=matcher,optimizer=predictor,dhr_runner=fake_dhr)
    assert report["prediction_complete"] and report["cohort_size"]==20
    assert report["annotations_read"] is False
    assert len(report["rows"])==20 and len(events)==60
    assert all(row["status"]=="ok" for row in report["rows"])
    assert all(set(row["methods"])=={"analytic","f2","dhr"} for row in report["rows"])
    assert all(not Path(row["methods"]["analytic"]["output"]).is_absolute() for row in report["rows"])
    assert all(not Path(row["methods"]["dhr"]["field"]).is_absolute() for row in report["rows"])
    assert all(row["methods"]["dhr"]["geometry_certified"] is False for row in report["rows"])


def test_failures_remain_in_all20_and_native_dhr_does_not_require_matches(tmp_path):
    runner=module();args=assets(tmp_path)
    def matcher(config):
        if config.output.name.startswith("he_to_cc10_"):raise RuntimeError("deliberate matcher failure")
        return fake_match(config)
    def predictor(config):
        if config.output.name=="cc10_to_he_analytic.npz":raise RuntimeError("deliberate optimizer failure")
        return fake_optimize(config)
    report=runner.run(args,production=False,match_extractor=matcher,optimizer=predictor,dhr_runner=fake_dhr)
    rows={row["name"]:row for row in report["rows"]}
    assert report["prediction_complete"] and len(rows)==20
    assert rows["he_to_cc10"]["raw_matches"]["status"]=="failed"
    assert rows["he_to_cc10"]["methods"]["analytic"]["status"]=="skipped"
    assert rows["he_to_cc10"]["methods"]["dhr"]["status"]=="ok"
    assert rows["cc10_to_he"]["methods"]["analytic"]["status"]=="failed"
    assert rows["cc10_to_he"]["methods"]["f2"]["status"]=="ok"


def test_nonpositive_affine_is_retained_as_input_failure(tmp_path):
    runner=module();args=assets(tmp_path)
    np.savez(args.affines_from/"he_to_cc10_affine.npz",post_affine_matrix=np.diag([-1.,1.]).astype(np.float32),
        post_affine_offset=np.zeros(2,dtype=np.float32),affine_source="direct_image_superglue")
    report=runner.run(args,production=False,match_extractor=fake_match,optimizer=fake_optimize,dhr_runner=fake_dhr)
    row=report["rows"][0]
    assert row["input_status"]=="failed" and "positive" in row["input_error"]
    assert all(method["status"]=="skipped" for method in row["methods"].values())
    assert len(report["rows"])==20


def test_two_tiny_actual_predictions_keep_same_affine_and_actual_counts(tmp_path):
    runner=module();args=assets(tmp_path);args.output.mkdir()
    pair=runner.pairs(args)[0]
    for method in ("analytic","f2"):
        config=runner.make_configuration(pair,method,args,production=False)
        if not config.matches.exists():
            fake_match(argparse.Namespace(fixed=config.fixed,moving=config.moving,affine=config.affine,
                output=config.matches,image_side=config.image_side))
        result=optimize(config)
        assert result["gradient_steps"]==4 and result["evaluations"]==8
        assert result["failed_trials"]==0 and result["saved_binary_certificate"]["valid"]
        with np.load(config.output,allow_pickle=False) as saved, np.load(pair["affine"],allow_pickle=False) as affine:
            np.testing.assert_array_equal(saved["post_affine_matrix"],affine["post_affine_matrix"])
            np.testing.assert_array_equal(saved["post_affine_offset"],affine["post_affine_offset"])
        assert result["landmarks_used"] is False


def test_affine_provenance_failure_keeps_cohort_without_calling_any_method(tmp_path):
    runner=module();args=assets(tmp_path)
    path=args.affines_from/"affines.json"
    source=json.loads(path.read_text());source["landmarks_initial_DHR_affine_full_DHR_field_loaded"]=True
    path.write_text(json.dumps(source))
    def forbidden(*args,**kwargs):pytest.fail("unverified affine provenance must not reach prediction")
    report=runner.run(args,production=False,match_extractor=forbidden,optimizer=forbidden,dhr_runner=forbidden)
    assert report["prediction_complete"] and len(report["rows"])==20
    assert all(row["input_status"]=="failed" for row in report["rows"])
    assert report["method_success_counts"]==dict(analytic=0,f2=0,dhr=0)


@pytest.mark.parametrize("field,value",[("global_geometric_ransac_used",True),
    ("targets_manual_landmarks_or_dense_teacher_loaded",True)])
def test_invalid_raw_match_provenance_skips_safe_methods_but_not_native(tmp_path,field,value):
    runner=module();args=assets(tmp_path)
    def matcher(config):
        record=fake_match(config);record[field]=value
        config.output.write_text(json.dumps(record));return record
    def forbidden(*args,**kwargs):pytest.fail("invalid match provenance must not reach safe optimizer")
    report=runner.run(args,production=False,match_extractor=matcher,optimizer=forbidden,dhr_runner=fake_dhr)
    assert report["successful_raw_match_directions"]==0
    assert report["method_success_counts"]==dict(analytic=0,f2=0,dhr=20)
    assert all(row["raw_matches"]["status"]=="failed" for row in report["rows"])


def test_saved_affine_mismatch_is_a_method_failure_not_silently_success(tmp_path):
    runner=module();args=assets(tmp_path)
    def predictor(config):
        result=fake_optimize(config)
        with np.load(config.output,allow_pickle=False) as saved:
            payload={key:saved[key].copy() for key in saved.files}
        payload["post_affine_offset"][0]+=.01
        np.savez(config.output,**payload)
        return result
    report=runner.run(args,production=False,match_extractor=fake_match,optimizer=predictor,dhr_runner=fake_dhr)
    assert report["method_success_counts"]==dict(analytic=0,f2=0,dhr=20)
    assert all("common affine" in row["methods"]["analytic"]["error"] for row in report["rows"])


def test_occupied_output_refused_without_overwriting(tmp_path):
    runner=module();args=assets(tmp_path);args.output.mkdir()
    path=args.output/"predictions.json";path.write_text("preserve")
    with pytest.raises(FileExistsError):runner.run(args,production=False)
    assert path.read_text()=="preserve"
