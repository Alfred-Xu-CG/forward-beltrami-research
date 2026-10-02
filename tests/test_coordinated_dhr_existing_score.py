import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.coordinated_dhr_existing_inputs import existing_rows, STAINS, PREFIX
from tools.coordinated_dhr_existing_score import native_metrics, score


def test_rectangular_native_axes_padding_resampling_and_canvas():
    # Source native frame 30x22, target 28x20; padded frame is 30x22.
    # Target pads x=1,y=1; source pads zero. At preprocessing ratio2,
    # displacement (1,-.5) means native displacement (2,-1).
    field = np.zeros((2, 11, 15), np.float32)
    field[0] = 1
    field[1] = -.5
    params = dict(source_resample_ratio=1., target_resample_ratio=1.,
                  initial_resample_ratio=2., pad_1=[[0, 0], [0, 0]], pad_2=[[1, 1], [1, 1]])
    fixed = np.array([[7., 8.], [17., 12.]])
    target = fixed + [3., 0.]
    layout = dict(original_wh=[30, 22], effective_original_to_canvas_scale_xy=[2., 3.], padding_xy=[5, 7])
    result = native_metrics(field, params, fixed, target, (28, 20), (30, 22), layout, ["1", "2"])
    assert result["native_moving_pixels"]["mean"] < 3e-6
    assert result["canvas_pixels"]["mean"] < 8e-6
    result = native_metrics(field, params, fixed, target+[1., 0.], (28, 20), (30, 22), layout, ["1", "2"])
    assert result["native_moving_pixels"]["mean"] == pytest.approx(1., abs=3e-6)
    assert result["canvas_pixels"]["mean"] == pytest.approx(2., abs=8e-6)


def manifest(root):
    rows = existing_rows(root)
    for row in rows:
        row.update(status="failed", error="deliberate retained failure")
    return dict(prediction_complete=True, annotations_read=False, pair_denominator=22,
                preset="default_initial_nonrigid", rows=rows)


@pytest.mark.parametrize("change", ["unfinished", "pending", "missing", "reversed", "wrong_preset"])
def test_all22_terminal_manifest_required_before_labels(tmp_path, monkeypatch, change):
    value = manifest(tmp_path / "absent_images")
    if change == "unfinished": value["prediction_complete"] = False
    if change == "pending": value["rows"][-1]["status"] = "pending"
    if change == "missing": value["rows"].pop()
    if change == "reversed": value["rows"][0]["fixed"], value["rows"][0]["moving"] = value["rows"][0]["moving"], value["rows"][0]["fixed"]
    if change == "wrong_preset": value["preset"] = "default_initial_nonrigid_fast"
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): score(path, tmp_path / "absent_images")


def test_all_prediction_failures_remain_no_all22_average_no_labels(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(manifest(tmp_path / "absent_images")))
    result = score(path, tmp_path / "absent_images")
    assert result["pair_denominator"] == 22 and result["scored_pairs"] == 0
    assert len(result["rows"]) == 22
    assert result["lung_all20"]["all_directions"] is None
    assert result["equal_specimen_canvas"] is None


def test_score_failure_retained_when_native_export_absent(tmp_path):
    value = manifest(tmp_path / "absent_images")
    value["rows"][0].update(status="ok", field="missing.mha", postprocessing_params="missing.json")
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(value))
    result = score(path, tmp_path / "absent_images")
    assert result["rows"][0]["status"] == "failed"
    assert result["lung_all20"]["all_directions"] is None


def _complete_fixture(tmp_path):
    import SimpleITK as sitk
    root = tmp_path / "data"
    value = manifest(root)
    def write_json(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    def labels(path, count, scale=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [" ,X,Y"]
        for k in range(1, count+1):
            point = np.array([5.+k%20, 4.+k%10])
            point = (point+.5)*scale-.5
            rows.append(f"{k},{point[0]},{point[1]}")
        path.write_text("\n".join(rows)+"\n")
    for image in {r[role] for r in value["rows"] for role in ("fixed", "moving")}:
        path = Path(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 20)).save(path)
    def layout(source):
        return dict(source=source, original_wh=[32,20], resized_wh=[256,160],
                    effective_original_to_canvas_scale_xy=[8.,8.], padding_xy=[128,176])
    for stain in STAINS:
        if stain == "he": continue
        case = next(r for r in value["rows"] if r["name"] == f"he_to_{stain}")
        write_json(root/f"lung_lesion3_eval/canvas/{stain}_layout.json",
                   dict(side=512, fixed=layout(case["fixed"]), moving=layout(case["moving"])))
    for stain in STAINS:
        labels(root/f"lung_lesion3_eval/annotations50/{PREFIX}{STAINS[stain]}-les3.csv", 80, 10)
    for row in value["rows"][-2:]:
        write_json(root/f"birl_anhir_dev/canvas/{row['name']}_layout.json",
                   dict(side=512, fixed=layout(row["fixed"]), moving=layout(row["moving"])))
    for stain in ("CD4", "CD68"):
        labels(root/f"HistoReg_CD68_CD4/Landmarks_{stain}.csv", 77)
    for stain,count in (("HE",71), ("PanCytokeratin",69)):
        labels(root/f"birl_anhir_dev/labels_eval_only/rat-kidney_/scale-5pc/Rat-Kidney_{stain}.csv", count)
    field = np.zeros((2,20,32), np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(field, isVector=False), str(tmp_path/"field.mha"))
    write_json(tmp_path/"params.json", dict(source_resample_ratio=1., target_resample_ratio=1.,
               initial_resample_ratio=1., pad_1=[[0,0],[0,0]], pad_2=[[0,0],[0,0]]))
    for row in value["rows"]:
        row.update(status="ok", field="field.mha", postprocessing_params="params.json",
                   fixed_original_wh=[32,20], moving_original_wh=[32,20])
    path = tmp_path / "predictions.json"
    write_json(path, value)
    return path, root, value


def test_all22_native_end_to_end_preserves_label_conversion_and_denominators(tmp_path):
    path, root, _ = _complete_fixture(tmp_path)
    result = score(path, root)
    assert result["scored_pairs"] == 22
    assert [r["scored_landmarks"] for r in result["rows"]] == [80]*20+[77,69]
    assert result["rows"][-1]["fixed_only_ids"] == ["70", "71"]
    assert result["equal_specimen_canvas"]["mean_pair_mean"] < 1e-5
    assert all(not r["topology"]["global_homeomorphism_certified"] for r in result["rows"])
    assert result["rows"][0]["field_scalars"]["maximum_displacement_norm"] == 0


def test_native_size_mismatch_fails_one_direction_without_dropping_it(tmp_path):
    path, root, value = _complete_fixture(tmp_path)
    value["rows"][0]["fixed_original_wh"] = [20,32]
    path.write_text(json.dumps(value))
    result = score(path, root)
    assert result["scored_pairs"] == 21
    assert result["rows"][0]["status"] == "failed"
    assert result["lung_all20"]["all_directions"] is None
    assert result["equal_specimen_canvas"] is None


def test_initial_only_same22_dispatch_never_reads_dense_fields(tmp_path,monkeypatch):
    import SimpleITK as sitk
    from tools.coordinated_dhr_existing_score import score_initial
    path,root,value=_complete_fixture(tmp_path)
    for row in value["rows"]:
        row.update(initial_transform=[[[1.,0.,0.],[0.,1.,0.]]],preprocessed_shape=[1,1,20,32],
                   initial_transform_frame="preprocessed normalized [-1,1] target-to-source affine; align_corners=False")
    path.write_text(json.dumps(value))
    monkeypatch.setattr(sitk,"ReadImage",lambda *a,**k:pytest.fail("initial-only must not read fields"))
    result=score_initial(path,root)
    assert result["scored_pairs"]==22
    assert result["equal_specimen_canvas"]["mean_pair_mean"]<1e-12
    assert result["final_dense_fields_read"] is False
    value["rows"][0]["initial_transform_frame"]="ambiguous"
    path.write_text(json.dumps(value))
    result=score_initial(path,root)
    assert result["scored_pairs"]==21 and result["equal_specimen_canvas"] is None
