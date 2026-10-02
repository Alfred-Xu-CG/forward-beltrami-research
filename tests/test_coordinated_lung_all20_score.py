"""Independent frames and failure-denominator fixtures, no real annotations."""
import csv
import itertools
import json
from pathlib import Path

import torch
import numpy as np
import pytest

from tools import coordinated_lung_all20_score as scorer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def identity():
    y, x = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij")
    return np.stack((x, y), -1)[None]


@pytest.fixture
def cohort(tmp_path):
    canvas, labels, predictions = (tmp_path / s for s in ("canvas", "labels", "predictions"))
    for directory in (canvas, labels, predictions):
        directory.mkdir()
    # Independent nominal50->5pc coordinates; anisotropic scales and nonzero pad.
    layout = dict(original_wh=[1000, 800], resized_wh=[400, 320], padding_xy=[50, 70],
                  effective_original_to_canvas_scale_xy=[.4, .4])
    for stain in scorer.STAIN_NAME:
        if stain != "he":
            write_json(canvas / f"{stain}_layout.json", dict(side=512, fixed=layout, moving=layout))
        filename = labels / f"{scorer.PREFIX}{scorer.STAIN_NAME[stain]}-les3.csv"
        with filename.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([" ", "X", "Y"])
            for k in range(80):
                native = np.array([100 + 7*(k % 10), 100 + 7*(k // 10)], dtype=float)
                original50 = 10*(native + .5) - .5
                writer.writerow([str(k+1), *original50])
    rows = []
    for fixed, moving in scorer.PAIRS:
        name = f"{fixed}_to_{moving}"
        affine = f"{name}_affine.npz"
        a, b = np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
        np.savez(predictions / affine, post_affine_matrix=a, post_affine_offset=b)
        methods = {}
        for method in ("analytic", "f2"):
            archive, report = f"{name}_{method}.npz", f"{name}_{method}.json"
            v = identity()
            np.savez(predictions / archive, vertices=v, boundary_reference=v,
                     post_affine_matrix=a, post_affine_offset=b, interpolation=np.asarray("p1_ac"))
            write_json(predictions / report, dict(configuration=dict(interpolation="p1_ac"),
                                                  landmarks_used=False, gradient_steps=300))
            methods[method] = dict(status="ok", output=archive, report=report)
        methods["dhr"] = dict(status="failed", error="predeclared synthetic missing native export")
        rows.append(dict(name=name, fixed_stain=fixed, moving_stain=moving,
                         affine=affine, input_status="ok", raw_matches=dict(status="ok"), methods=methods))
    manifest = dict(prediction_complete=True, cohort_size=20, annotations_read=False, rows=rows)
    write_json(predictions / "predictions.json", manifest)
    return canvas, labels, predictions, manifest


def test_annotation50_to5_pixel_center_conversion(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text(" ,X,Y\n1,49.5,99.5\n2,0,0\n", encoding="utf-8")
    points = scorer.scaled_landmarks(path)
    np.testing.assert_array_equal(points["1"], [4.5, 9.5])
    np.testing.assert_array_equal(points["2"], [-.45, -.45])


def test_half_pixel_letterbox_inverse_and_padding():
    layout = dict(effective_original_to_canvas_scale_xy=[.5, .25], padding_xy=[6, 10])
    pixels = np.array([[-.5, -.5], [0, 0], [199.5, 99.5], [13.25, 4.75]])
    expected = (pixels + .5)*[.5, .25] + [6, 10]
    unit = scorer.original_pixel_to_canvas_unit(pixels, layout, 512)
    np.testing.assert_array_equal(unit*512, expected)
    np.testing.assert_allclose(scorer.canvas_unit_to_original_pixel(unit, layout, 512), pixels, atol=0, rtol=0)


def test_independent_literal_ac_nonaffine_quadrilateral():
    # a,b,c,d are intentionally NOT a parallelogram. Query upper/right triangle
    # and lower/left triangle plus diagonal and boundary endpoint.
    a, b, c, d = np.array([[.1, .2], [1.1, .1], [1.3, 1.2], [0, 1.]])
    v = np.array([[a, b], [d, c]])
    query = np.array([[.8, .3], [.2, .7], [.4, .4], [1, 1], [0, .3]])
    expected = np.stack([.2*a+.5*b+.3*c, .3*a+.2*c+.5*d,
                         .6*a+.4*c, c, .7*a+.3*d])
    np.testing.assert_allclose(scorer.p1_at_queries_numpy(v, query, "ac"), expected, atol=3e-16, rtol=0)


def test_native_saved_field_pixel_units_and_half_pixel():
    params = dict(target_resample_ratio=1, source_resample_ratio=1,
                  initial_resample_ratio=1, pad_1=[[0, 0], [0, 0]], pad_2=[[0, 0], [0, 0]])
    field = np.zeros((2, 512, 512), np.float32)
    field[0] = 8; field[1] = -4
    query = torch.tensor([[[[.125, .25], [.5/512, .5/512], [511.5/512, 511.5/512]]]])
    mapped = scorer.dhr_map_at_unit_queries(field, params, fixed_size=(512, 512),
                                           moving_size=(512, 512), query=query)
    # Independently: f=(512q-.5); g=f+saved_pixel_delta; unit=(g+.5)/512.
    torch.testing.assert_close(mapped, query+torch.tensor([8/512, -4/512]), atol=0, rtol=0)
    np.testing.assert_array_equal(scorer.q1_corner_determinants(field), np.ones((4, 511, 511)))


def test_native_nonconstant_linear_field_not_normalized_displacement():
    y, x = np.indices((8, 9), dtype=np.float32)
    field = np.stack((.25*x+.5*y, -.125*x+.25*y))
    params = dict(target_resample_ratio=1, source_resample_ratio=1,
                  initial_resample_ratio=1, pad_1=[[0, 0], [0, 0]], pad_2=[[0, 0], [0, 0]])
    xy = np.array([[2.25, 3.75], [4.5, 2.5]], dtype=np.float32)
    query = torch.tensor((xy+.5)/[9, 8], dtype=torch.float32).reshape(1, 1, 2, 2)
    mapped = scorer.dhr_map_at_unit_queries(field, params, fixed_size=(9, 8), moving_size=(9, 8), query=query)
    delta = np.stack((.25*xy[:, 0]+.5*xy[:, 1], -.125*xy[:, 0]+.25*xy[:, 1]), -1)
    np.testing.assert_allclose(mapped[0, 0].numpy(), (xy+delta+.5)/[9, 8], atol=1e-7, rtol=0)
    expected_det = 1.25*1.25 - .5*(-.125)
    np.testing.assert_allclose(scorer.q1_corner_determinants(field), expected_det, atol=0, rtol=0)


def test_all20_identity_and_failures_preserved(cohort):
    canvas, labels, predictions, _ = cohort
    report = scorer.score(canvas, labels, predictions)
    assert len(report["rows"]) == 20 and report["specimen_count"] == 1
    assert report["direction_denominator"] == 20
    for row in report["rows"]:
        assert row["landmark_count"] == 80
        for method in ("analytic", "f2", "common_affine"):
            item = row["methods"][method]
            assert item["status"] == "ok"
            assert item["metrics"]["canvas_pixels"]["max"] < 1e-13
            assert item["metrics"]["original_moving_5pc_pixels"]["max"] < 3e-13
    for method in ("analytic", "f2", "common_affine"):
        assert report["aggregates"][method]["scored_directions"] == 20
        assert report["aggregates"][method]["all20_equal_direction_metrics"] is not None
    assert report["aggregates"]["dhr"]["failed_or_skipped_directions"] == 20
    assert report["aggregates"]["dhr"]["all20_equal_direction_metrics"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change", ["incomplete", "annotation", "duplicate", "absent", "pending", "raw_pending", "reorder", "bad_stain"])
def test_manifest_rejected_before_any_label_open(cohort, monkeypatch, change):
    canvas, labels, predictions, manifest = cohort
    if change == "incomplete": manifest["prediction_complete"] = False
    elif change == "annotation": manifest["annotations_read"] = True
    elif change == "duplicate": manifest["rows"][-1] = manifest["rows"][0]
    elif change == "absent": manifest["rows"].pop()
    elif change == "pending": manifest["rows"][0]["methods"]["analytic"]["status"] = "pending"
    elif change == "raw_pending": manifest["rows"][0]["raw_matches"]["status"] = "pending"
    elif change == "reorder": manifest["rows"].reverse()
    else: manifest["rows"][0]["moving_stain"] = "unknown"
    write_json(predictions / "predictions.json", manifest)
    def forbidden(*args): raise AssertionError("labels opened before completed cohort validation")
    monkeypatch.setattr(scorer, "scaled_landmarks", forbidden)
    with pytest.raises(ValueError): scorer.score(canvas, labels, predictions)


def test_missing_manifest_before_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "scaled_landmarks", lambda *args: pytest.fail("labels read"))
    with pytest.raises(FileNotFoundError): scorer.score(tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize("bad", ["missing_map", "affine", "interpolation", "fold", "boundary", "report", "nonfinite"])
def test_invalid_ok_export_keeps_denominator(cohort, bad):
    canvas, labels, predictions, manifest = cohort
    method = manifest["rows"][0]["methods"]["analytic"]
    path = predictions / method["output"]
    if bad == "missing_map": path.unlink()
    elif bad == "report":
        write_json(predictions / method["report"], dict(configuration=dict(interpolation="p1_ac"), landmarks_used=True))
    else:
        with np.load(path) as archive: data = {key: archive[key] for key in archive.files}
        if bad == "affine": data["post_affine_offset"] = np.array([.01, 0], dtype=np.float32)
        elif bad == "interpolation": data["interpolation"] = np.asarray("q1")
        elif bad == "fold": data["vertices"][0, 1, 1] = [2, 2]
        elif bad == "boundary": data["vertices"][0, 0, 1, 0] += .1
        elif bad == "nonfinite": data["vertices"][0, 1, 1, 0] = np.nan
        np.savez(path, **data)
    report = scorer.score(canvas, labels, predictions)
    aggregate = report["aggregates"]["analytic"]
    assert aggregate["direction_denominator"] == 20 and aggregate["scored_directions"] == 19
    assert aggregate["all20_equal_direction_metrics"] is None
    assert aggregate["successful_directions_only"]["canvas_pixels"] is not None
    assert aggregate["unscored_worse_than_affine_directions"] == 1
    assert report["rows"][0]["methods"]["analytic"]["status"] == "failed"
    assert report["rows"][0]["methods"]["f2"]["status"] == "ok"


def test_equal_direction_not_patient_or_landmark_pooling():
    rows = []
    for k in range(20):
        metrics = {unit: dict(mean=float(k), p90=float(k+1), max=float(2*k))
                   for unit in ("canvas_pixels", "original_moving_5pc_pixels")}
        affine = {unit: dict(mean=10.) for unit in metrics}
        rows.append(dict(methods=dict(analytic=dict(status="ok", metrics=metrics),
                                      common_affine=dict(status="ok", metrics=affine))))
    aggregate = scorer._aggregate(rows, "analytic")
    summary = aggregate["all20_equal_direction_metrics"]["canvas_pixels"]
    assert summary["mean_of_direction_means"] == 9.5
    assert summary["p90_of_direction_means"] == pytest.approx(17.1)
    assert summary["max_of_direction_means"] == 19
    assert summary["mean_of_direction_p90"] == 10.5
    assert summary["maximum_direction_landmark_error"] == 38
    assert aggregate["worse_than_common_affine_count"] == 9


def test_missing_or_duplicate_landmark_ids_not_dropped(cohort):
    canvas, labels, predictions, _ = cohort
    filename = labels / f"{scorer.PREFIX}{scorer.STAIN_NAME['ki67']}-les3.csv"
    text = filename.read_text()
    filename.write_text("\n".join(text.splitlines()[:-1])+"\n")
    with pytest.raises(ValueError, match="exact same eighty"):
        scorer.score(canvas, labels, predictions)
    filename.write_text(text+text.splitlines()[1]+"\n")
    with pytest.raises(ValueError, match="duplicate"):
        scorer.score(canvas, labels, predictions)


def test_native_export_provenance_and_signs(cohort):
    sitk = pytest.importorskip("SimpleITK")
    canvas, labels, predictions, manifest = cohort
    native = predictions / "native"
    native.mkdir()
    field = np.zeros((2, 512, 512), dtype=np.float32)
    field[0] = 8; field[1] = -4
    sitk.WriteImage(sitk.GetImageFromArray(field), str(native / "field.mha"))
    write_json(native / "params.json", dict(target_resample_ratio=1, source_resample_ratio=1,
        initial_resample_ratio=1, pad_1=[[0, 0], [0, 0]], pad_2=[[0, 0], [0, 0]]))
    write_json(native / "runtime.json", dict(image_side=512, initial_resample_ratio=1,
        post_affine_matrix=np.eye(2).tolist(), post_affine_offset=[0, 0], runtime_seconds=.5))
    write_json(native / "config.json", dict(loading_params=dict(loader="pil", source_resample_ratio=1,
                                                               target_resample_ratio=1)))
    method = dict(status="ok", field="native/field.mha", postprocessing_params="native/params.json",
                  report="native/runtime.json", configuration="native/config.json")
    manifest["rows"][0]["methods"]["dhr"] = method
    write_json(predictions / "predictions.json", manifest)
    report = scorer.score(canvas, labels, predictions)
    scored = report["rows"][0]["methods"]["dhr"]
    assert scored["status"] == "ok"
    assert scored["geometry"]["minimum_corner_determinant"] == 1
    assert scored["geometry"]["nonpositive_corner_count"] == 0
    assert scored["metrics"]["canvas_pixels"]["mean"] == pytest.approx(np.sqrt(80), abs=4e-5)
    assert scored["metrics"]["original_moving_5pc_pixels"]["mean"] == pytest.approx(np.sqrt(80)/.4, abs=1e-4)
    # No native affine mismatch is scored, no method fallback.
    runtime = scorer._json(native / "runtime.json")
    runtime["post_affine_offset"] = [1, 0]
    write_json(native / "runtime.json", runtime)
    report = scorer.score(canvas, labels, predictions)
    assert report["rows"][0]["methods"]["dhr"]["status"] == "failed"
    assert report["aggregates"]["dhr"]["scored_directions"] == 0


def test_affine_override_is_read_only_and_requires_export_equality(cohort, tmp_path):
    canvas, labels, predictions, manifest = cohort
    overrides = tmp_path / "affines"
    overrides.mkdir()
    for row in manifest["rows"]:
        (overrides / row["affine"]).write_bytes((predictions / row["affine"]).read_bytes())
        row["affine"] = "/remote/no-longer-mounted/"+row["affine"]
    write_json(predictions / "predictions.json", manifest)
    before = (predictions / "predictions.json").read_bytes()
    report = scorer.score(canvas, labels, predictions, affines_from=overrides)
    assert report["aggregates"]["analytic"]["scored_directions"] == 20
    assert (predictions / "predictions.json").read_bytes() == before
    np.savez(overrides / "he_to_cc10_affine.npz", post_affine_matrix=np.eye(2, dtype=np.float32),
             post_affine_offset=np.array([.1, 0], dtype=np.float32))
    report = scorer.score(canvas, labels, predictions, affines_from=overrides)
    assert report["rows"][0]["methods"]["analytic"]["status"] == "failed"
    assert report["rows"][0]["methods"]["common_affine"]["status"] == "ok"


@pytest.mark.parametrize("bad", ["missing", "absent_key", "nonpositive"])
def test_missing_or_invalid_affine_keeps_all_direction_failures(cohort, bad):
    canvas, labels, predictions, manifest = cohort
    row = manifest["rows"][0]
    path = predictions / row["affine"]
    if bad == "missing": path.unlink()
    elif bad == "absent_key":
        del row["affine"]
        write_json(predictions / "predictions.json", manifest)
    else:
        np.savez(path, post_affine_matrix=np.diag([-1., 1.]), post_affine_offset=np.zeros(2))
    report = scorer.score(canvas, labels, predictions)
    assert len(report["rows"]) == 20
    for method in ("common_affine", "analytic", "f2"):
        assert report["rows"][0]["methods"][method]["status"] == "failed"
        assert report["aggregates"][method]["scored_directions"] == 19
        assert report["aggregates"][method]["direction_denominator"] == 20


def test_native_folds_reported_not_scoring_exclusion(cohort, monkeypatch):
    canvas, labels, predictions, manifest = cohort
    row = manifest["rows"][0]
    row["methods"]["dhr"] = dict(status="ok")
    write_json(predictions / "predictions.json", manifest)
    y, x = np.indices((512, 512), dtype=np.float32)
    # Saved mapping fx=511-x, fy=y: local determinant -1 everywhere.
    field = np.stack((511-2*x, np.zeros_like(y)))
    corners = scorer.q1_corner_determinants(field)
    np.testing.assert_array_equal(corners, -np.ones((4, 511, 511)))
    params = dict(target_resample_ratio=1, source_resample_ratio=1, initial_resample_ratio=1,
                  pad_1=[[0, 0], [0, 0]], pad_2=[[0, 0], [0, 0]])
    def load_native(*args):
        return field, params, dict(nonpositive_corner_count=int((corners <= 0).sum()),
            cells_with_nonpositive_corner=int((corners <= 0).any(0).sum()),
            geometry_scope="local signs, no global certificate")
    monkeypatch.setattr(scorer, "_native", load_native)
    report = scorer.score(canvas, labels, predictions)
    item = report["rows"][0]["methods"]["dhr"]
    assert item["status"] == "ok" and item["geometry"]["nonpositive_corner_count"] == 4*511**2
    assert len(item["metrics"]["canvas_pixels"]["per_landmark"]) == 80
    assert report["aggregates"]["dhr"]["scored_directions"] == 1


@pytest.mark.parametrize("bad", ["params_ratio", "params_padding", "config_ratio", "side"])
def test_native_frame_guard_before_field_load(tmp_path, monkeypatch, bad):
    # The guard should reject metadata, not accept a wrong native-frame conversion.
    pytest.importorskip("SimpleITK")
    method = dict(field="absent.mha", postprocessing_params="params.json",
                  report="runtime.json", configuration="config.json")
    params = dict(target_resample_ratio=1, source_resample_ratio=1, initial_resample_ratio=1,
                  pad_1=[[0, 0], [0, 0]], pad_2=[[0, 0], [0, 0]])
    runtime = dict(image_side=512, initial_resample_ratio=1,
                   post_affine_matrix=np.eye(2).tolist(), post_affine_offset=[0, 0])
    config = dict(loading_params=dict(loader="pil", source_resample_ratio=1, target_resample_ratio=1))
    if bad == "params_ratio": params["initial_resample_ratio"] = 2
    elif bad == "params_padding": params["pad_2"][0][0] = 1
    elif bad == "config_ratio": config["loading_params"]["source_resample_ratio"] = .5
    else: runtime["image_side"] = 1024
    for name, value in (("params", params), ("runtime", runtime), ("config", config)):
        write_json(tmp_path / f"{name}.json", value)
    with pytest.raises(ValueError):
        scorer._native(method, tmp_path, np.eye(2), np.zeros(2))
