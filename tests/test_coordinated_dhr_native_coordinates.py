"""Native DHR frame checks using its installed resampler and actual MHA saver.

These exercise rectangular, unequal, padded images with nonintegral resampling.
They do not run a registration optimizer or use evaluation annotations.
"""
from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from tools.digital_compare_appearance import dhr_map_at_unit_queries


@pytest.fixture(scope="module")
def native_dhr():
    pytest.importorskip("deeperhistreg")  # Installs the package's legacy aliases.
    from dhr_utils import utils
    from deeperhistreg.dhr_input_output.dhr_savers.displacement_saver import DisplacementFieldSaver
    import SimpleITK
    return utils, DisplacementFieldSaver, SimpleITK


def test_nominal_ratio_not_floored_extent_controls_dhr_pixel_centers(native_dhr):
    u, _, _ = native_dhr
    h, w, ratio = 91, 127, 1.37
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ramps = torch.stack((xx, yy)).float()[None]
    reduced = u.resample(ramps, ratio)
    nh, nw = reduced.shape[-2:]
    assert (nh, nw) == (int(h / ratio), int(w / ratio))
    expected_x = (torch.arange(nw) + .5) * ratio - .5
    expected_y = (torch.arange(nh) + .5) * ratio - .5
    torch.testing.assert_close(reduced[0, 0, nh // 2], expected_x, atol=2e-5, rtol=0)
    torch.testing.assert_close(reduced[0, 1, :, nw // 2], expected_y, atol=2e-5, rtol=0)
    # Deliberate wrong interpretation is detectably different on this fixture.
    wrong_x = (torch.arange(nw) + .5) * (w / nw) - .5
    assert float((expected_x - wrong_x).abs().max()) > .5


@pytest.mark.parametrize("source_ratio,target_ratio,initial_ratio", [
    (1., 1., 1.37), (.61, .73, 1.19), (1., 1., 1.),
])
@pytest.mark.parametrize("affine", [False, True])
def test_saved_rectangular_native_map_has_expected_pixel_coordinates(
        native_dhr, tmp_path, source_ratio, target_ratio, initial_ratio, affine):
    u, saver_type, sitk = native_dhr
    fixed_wh, moving_wh = np.array([127, 91]), np.array([113, 103])
    fixed = u.resample(torch.zeros(1, 1, *fixed_wh[::-1]), 1 / target_ratio)
    moving = u.resample(torch.zeros(1, 1, *moving_wh[::-1]), 1 / source_ratio)
    source_pad, target_pad, params = u.pad_to_same_size(moving, fixed, 0.)
    reduced = u.resample(target_pad, initial_ratio)
    h, w = reduced.shape[-2:]
    pad_f = np.array([params["pad_2"][1][0], params["pad_2"][0][0]])
    pad_m = np.array([params["pad_1"][1][0], params["pad_1"][0][0]])
    params.update(source_resample_ratio=source_ratio,
                  target_resample_ratio=target_ratio,
                  initial_resample_ratio=initial_ratio)
    # Begin with a known affine in ORIGINAL zero-based pixel-center coordinates.
    matrix = np.array([[.94, .07], [-.04, 1.02]]) if affine else np.eye(2)
    offset = np.array([2.3, -1.7]) if affine else np.zeros(2)
    yy, xx = np.mgrid[:h, :w]
    lattice = np.stack((xx, yy), axis=-1)
    native_fixed = ((lattice + .5) * initial_ratio - pad_f) / target_ratio - .5
    native_moving = native_fixed @ matrix.T + offset
    moving_lattice = ((native_moving + .5) * source_ratio + pad_m) / initial_ratio - .5
    pixel_displacement = moving_lattice - lattice
    normalized = torch.tensor(pixel_displacement / np.array([w / 2, h / 2]),
                              dtype=torch.float32)[None]
    path = Path(tmp_path) / "rectangular_native.mha"
    saver_type().save(normalized, path)
    saved = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert saved.shape == (2, h, w)
    np.testing.assert_allclose(saved.transpose(1, 2, 0), pixel_displacement,
                               rtol=1e-6, atol=4e-6)
    # Queries are off lattice, well inside the sampled field: no extrapolation.
    sample_lattice = np.array([[.21*w, .24*h], [.65*w, .72*h], [.43*w, .37*h]])
    native_queries = ((sample_lattice + .5)*initial_ratio-pad_f)/target_ratio-.5
    units = torch.tensor((native_queries+.5)/fixed_wh, dtype=torch.float32)[None, None]
    mapped = dhr_map_at_unit_queries(saved, params, fixed_size=tuple(fixed_wh),
        moving_size=tuple(moving_wh), query=units)[0, 0].numpy()*moving_wh-.5
    expected = native_queries @ matrix.T + offset
    np.testing.assert_allclose(mapped, expected, atol=3e-5, rtol=0)


def test_released_scorer_native_three_pair_manifest_and_missing_failure_policy(native_dhr, tmp_path):
    """Actual MHA files, six native TIFF/CSVs, unequal padding, no square-canvas shortcut."""
    from tools.coordinated_dhr_released_score import score
    from tools.digital_acrobat_pair_canvas import compute_layout
    u, saver_type, _ = native_dhr
    source, prediction, comparison = (tmp_path / name for name in ("source", "prediction", "comparison"))
    prediction.mkdir(); comparison.mkdir()
    fixed_wh, moving_wh = np.array([127, 91]), np.array([113, 103])
    fixed_points = np.column_stack((np.linspace(20, 90, 124), 40 + 15*np.sin(np.arange(124))))
    rows = []
    for moving, fixed in ((2, 3), (7, 8), (10, 11)):
        name = f"miit_{moving}_to_{fixed}"
        # Deliberate reflection checks that topology is reported, not used to drop accuracy.
        matrix = np.diag([-1., 1.]) if moving == 10 else np.array([[.94, .07], [-.04, 1.02]])
        offset = np.array([110., 0.]) if moving == 10 else np.array([2.3, -1.7])
        moving_points = fixed_points @ matrix.T + offset
        for section, wh, points, missing_index in ((fixed, fixed_wh, fixed_points, 10),
                                                  (moving, moving_wh, moving_points, 20)):
            image_dir, labels = source/str(section)/"images", source/str(section)/"landmarks"
            image_dir.mkdir(parents=True); labels.mkdir()
            Image.new("RGB", tuple(wh), "white").save(image_dir/"image.tif")
            with (labels/f"{section:02d}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["label", "x", "y"])
                for i, p in enumerate(points):
                    writer.writerow([f"label{i:03d}", *(p if i != missing_index else [float("inf")]*2)])
        src, trg, params = u.pad_to_same_size(torch.zeros(1,1,*moving_wh[::-1]),
                                             torch.zeros(1,1,*fixed_wh[::-1]), 0.)
        ratio = 1.37
        h, w = u.resample(trg, ratio).shape[-2:]
        pad_f = np.array([params["pad_2"][1][0], params["pad_2"][0][0]])
        pad_m = np.array([params["pad_1"][1][0], params["pad_1"][0][0]])
        yy, xx = np.mgrid[:h, :w]; lattice = np.stack((xx, yy), axis=-1)
        x_native = (lattice+.5)*ratio-pad_f-.5
        y_native = x_native@matrix.T+offset
        df = ((y_native+.5+pad_m)/ratio-.5-lattice)/np.array([w/2,h/2])
        saver_type().save(torch.tensor(df,dtype=torch.float32)[None], prediction/(name+".mha"))
        params.update(source_resample_ratio=1., target_resample_ratio=1., initial_resample_ratio=ratio)
        (prediction/(name+".json")).write_text(json.dumps(params))
        moving_layout, fixed_layout = compute_layout([
            (*moving_wh[::-1],1.,1.), (*fixed_wh[::-1],1.,1.)], side=512)
        moving_layout["original_wh"] = moving_wh.tolist()
        fixed_layout["original_wh"] = fixed_wh.tolist()
        (comparison/(name+"_layout.json")).write_text(json.dumps(dict(side=512,moving=moving_layout,fixed=fixed_layout)))
        rows.append(dict(name=name,moving_section=moving,fixed_section=fixed,status="ok",
                         fixed_original_wh=fixed_wh.tolist(),moving_original_wh=moving_wh.tolist(),
                         field=name+".mha",postprocessing_params=name+".json"))
    manifest = dict(prediction_complete=True,annotations_read=False,preset="synthetic_known_affine",rows=rows)
    manifest_path = prediction/"predictions.json"
    manifest_path.write_text(json.dumps(manifest))
    result = score(prediction, source, comparison)
    assert result["scored_pairs"] == 3 and result["failed_pairs"] == 0
    for row in result["rows"]:
        assert row["nominal_landmarks"] == 124 and row["scored_landmarks"] == 122
        assert row["unavailable_pair_labels"] == ["label010", "label020"]
        assert row["metrics"]["native_moving_pixels"]["maximum"] < 4e-5
    assert result["rows"][2]["topology"]["nonpositive_corners"] > 0
    manifest["rows"][1].update(status="failed",error="deliberate prediction failure")
    manifest_path.write_text(json.dumps(manifest))
    failed = score(prediction, source, comparison)
    assert failed["pair_denominator"] == 3 and failed["failed_pairs"] == 1
    assert failed["all_three"] is None
