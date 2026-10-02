"""Fixed tissue support retains native axes, categorical resize and static weights."""
import json

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_tissue_support import prepare_tissue_support, load_tissue_support


def inputs(tmp_path):
    native = np.zeros((6, 10), dtype=np.uint8)
    native[0:4, 4:8] = 1  # nonsymmetric, away from both coordinate origins
    mask, original, canvas = [tmp_path / name for name in ("mask.tif", "native.tif", "fixed.png")]
    Image.fromarray(native).save(mask)
    Image.new("RGB", (10, 6), (120, 90, 70)).save(original)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(canvas)
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps(dict(side=8, fixed=dict(original_wh=[10, 6], resized_wh=[5, 3],
        padding_xy=[1, 2], effective_original_to_canvas_scale_xy=[.5, .5],
        canvas_png=str(canvas), source=str(original)))))
    return mask, original, layout, canvas, tmp_path / "support.npz"


def test_axes_padding_nearest_and_area_weights(tmp_path):
    paths = inputs(tmp_path)
    record = prepare_tissue_support(*paths)
    support, metadata = load_tissue_support(paths[-1], paths[-2], 8, dtype=torch.float64)
    expected = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    expected[:, :, 2:4, 3:5] = 1
    assert torch.equal(support, expected)
    assert record == metadata
    assert record["manual_correspondence_landmarks_read"] is False
    coarse = F.interpolate(support, size=(4, 4), mode="area")
    assert set(coarse.unique().tolist()) == {0., .5}
    assert float(coarse.sum()) * 4 == float(support.sum())
    with pytest.raises(ValueError, match="frame"):
        load_tissue_support(paths[-1], tmp_path / "other.png", 8)
    with pytest.raises(FileExistsError):
        prepare_tissue_support(*paths)


@pytest.mark.parametrize("failure", ["dimensions", "orientation", "nonbinary", "empty", "layout"])
def test_reject_incompatible_native_inputs(tmp_path, failure):
    paths = inputs(tmp_path)
    if failure == "dimensions":
        Image.fromarray(np.ones((5, 10), dtype=np.uint8)).save(paths[0])
    elif failure == "orientation":
        Image.fromarray(np.ones((6, 10), dtype=np.uint8)).save(paths[0], tiffinfo={274: 3})
    elif failure == "nonbinary":
        Image.fromarray(np.full((6, 10), 2, dtype=np.uint8)).save(paths[0])
    elif failure == "empty":
        Image.fromarray(np.zeros((6, 10), dtype=np.uint8)).save(paths[0])
    else:
        layout = json.loads(paths[2].read_text())
        layout["fixed"]["effective_original_to_canvas_scale_xy"] = [.6, .5]
        paths[2].write_text(json.dumps(layout))
    with pytest.raises(ValueError):
        prepare_tissue_support(*paths)
    assert not paths[-1].exists()


def test_loader_changes_only_fixed_support_and_leaves_default_identical(tmp_path):
    from tools.coordinated_real_case import load_registration_evidence
    paths = inputs(tmp_path)
    prepare_tissue_support(*paths)
    # Use actual textured canvas evidence; preparation/frame stays unchanged.
    rng = np.random.default_rng(943)
    Image.fromarray(rng.integers(20, 250, (8, 8, 3), dtype=np.uint8)).save(paths[-2])
    baseline = load_registration_evidence(paths[-2], paths[-2], 8, dtype=torch.float64)
    explicit_default = load_registration_evidence(paths[-2], paths[-2], 8, dtype=torch.float64, fixed_mask_path=None)
    changed = load_registration_evidence(paths[-2], paths[-2], 8, dtype=torch.float64, fixed_mask_path=paths[-1])
    assert all(torch.equal(a, b) for a, b in zip(baseline[:3], explicit_default[:3]))
    assert baseline[3] == explicit_default[3]
    assert torch.equal(changed[0], baseline[0]) and torch.equal(changed[1], baseline[1])
    assert torch.equal(changed[2], load_tissue_support(paths[-1], paths[-2], 8, dtype=torch.float64)[0])
    assert not torch.equal(changed[2], baseline[2])
    assert changed[3]["supplied_tissue_annotation"] is True


def test_masked_value_gradient_fixed_denominator_and_unchanged_priors():
    from tools.coordinated_real_case import Evidence
    rng = torch.Generator().manual_seed(71)
    fixed = torch.rand(1, 1, 16, 16, generator=rng, dtype=torch.float64)
    moving = torch.rand(1, 1, 16, 16, generator=rng, dtype=torch.float64)
    mask = torch.zeros_like(fixed)
    mask[:, :, 3:13, 2:6] = 1.
    a, b = torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    args = (fixed, moving, a, b, "mind", 3., 1., 1e-4)
    roi = Evidence(*args, fixed_mask=mask, interpolation="p1_ac", strain_model="p1_arap")
    full = Evidence(*args, fixed_mask=torch.ones_like(mask), interpolation="p1_ac", strain_model="p1_arap")
    line = torch.linspace(0, 1, 7, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    v = torch.stack((xx, yy), -1)[None]
    v[:, 1:-1, 1:-1] += torch.rand(1, 5, 5, 2, generator=rng, dtype=v.dtype) * .008
    v.requires_grad_()
    value, terms = roi(v)
    _, full_terms = full(v)
    assert not torch.equal(terms["image"], full_terms["image"])
    for name in ("strain", "shape"):
        assert torch.equal(terms[name], full_terms[name])
    gradient, = torch.autograd.grad(value, v)
    direction = torch.randn(v.shape, generator=rng, dtype=v.dtype) * .01
    eps = 1e-6
    numerical = (roi(v + eps * direction)[0] - roi(v - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close((gradient * direction).sum(), numerical, rtol=3e-5, atol=2e-8)
    # Alter feature values strictly outside every supported query. This must not
    # change the objective/map derivative, unlike an adaptive overlap mask.
    roi.moving_feature = roi.moving_feature.clone()
    roi.moving_feature[:, :, :, 12:] += 17.
    updated, _ = roi(v)
    updated_gradient, = torch.autograd.grad(updated, v)
    assert torch.equal(value, updated) and torch.equal(gradient, updated_gradient)
    assert float(roi.denominator) == 40.


def test_pilot_mask_arm_forces_continuation_and_records_path(tmp_path):
    from tools.coordinated_miit_multiscale_pilot import configuration
    report = dict(configuration=dict(method="analytic", output_selection="best_full", fixed="fixed.png"))
    cfg = configuration(report, tmp_path / "out.npz", "continuation", fixed_mask=tmp_path / "support.npz")
    assert cfg.fixed_mask == tmp_path / "support.npz"
    assert cfg.mind_frame == "shared_affine" and cfg.image_objective == "continuation"
    with pytest.raises(ValueError, match="support only"):
        configuration(report, tmp_path / "out.npz", "simultaneous_multiscale", fixed_mask=tmp_path / "support.npz")
