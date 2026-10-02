from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from tools import coordinated_matchanything as module


def points(count=8):
    return np.stack((np.arange(count) * 16. + 40., np.full(count, 120.)), axis=1)


def test_point_record_preserves_pixel_centers_confidence_and_one_affine():
    p0 = points()
    p1 = p0 + np.array([3., -5.])
    matrix = np.array([[.8, .2], [-.1, .9]], dtype=np.float32)
    offset = np.array([.03, .1], dtype=np.float32)
    confidence = np.linspace(.2, .9, 8)
    result = module.point_record(p0, p1, confidence, matrix, offset)
    assert result['status'] == 'ok'
    np.testing.assert_array_equal(result['source_points_unit'], (p0 + .5) / 512)
    np.testing.assert_array_equal(result['target_points_unit'], (p1 + .5) / 512)
    np.testing.assert_array_equal(result['confidence'], confidence)
    assert result['eligible_matches'] == 8
    assert not result['global_geometric_ransac_used']


def test_finite_domain_filter_retains_unit_endpoints_without_clipping():
    p0 = np.array([[-.5, -.5], [511.5, 511.5], [-.5001, 1], [1, 511.5001], [2, 2]])
    p1 = np.array([[10., 10.], [511.5, 511.5], [1, 1], [1, 1], [-1, 1]])
    result = module.point_record(p0, p1, np.ones(5), np.eye(2), np.zeros(2))
    np.testing.assert_array_equal(result['source_points_unit'], [[0, 0], [1, 1]])
    assert result['discarded_out_of_unit_domain'] == 3
    assert result['raw_matches'] == 2
    assert result['status'] == 'insufficient_matches'


def test_static_world_eligibility_keeps_table_but_counts_original_canvas():
    p0 = points()
    p1 = p0.copy()
    p1[-1] = [511., 400.]
    result = module.point_record(p0, p1, np.ones(8), np.eye(2), np.array([.1, 0.]))
    assert len(result['source_points_unit']) == 8
    assert result['eligible_matches'] == 7
    assert result['eligible_confidence_mass'] == 7
    assert result['status'] == 'insufficient_matches'


@pytest.mark.parametrize('which', ['source', 'target', 'confidence'])
def test_nonfinite_is_failure_even_when_point_would_be_discarded(which):
    p0, p1, confidence = points(), points(), np.ones(8)
    p0[0] = [-100., -100.]
    {'source': p0, 'target': p1, 'confidence': confidence}[which].flat[0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        module.point_record(p0, p1, confidence, np.eye(2), np.zeros(2))


@pytest.mark.parametrize('confidence', [np.full(8, -0.1), np.full(8, 1.01)])
def test_confidence_outside_unit_interval_is_failure(confidence):
    with pytest.raises(ValueError, match='confidence'):
        module.point_record(points(), points(), confidence, np.eye(2), np.zeros(2))


def test_no_points_and_all_zero_confidence_are_explicit_insufficient():
    for p0, confidence in [(np.empty((0, 2)), np.empty(0)), (points(), np.zeros(8))]:
        result = module.point_record(p0, p0, confidence, np.eye(2), np.zeros(2))
        assert result['status'] == 'insufficient_matches'
        assert result['eligible_matches'] == 0
        assert result['eligible_confidence_mass'] == 0


def test_source_support_is_containing_pixel_diagnostic_not_filter():
    p0 = points()
    p0[0] = [511.5, 511.5]
    support = np.zeros((512, 512), bool)
    support[511, 511] = True
    result = module.point_record(p0, p0, np.ones(8), np.eye(2), np.zeros(2), fixed_support=support)
    assert result['raw_matches'] == 8
    assert result['source_original_gray_support_fraction'] == .125


def test_duplicate_coordinates_are_reported_but_never_deduplicated():
    p0, p1 = points(), points()
    p0[1] = p0[0]
    p1[2:4] = p1[0]
    result = module.point_record(p0, p1, np.ones(8), np.eye(2), np.zeros(2))
    assert result['raw_matches'] == result['eligible_matches'] == 8
    assert result['unique_source_coordinates'] == 7
    assert result['unique_target_coordinates'] == 6
    assert result['maximum_source_multiplicity'] == 2
    assert result['maximum_target_multiplicity'] == 3


def test_gray_reader_is_ordinary_pil_l_without_resize(tmp_path):
    pixels = np.zeros((512, 512, 3), np.uint8)
    pixels[..., 0] = 255
    path = tmp_path / 'red.png'
    Image.fromarray(pixels).save(path)
    gray = module.read_matcher_gray(path)
    assert gray.shape == (1, 1, 512, 512)
    torch.testing.assert_close(gray, torch.full_like(gray, 76. / 255))
    wrong = tmp_path / 'wrong.png'
    Image.fromarray(pixels[:256]).save(wrong)
    with pytest.raises(ValueError, match='512'):
        module.read_matcher_gray(wrong)


def test_extractor_uses_border_prewarp_and_writes_compatible_table(tmp_path, monkeypatch):
    captured = {}

    class FakeModel(torch.nn.Module):
        def forward(self, batch):
            captured.update({key: value.clone() for key, value in batch.items()})
            batch['mkpts0_f'] = torch.tensor(points(), dtype=torch.float32)
            batch['mkpts1_f'] = torch.tensor(points() + [1., 0.], dtype=torch.float32)
            batch['mconf'] = torch.ones(8) * .7

    monkeypatch.setattr(module, '_load_model', lambda source, checkpoint, device: (FakeModel(), {'strict_keys': True}))
    fixed, moving = tmp_path / 'fixed.png', tmp_path / 'moving.png'
    raster = np.broadcast_to(np.arange(512, dtype=np.float64)[None, :] / 511 * 255, (512, 512)).astype(np.uint8)
    Image.fromarray(raster).save(fixed)
    Image.fromarray(raster[:, ::-1]).save(moving)
    affine = tmp_path / 'affine.npz'
    matrix, offset = np.eye(2, dtype=np.float32), np.array([.1, -.1], dtype=np.float32)
    np.savez(affine, post_affine_matrix=matrix, post_affine_offset=offset)
    extractor = module.FrozenMatchAnything(tmp_path, tmp_path / 'test.ckpt', device='cpu')
    output = tmp_path / 'matches.json'
    result = extractor.extract(fixed, moving, affine, output=output)
    assert output.exists() and result['status'] == 'ok'
    torch.testing.assert_close(captured['image0'], module.read_matcher_gray(fixed))
    # Literal independent horizontal +51.2px sample at the upper-left point.
    # Reversal of uint8-rounded ramp differs by one count from 255-raster.
    expected = .8 * raster[0, 460] / 255 + .2 * raster[0, 459] / 255
    assert float(captured['image1'][0, 0, 0, 0]) == pytest.approx(expected, abs=2e-5)
    assert captured['image1'][0, 0, 0, -1] == 0  # right BORDER, not wrap/white
    assert result['targets_manual_landmarks_or_dense_teacher_loaded'] is False
    from tools.coordinated_real_case import load_image_matches
    loaded, _ = load_image_matches(output, matrix, offset, fixed_path=fixed, moving_path=moving,
                                   image_side=512, device='cpu', dtype=torch.float64, robust_scale=8.)
    assert loaded.source.shape == (1, 8, 2)
    with pytest.raises(FileExistsError):
        extractor.extract(fixed, moving, affine, output=output)
    extractor.close()
    with pytest.raises(RuntimeError, match='closed'):
        extractor.extract(fixed, moving, affine)


def test_point_record_rejects_reflection_and_malformed_arrays():
    with pytest.raises(ValueError, match='positive'):
        module.point_record(points(), points(), np.ones(8), np.diag([-1., 1.]), np.zeros(2))
    with pytest.raises(ValueError, match='shape'):
        module.point_record(points(), points()[:3], np.ones(8), np.eye(2), np.zeros(2))
