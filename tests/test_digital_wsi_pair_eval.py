import csv

import pytest
from PIL import Image

from tools.digital_wsi_pair import make_identity, save_transform
from tools.digital_wsi_pair_eval import evaluate_transform, read_landmarks


def _write_landmarks(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([" ", "X", "Y"])
        writer.writerows(rows)


def test_identity_tre_uses_matched_ids_and_pixel_xy(tmp_path):
    moving = tmp_path / "moving.jpg"
    fixed = tmp_path / "fixed.jpg"
    Image.new("RGB", (12, 8)).save(moving)
    Image.new("RGB", (10, 9)).save(fixed)
    transform_path = tmp_path / "identity.json"
    save_transform(make_identity(moving, fixed), transform_path)
    moving_csv = tmp_path / "moving.csv"
    fixed_csv = tmp_path / "fixed.csv"
    _write_landmarks(moving_csv, [(2, 5.5, 2), (1, 1, 1)])
    _write_landmarks(fixed_csv, [(1, 4, 5), (2, 5.5, 2)])

    result = evaluate_transform(transform_path, moving_csv, fixed_csv)
    assert result["n_landmarks"] == 2
    assert result["mean_tre_px"] == pytest.approx(2.5)
    assert result["median_tre_px"] == pytest.approx(2.5)
    assert result["p95_tre_px"] == pytest.approx(4.75)
    assert result["max_tre_px"] == pytest.approx(5.0)
    assert result["mapped_inside_fixed_count"] == 2
    assert result["unit"] == "pixel"


def test_rejects_missing_or_duplicate_landmark_ids(tmp_path):
    path = tmp_path / "landmarks.csv"
    _write_landmarks(path, [(1, 2, 3), (1, 4, 5)])
    with pytest.raises(ValueError, match="duplicate"):
        read_landmarks(path, (10, 10))
    _write_landmarks(path, [(1, 2, 3), (2, 4, 5)])
    assert set(read_landmarks(path, (10, 10))) == {"1", "2"}


def test_rejects_different_landmark_id_sets(tmp_path):
    moving = tmp_path / "moving.jpg"
    fixed = tmp_path / "fixed.jpg"
    Image.new("RGB", (10, 10)).save(moving)
    Image.new("RGB", (10, 10)).save(fixed)
    transform_path = tmp_path / "identity.json"
    save_transform(make_identity(moving, fixed), transform_path)
    moving_csv = tmp_path / "moving.csv"
    fixed_csv = tmp_path / "fixed.csv"
    _write_landmarks(moving_csv, [(1, 1, 1)])
    _write_landmarks(fixed_csv, [(2, 1, 1)])
    with pytest.raises(ValueError, match="ID sets differ"):
        evaluate_transform(transform_path, moving_csv, fixed_csv)


def test_rejects_swapped_landmark_frame(tmp_path):
    moving = tmp_path / "moving.jpg"
    fixed = tmp_path / "fixed.jpg"
    Image.new("RGB", (12, 8)).save(moving)
    Image.new("RGB", (10, 9)).save(fixed)
    transform_path = tmp_path / "identity.json"
    save_transform(make_identity(moving, fixed), transform_path)
    moving_csv = tmp_path / "moving.csv"
    fixed_csv = tmp_path / "fixed.csv"
    _write_landmarks(moving_csv, [(1, 11, 2)])
    _write_landmarks(fixed_csv, [(1, 11, 2)])
    with pytest.raises(ValueError, match="outside"):
        evaluate_transform(transform_path, moving_csv, fixed_csv)


def test_rejects_swapped_role_files_even_when_points_fit_both_images(tmp_path):
    moving = tmp_path / "Images_CD68.jpg"
    fixed = tmp_path / "Images_CD4.jpg"
    Image.new("RGB", (12, 8)).save(moving)
    Image.new("RGB", (12, 8)).save(fixed)
    transform_path = tmp_path / "identity.json"
    save_transform(make_identity(moving, fixed), transform_path)
    moving_csv = tmp_path / "Landmarks_CD68.csv"
    fixed_csv = tmp_path / "Landmarks_CD4.csv"
    _write_landmarks(moving_csv, [(1, 1, 1)])
    _write_landmarks(fixed_csv, [(1, 2, 2)])
    with pytest.raises(ValueError, match="role"):
        evaluate_transform(transform_path, fixed_csv, moving_csv)


def test_rejects_nonpositive_affine_determinant(tmp_path):
    moving = tmp_path / "moving.jpg"
    fixed = tmp_path / "fixed.jpg"
    Image.new("RGB", (12, 8)).save(moving)
    Image.new("RGB", (10, 9)).save(fixed)
    transform = make_identity(moving, fixed)
    transform["matrix_moving_to_fixed_xy"] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    transform_path = tmp_path / "bad.json"
    save_transform(transform, transform_path)
    moving_csv = tmp_path / "moving.csv"
    fixed_csv = tmp_path / "fixed.csv"
    _write_landmarks(moving_csv, [(1, 1, 1)])
    _write_landmarks(fixed_csv, [(1, 1, 1)])
    with pytest.raises(ValueError, match="determinant"):
        evaluate_transform(transform_path, moving_csv, fixed_csv)
