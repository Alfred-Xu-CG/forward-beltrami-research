import math

import numpy as np
import pytest

from tools.digital_q1_schedule_targets import create_sine_shear_target


@pytest.mark.parametrize("amplitude", [.10, .24])
def test_sine_shear_target_is_saved_safe_and_has_exact_boundary(tmp_path, amplitude):
    path = tmp_path / "teacher.npz"
    report = create_sine_shear_target(path, side=33, amplitude=amplitude)
    assert report["saved_binary_valid"]
    assert report["saved_binary_nonpositive_corners"] == 0
    assert report["continuum_jacobian_lower_bound"] == pytest.approx(
        1 - amplitude * math.pi
    )
    with np.load(path) as archive:
        teacher = archive["teacher_vertices"]
        reference = archive["boundary_reference"]
    assert teacher.shape == (1, 33, 33, 2)
    assert np.array_equal(teacher[:, 0], reference[:, 0])
    assert np.array_equal(teacher[:, -1], reference[:, -1])
    assert np.array_equal(teacher[:, :, 0], reference[:, :, 0])
    assert np.array_equal(teacher[:, :, -1], reference[:, :, -1])
    assert teacher[0, 16, 16, 0] == np.float32(.5 + amplitude)


def test_sine_shear_center_report_rejects_even_side(tmp_path):
    with pytest.raises(ValueError, match="odd"):
        create_sine_shear_target(tmp_path / "even.npz", side=4, amplitude=.1)
