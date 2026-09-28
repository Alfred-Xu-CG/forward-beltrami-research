"""Classical translation and border controls have the intended geometry."""

import numpy as np
import torch

from tools.digital_q1_synthetic_controls import phase_shift_normalized, zero_boundary
from tools.digital_q1_synthetic_translation import make_translated_moving


def test_phase_correlation_returns_fixed_to_moving_shift():
    rng = np.random.default_rng(5)
    fixed = torch.from_numpy(rng.random((1, 1, 128, 128), dtype=np.float32))
    shift = torch.tensor([[4 / 127, -3 / 127]], dtype=torch.float32)
    moving = make_translated_moving(fixed, shift)
    fixed_array = fixed[0, 0].numpy()
    moving_array = moving[0, 0].numpy()
    fixed_before = fixed_array.copy()
    moving_before = moving_array.copy()
    estimated = phase_shift_normalized(fixed_array, moving_array)
    np.testing.assert_allclose(estimated, shift[0].numpy(), atol=.004)
    np.testing.assert_array_equal(fixed_array, fixed_before)
    np.testing.assert_array_equal(moving_array, moving_before)


def test_zero_boundary_removes_only_specified_margin():
    image = torch.ones((2, 1, 17, 19))
    masked = zero_boundary(image, 3)
    assert torch.count_nonzero(masked[:, :, :3]) == 0
    assert torch.count_nonzero(masked[:, :, :, -3:]) == 0
    assert torch.all(masked[:, :, 3:-3, 3:-3] == 1)
