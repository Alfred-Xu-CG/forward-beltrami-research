"""Native theta scoring: rectangular normalization, unequal loading, padding."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_dhr_initial_score import map_initial_native
from tools.digital_compare_appearance import dhr_map_at_unit_queries


def test_hand_calculated_rectangular_nonunit_loading_and_padding():
    params = dict(source_resample_ratio=.25, target_resample_ratio=.5, initial_resample_ratio=2.,
                  pad_1=[[2, 0], [5, 0]], pad_2=[[7, 0], [3, 0]])
    theta = [[[.8, .1, .04], [-.2, 1.1, -.03]]]
    result = map_initial_native([[19.5, 29.5]], theta, params, (30, 40))
    # Fixed preprocessed pixel index=(6,10.5), center-from-boundary=(6.5,11),
    # normalized=(-.675,-4/15). Mapped normalized=(-79/150,-113/600),
    # moving preprocessed pixel index=(8.966...,11.675).
    np.testing.assert_allclose(result, [[55.233333333333334, 88.9]], rtol=0, atol=2e-13)


def test_agrees_with_torch_native_affine_grid_and_pixel_displacement_helper():
    h, w = 23, 37
    theta = torch.tensor([[[.9, .15, .04], [-.12, 1.04, -.06]]], dtype=torch.float32)
    identity = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    field = F.affine_grid(theta, (1, 1, h, w), align_corners=False) - F.affine_grid(identity, (1, 1, h, w), align_corners=False)
    pixels = (field[0] * torch.tensor([w/2, h/2])).permute(2, 0, 1).numpy()
    params = dict(source_resample_ratio=.8, target_resample_ratio=.6, initial_resample_ratio=1.3,
                  pad_1=[[2, 1], [3, 4]], pad_2=[[1, 2], [4, 1]])
    fixed_wh, moving_wh = (60, 40), (50, 35)
    points = np.array([[7.123, 5.25], [22.4, 17.89], [48.123, 30.321]])
    direct = map_initial_native(points, theta.numpy(), params, (h, w))
    query = torch.tensor((points+.5)/np.array(fixed_wh), dtype=torch.float32)[None, None]
    unit = dhr_map_at_unit_queries(pixels, params, fixed_size=fixed_wh, moving_size=moving_wh, query=query)[0, 0].numpy()
    reconstructed = unit * np.array(moving_wh) - .5
    np.testing.assert_allclose(direct, reconstructed, rtol=0, atol=7e-6)


def test_invalid_transform_and_scale_rejected():
    params = dict(source_resample_ratio=1, target_resample_ratio=1, pad_1=[[0,0],[0,0]],pad_2=[[0,0],[0,0]])
    with pytest.raises(ValueError):
        map_initial_native([[1,2]], [[[np.nan,0,0],[0,1,0]]], params, (20,30))
    params["target_resample_ratio"] = 0
    with pytest.raises(ValueError):
        map_initial_native([[1,2]], [[[1,0,0],[0,1,0]]], params, (20,30))
