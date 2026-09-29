"""Physical-canvas affine conjugacy check without any image or landmarks."""

import numpy as np

from tools.digital_histo_canvas_affine import convert, native_unit_to_canvas_affine


def test_affine_conjugation_commutes_with_canvas_coordinates():
    layout = {"side": 512,
              "fixed": {"resized_wh": [401, 512], "padding_xy": [55, 0]},
              "moving": {"resized_wh": [387, 511], "padding_xy": [62, 0]}}
    native_m = np.array([[.93, .15], [-.08, .97]])
    native_b = np.array([-.036, .029])
    cm, cb = convert(native_m, native_b, layout)
    df, tf = native_unit_to_canvas_affine(layout["fixed"], 512)
    dm, tm = native_unit_to_canvas_affine(layout["moving"], 512)
    q = np.array([[.1, .7], [.5, .5], [.9, .2]])
    via_native = (q @ native_m.T + native_b) @ dm.T + tm
    via_canvas = (q @ df.T + tf) @ cm.T + cb
    np.testing.assert_allclose(via_canvas, via_native, rtol=0, atol=2e-16)
    assert np.linalg.det(cm) > 0
