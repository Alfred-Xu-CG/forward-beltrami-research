"""Pyramid selection must avoid loading enormous full-resolution TIFF pages."""

from tools.digital_acrobat_thumbnail import choose_level


def test_choose_coarsest_level_above_requested_sampling_size() -> None:
    assert choose_level([(20608, 20480), (10304, 10240),
                         (5152, 5120), (2576, 2560),
                         (1288, 1280), (644, 640),
                         (322, 320)], 512) == 5
    assert choose_level([(400, 200), (200, 100)], 512) == 0
