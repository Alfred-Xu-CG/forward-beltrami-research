"""GPU pilot must switch all DHR device flags together."""

import pytest

from tools import digital_dhr_baseline


@pytest.mark.parametrize("device,use_cuda", [("cpu", False), ("cuda", True)])
def test_configure_dhr_device_updates_every_stage(device, use_cuda):
    params = {
        "device": "cpu",
        "initial_registration_params": {"device": "cpu", "cuda": False},
        "nonrigid_registration_params": {"device": "cpu"},
    }
    digital_dhr_baseline.configure_device(params, device)
    assert params["device"] == device
    assert params["initial_registration_params"]["device"] == device
    assert params["initial_registration_params"]["cuda"] is use_cuda
    assert params["nonrigid_registration_params"]["device"] == device


def test_configure_dhr_device_rejects_unknown_target():
    with pytest.raises(ValueError, match="device"):
        digital_dhr_baseline.configure_device({"initial_registration_params": {},
                                               "nonrigid_registration_params": {}}, "maybe")


def test_registration_size_override_is_explicit_and_validated():
    params = {"initial_registration_params": {"registration_sizes": [150, 200, 250]}}
    digital_dhr_baseline.configure_registration_sizes(params, None)
    assert params["initial_registration_params"]["registration_sizes"] == [150, 200, 250]
    digital_dhr_baseline.configure_registration_sizes(params, [150, 250, 400])
    assert params["initial_registration_params"]["registration_sizes"] == [150, 250, 400]
    for invalid in ([], [0], [-100], [150, 150]):
        with pytest.raises(ValueError, match="registration sizes"):
            digital_dhr_baseline.configure_registration_sizes(params, invalid)
