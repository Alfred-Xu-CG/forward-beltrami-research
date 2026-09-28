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
