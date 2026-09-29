"""The network timing path includes image sampling but excludes image I/O."""

import torch

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_q1_network_benchmark import benchmark_tensors, benchmark_vjp_tensors


def test_inference_benchmark_runs_full_q1_warp():
    torch.manual_seed(5)
    model = Q1ImageRegistrationNetwork(seed_side=5, final_side=9, width=4,
                                       feature_side=16, flow_hint=False)
    fixed = torch.rand(1, 1, 16, 16)
    moving = torch.rand(1, 1, 16, 16)
    report = benchmark_tensors(model, fixed, moving, warmups=1, repeats=2)
    assert report["control_vertices"] == 81
    assert report["image_pixels"] == 256
    assert report["forward_median_seconds"] > 0
    assert report["forward_plus_image_warp_median_seconds"] > 0
    assert report["cuda_peak_allocated_bytes"] is None


def test_vjp_benchmark_includes_images_encoder_decoder_and_warp():
    torch.manual_seed(7)
    model = Q1ImageRegistrationNetwork(seed_side=5, final_side=9, width=4,
                                       feature_side=16, flow_hint=False)
    fixed = torch.rand(1, 1, 16, 16)
    moving = torch.rand(1, 1, 16, 16)
    report = benchmark_vjp_tensors(model, fixed, moving, warmups=1, repeats=2)
    assert report["control_vertices"] == 81
    assert report["forward_vjp_median_seconds"] > 0
    assert report["peak_cuda_allocated_bytes"] is None
    assert report["input_gradients_finite"] is True
    assert report["model_gradients_finite"] is True
