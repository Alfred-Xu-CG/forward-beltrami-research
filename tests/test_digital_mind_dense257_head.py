"""True dense 257² safe head zero baseline and first-order gradient."""

import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_mind_dense257_head import DenseSafeHead, dense_features
from tools.digital_q1_dhr_distill import identity_vertices


def test_dense_head_zero_identity_then_nonzero_finite_vjp():
    torch.manual_seed(4)
    coarse = identity_vertices(257, device=torch.device("cpu"))
    features = torch.randn((1, 88, 256, 256))
    head = DenseSafeHead(width=8)
    zero = head(coarse, features)
    torch.testing.assert_close(zero, coarse, rtol=0, atol=0)
    assert validate_q1_map(zero, coarse)["valid"]
    with torch.no_grad():
        head.encoder[-1].weight.fill_(.001)
    moved = head(coarse, features)
    assert validate_q1_map(moved, coarse)["valid"]
    assert torch.any(moved != coarse)
    moved.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters())


def test_pixel_center_geometry_identity_has_zero_displacement():
    base = torch.zeros((1, 43, 128, 128))
    image = torch.zeros((1, 1, 512, 512))
    identity = identity_vertices(257, device=torch.device("cpu"))
    corrected, *_ = dense_features(base, image, image, identity,
                                   feature_geometry="pixel_center")
    legacy, *_ = dense_features(base, image, image, identity,
                                feature_geometry="legacy_align_corners")
    assert torch.max(torch.abs(corrected[:, 84:86])) < 1e-6
    assert torch.max(torch.abs(legacy[:, 84:86])) > 1e-3


def test_pixel_center_feature_uses_fixed_diagonal_p1_not_q1():
    base = torch.zeros((1, 43, 128, 128))
    image = torch.zeros((1, 1, 512, 512))
    vertices = identity_vertices(257, device=torch.device("cpu")).clone()
    vertices[:, 0, 1, 1] += .002
    corrected, *_ = dense_features(base, image, image, vertices,
                                   feature_geometry="pixel_center")
    # The first cell center is exactly on the (0,0)--(1,1) P1 edge;
    # perturbing its other two corners must not change this sampled value.
    assert abs(float(corrected[0, 1, 0, 0])) < 1e-6
