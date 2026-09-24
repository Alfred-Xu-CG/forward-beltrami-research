"""An exact fixed-triangulation P1 identifiability counterexample."""

import torch


def test_nonconstant_single_image_has_two_distinct_safe_p1_maps() -> None:
    side = 65
    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    bump = xx * (1 - xx) * yy * (1 - yy)
    identity = torch.stack((xx, yy), dim=-1)
    alternative = torch.stack((xx, yy + bump), dim=-1)

    assert torch.equal(alternative[0], identity[0])
    assert torch.equal(alternative[-1], identity[-1])
    assert torch.equal(alternative[:, 0], identity[:, 0])
    assert torch.equal(alternative[:, -1], identity[:, -1])
    assert alternative[side // 2, side // 2, 1] - .5 == 1 / 16

    # For the SW--NE diagonal, each face's determinant is 1 plus the
    # interpolant's slope along a vertical grid edge.
    vertical_slope = (bump[1:] - bump[:-1]) * (side - 1)
    all_face_jacobians = 1 + vertical_slope
    assert all_face_jacobians.min() >= .75

    # The moving image I(x,y)=x is nonconstant.  Its exact continuous
    # pullback is x under either P1 map, not merely at grid vertices.
    assert torch.equal(identity[..., 0], alternative[..., 0])
