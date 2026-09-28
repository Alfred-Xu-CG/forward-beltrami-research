"""Independent small-grid contracts for the deployed Q1 map."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    SafeColoredQ1Relaxation,
    SafePatchQ1Pass,
    StaggeredPatchQ1Layer,
    q1_corner_determinants,
    q1_dyadic_refine,
    validate_q1_map,
)


def test_four_corners_detect_concavity_missed_by_sw_ne_p1() -> None:
    # SW, SE, NE, NW. Both old SW--NE P1 triangles have positive area,
    # but the Q1 Jacobian at SW is negative.
    a = torch.tensor((0.0, 0.0), dtype=torch.float64)
    b = torch.tensor((1.0, 0.0), dtype=torch.float64)
    c = torch.tensor((1.0, 1.0), dtype=torch.float64)
    d = torch.tensor((-1.0, -0.5), dtype=torch.float64)
    old_low = torch.linalg.det(torch.stack((b - a, c - a)))
    old_up = torch.linalg.det(torch.stack((c - a, d - a)))
    assert old_low > 0 and old_up > 0
    vertices = torch.stack((torch.stack((a, b)), torch.stack((d, c))))[None]
    corner = q1_corner_determinants(vertices)[0, 0, 0]
    assert torch.equal(corner, torch.tensor((-0.5, 1.0, 2.0, 0.5), dtype=torch.float64))


def test_q1_refinement_preserves_values_at_child_centers() -> None:
    # A non-parallelogram convex quad: Q1 center is the average of *four*
    # corners, unlike the existing SW--NE P1 center.
    a = torch.tensor((0.0, 0.0), dtype=torch.float64)
    b = torch.tensor((1.0, 0.0), dtype=torch.float64)
    c = torch.tensor((1.0, 1.0), dtype=torch.float64)
    d = torch.tensor((0.0, 0.8), dtype=torch.float64)
    vertices = torch.stack((torch.stack((a, b)), torch.stack((d, c))))[None]
    fine = q1_dyadic_refine(vertices)
    assert fine.shape == (1, 3, 3, 2)
    assert torch.equal(fine[0, 1, 1], (a + b + c + d) / 4)
    assert not torch.equal(fine[0, 1, 1], (a + c) / 2)
    assert bool((q1_corner_determinants(fine) > 0).all())


def test_colored_q1_layer_stops_center_before_concave_cell() -> None:
    axis = torch.linspace(0, 1, 3, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    bad = base.clone()
    bad[:, 1, 1] = torch.tensor((0.125, 0.125), dtype=base.dtype)
    assert q1_corner_determinants(bad).amin() < 0
    latent = torch.full((1, 1, 1, 2), -2.0, dtype=base.dtype, requires_grad=True)
    layer = SafeColoredQ1Relaxation(3, safety_fraction=0.75)
    output = layer(base, latent)
    assert q1_corner_determinants(output).amin() > 0
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.equal(output[:, -1], base[:, -1])
    output.sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_patch_q1_pass_moves_adjacent_vertices_without_folding() -> None:
    torch.manual_seed(92029)
    axis = torch.linspace(0, 1, 9, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    latent = (8 * torch.randn(1, 7, 7, 2, dtype=base.dtype)).requires_grad_()
    layer = SafePatchQ1Pass(9, 4, minimum_jacobian=0.01)
    output = layer(base, latent)
    assert q1_corner_determinants(output).amin() > 0
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.equal(output[:, -1], base[:, -1])
    assert torch.equal(output[:, :, 0], base[:, :, 0])
    assert torch.equal(output[:, :, -1], base[:, :, -1])
    assert (output - base).abs().amax() > 1e-4
    output.square().sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_staggered_q1_passes_reach_old_patch_seam_center() -> None:
    axis = torch.linspace(0, 1, 9, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    latent = torch.zeros(1, 7, 7, 2, dtype=base.dtype)
    latent[:, 3, 3, 0] = 0.5  # Global vertex (4,4), fixed in first three passes.
    layer = StaggeredPatchQ1Layer(9, 4, minimum_jacobian=0.01)
    current = base
    for index, patch_pass in enumerate(layer.passes):
        current = patch_pass(current, latent)
        assert q1_corner_determinants(current).amin() > 0
        if index < 3:
            assert torch.equal(current[:, 4, 4], base[:, 4, 4])
    assert current[0, 4, 4, 0] > base[0, 4, 4, 0]
    assert torch.equal(layer(base, (latent,) * 4), current)


def test_single_patch_q1_grid_does_not_require_impossible_shifted_patches() -> None:
    axis = torch.linspace(0, 1, 5, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    latent = torch.full((1, 3, 3, 2), 0.2, dtype=base.dtype)
    layer = StaggeredPatchQ1Layer(5, 4)
    assert len(layer.passes) == 1
    mapped = layer(base, (latent,))
    assert q1_corner_determinants(mapped).amin() > 0
    assert (mapped[:, 1:-1, 1:-1] - base[:, 1:-1, 1:-1]).abs().amax() > 0


def test_q1_final_validator_rejects_fold_boundary_change_and_nan() -> None:
    axis = torch.linspace(0, 1, 3, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    assert validate_q1_map(identity, identity, chunk_rows=1)["valid"]
    folded = identity.clone()
    folded[:, 1, 1] = torch.tensor((0.125, 0.125), dtype=identity.dtype)
    report = validate_q1_map(folded, identity, chunk_rows=1)
    assert not report["valid"] and report["nonpositive_corners"] > 0
    moved_boundary = identity.clone()
    moved_boundary[:, 0, 1, 0] += 0.01
    report = validate_q1_map(moved_boundary, identity)
    assert not report["valid"] and report["boundary_max_error"] > 0
    nonfinite = identity.clone()
    nonfinite[:, 1, 1, 0] = float("nan")
    report = validate_q1_map(nonfinite, identity)
    assert not report["valid"] and report["nonfinite_coordinates"] > 0
