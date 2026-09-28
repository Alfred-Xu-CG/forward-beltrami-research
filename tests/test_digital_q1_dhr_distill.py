"""Tiny safe-decoder distillation fixture; teacher is not assumed topology-safe."""

from __future__ import annotations

import torch

from tools.digital_q1_dhr_distill import (
    _load_target_archive,
    _saved_output_summary,
    factor_affine_teacher,
    fit_target_vertices,
)


def test_fit_target_vertices_reduces_interior_error_and_keeps_boundary() -> None:
    axis = torch.linspace(0, 1, 17)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((x, y), dim=-1)[None]
    bubble = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    target = identity + torch.stack((.015 * bubble, -.01 * bubble), dim=-1)[None]
    mapped, result = fit_target_vertices(
        target, steps=8, learning_rate=.04, device="cpu",
    )
    assert result["best_interior_vertex_rmse"] < result["initial_interior_vertex_rmse"]
    assert result["finite_gradient_steps"] == 8
    assert result["nonpositive_corners"] == 0
    assert result["boundary_max_error"] == 0
    torch.testing.assert_close(mapped[:, 0], identity[:, 0])


def test_affine_factorization_recovers_positive_shear_and_unit_residual() -> None:
    axis = torch.linspace(0, 1, 17)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((x, y), dim=-1)[None]
    matrix = torch.tensor([[.94, .14], [-.07, .96]])
    offset = torch.tensor([-.03, .02])
    teacher = identity @ matrix.T + offset
    residual, fitted_matrix, fitted_offset = factor_affine_teacher(teacher, identity)
    assert torch.linalg.det(fitted_matrix) > 0
    torch.testing.assert_close(fitted_matrix, matrix, atol=2e-6, rtol=0)
    torch.testing.assert_close(fitted_offset, offset, atol=2e-6, rtol=0)
    torch.testing.assert_close(residual, identity, atol=2e-6, rtol=0)


def test_affine_factorization_rejects_reflection() -> None:
    axis = torch.linspace(0, 1, 17)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((x, y), dim=-1)[None]
    reflection = identity.clone()
    reflection[..., 0] = 1 - reflection[..., 0]
    import pytest
    with pytest.raises(ValueError, match="positive orientation"):
        factor_affine_teacher(reflection, identity)


def test_affine_fit_log_does_not_mislabel_residual_certificate_as_full_map() -> None:
    report = _saved_output_summary(
        {"valid": True, "nonpositive_corners": 0},
        affine_matrix=torch.tensor([[.94, .14], [-.07, .96]]).numpy(),
    )
    assert report["saved_binary_residual_valid"]
    assert report["saved_object_is_affine_postcomposition"]
    assert "saved_binary_valid" not in report
    assert "saved_composite_representation_valid" not in report


def test_target_archive_rejects_half_present_affine_metadata(tmp_path) -> None:
    import numpy as np
    import pytest
    path = tmp_path / "partial_affine.npz"
    np.savez_compressed(path, teacher_vertices=np.zeros((1, 17, 17, 2), dtype=np.float32),
                        post_affine_matrix=np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="together"):
        _load_target_archive(path)


def test_target_archive_rejects_orientation_reversing_affine(tmp_path) -> None:
    import numpy as np
    import pytest
    path = tmp_path / "negative_affine.npz"
    np.savez_compressed(path, teacher_vertices=np.zeros((1, 17, 17, 2), dtype=np.float32),
                        post_affine_matrix=np.diag([-1., 1.]).astype(np.float32),
                        post_affine_offset=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="positive orientation"):
        _load_target_archive(path)
