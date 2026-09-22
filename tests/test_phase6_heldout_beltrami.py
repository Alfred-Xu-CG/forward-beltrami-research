"""Independent finite differences for the held-out analytic QC evaluator."""

from __future__ import annotations

import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces  # noqa: E402


def test_target_jacobian_matches_independent_central_difference() -> None:
    points = torch.tensor([[[0.21, 0.34], [0.61, 0.77]]], dtype=torch.float64)
    coefficients = torch.tensor([[0.024, 0.043, -0.0015]], dtype=torch.float64)
    _, jacobian = _target_on_faces(points, coefficients)
    step = 1e-6
    for direction in range(2):
        shift = torch.zeros_like(points)
        shift[..., direction] = step
        plus, _ = _target_on_faces(points + shift, coefficients)
        minus, _ = _target_on_faces(points - shift, coefficients)
        observed = (plus - minus) / (2 * step)
        assert torch.allclose(jacobian[..., :, direction], observed, rtol=1e-9, atol=1e-9)


def test_beltrami_identity_and_positive_shear() -> None:
    identity = torch.eye(2, dtype=torch.float64)[None]
    assert _mu(identity).abs().max().item() == 0.0
    shear = torch.tensor([[[1.0, 0.4], [0.0, 1.0]]], dtype=torch.float64)
    coefficient = _mu(shear)
    assert coefficient.abs().item() < 1.0
    assert math.isclose(coefficient.abs().item(), 0.4 / math.sqrt(4.0 + 0.4**2), rel_tol=1e-14)
