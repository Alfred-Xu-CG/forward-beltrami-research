"""Independent finite differences for the held-out analytic QC evaluator."""

from __future__ import annotations

import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces  # noqa: E402
from phase6_train_multisample_image import make_dataset  # noqa: E402
from qcopt.mesh import structured_rectangle


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


def test_high32_target_sampled_p1_faces_remain_positive_at_amplitude_corners() -> None:
    mesh = structured_rectangle(256, 256)
    points = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)[None]
    coefficients = torch.tensor(
        [[0.015, 0.025, -0.002], [0.015, 0.025, 0.0025],
         [0.005, 0.010, -0.002], [0.005, 0.010, 0.0025]], dtype=torch.float64
    )
    mapped, _ = _target_on_faces(points, coefficients, fine_cycles=32)
    faces = torch.tensor(mesh.faces.copy())
    triangles = mapped[:, faces]
    first = triangles[:, :, 1] - triangles[:, :, 0]
    second = triangles[:, :, 2] - triangles[:, :, 0]
    areas = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    assert areas.min().item() * 256**2 > 0.01
    assert 2 * math.pi * math.hypot(0.015, 0.025) + 64 * math.pi * math.sqrt(2) * 0.0025 < 1.0


def test_high32_target_jacobian_matches_central_difference() -> None:
    points = torch.tensor([[[0.217, 0.344], [0.613, 0.776]]], dtype=torch.float64)
    coefficients = torch.tensor([[0.015, 0.025, 0.0025]], dtype=torch.float64)
    _, jacobian = _target_on_faces(points, coefficients, fine_cycles=32)
    step = 1e-7
    for direction in range(2):
        shift = torch.zeros_like(points)
        shift[..., direction] = step
        plus, _ = _target_on_faces(points + shift, coefficients, fine_cycles=32)
        minus, _ = _target_on_faces(points - shift, coefficients, fine_cycles=32)
        observed = (plus - minus) / (2 * step)
        assert torch.allclose(jacobian[..., :, direction], observed, rtol=1e-7, atol=1e-8)


def test_high32_synthesis_map_matches_independent_face_formula() -> None:
    _, _, synthesized, coefficients = make_dataset(3, 67, 55101, return_coefficients=True, target_family="high32")
    line = torch.linspace(0.0, 1.0, 67)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    points = torch.stack((xx, yy), dim=-1)[None].expand(3, -1, -1, -1).reshape(3, -1, 2)
    independent, _ = _target_on_faces(points, coefficients, fine_cycles=32)
    assert torch.allclose(synthesized.reshape(3, -1, 2), independent, atol=2e-7, rtol=0)
