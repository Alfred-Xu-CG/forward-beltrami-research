"""Original-grid P1 homeomorphism and VJP checks for colored latent motions."""

from __future__ import annotations

import torch

from phase6_train_multisample_image import ConvexQuadLocalImageEncoder
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import HierarchicalConvexQuadFreeCenterLayer, HierarchicalConvexQuadLocalLayer, SafeColoredVertexRelaxation


def _face_areas(control: torch.Tensor) -> torch.Tensor:
    side = control.shape[1]
    faces = torch.as_tensor(structured_rectangle(side - 1, side - 1).faces.copy(), dtype=torch.long)
    corners = control.reshape(control.shape[0], -1, 2)[:, faces]
    first = corners[:, :, 1] - corners[:, :, 0]
    second = corners[:, :, 2] - corners[:, :, 0]
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _random_base(side: int, dtype: torch.dtype) -> torch.Tensor:
    decoder = HierarchicalConvexQuadFreeCenterLayer(side)
    root = 0.25 * torch.randn(1, 1, 1, 2, dtype=dtype)
    levels = tuple(
        (0.25 * torch.randn(1, n, n - 1, dtype=dtype),
         0.25 * torch.randn(1, n - 1, n, dtype=dtype),
         0.25 * torch.randn(1, n - 1, n - 1, 2, dtype=dtype))
        for n in decoder.latent_sides
    )
    return decoder(root, levels)


def test_colored_relaxation_preserves_all_faces_and_boundary_for_large_latents() -> None:
    torch.manual_seed(3169)
    side = 17
    layer = SafeColoredVertexRelaxation(side, safety_fraction=0.85)
    base = _random_base(side, torch.float64)
    logits = 8 * torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64)
    output = layer(base, logits)
    assert torch.all(_face_areas(base) > 0)
    assert torch.all(_face_areas(output) > 0)
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.equal(output[:, -1], base[:, -1])
    assert torch.equal(output[:, :, 0], base[:, :, 0])
    assert torch.equal(output[:, :, -1], base[:, :, -1])


def test_arithmetic_opposite_edges_match_independent_mesh_faces() -> None:
    side = 17
    layer = SafeColoredVertexRelaxation(side)
    expected = {}
    for face in structured_rectangle(side - 1, side - 1).faces:
        for corner in range(3):
            vertex = int(face[corner])
            row, column = divmod(vertex, side)
            if 0 < row < side - 1 and 0 < column < side - 1:
                expected.setdefault(vertex, set()).add((int(face[(corner + 1) % 3]), int(face[(corner + 2) % 3])))
    observed = {}
    for color in range(4):
        vertices = getattr(layer, f"_vertices_{color}").tolist()
        edges = getattr(layer, f"_opposite_{color}").tolist()
        for vertex, vertex_edges in zip(vertices, edges):
            observed[vertex] = {tuple(edge) for edge in vertex_edges}
    assert observed == expected


def test_colored_relaxation_can_bend_one_parent_edge_without_composition() -> None:
    side = 9
    mesh = structured_rectangle(side - 1, side - 1)
    base = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float64).reshape(1, side, side, 2)
    logits = torch.zeros(1, side - 2, side - 2, 2, dtype=torch.float64)
    logits[0, 3, 2, 1] = 2.0
    output = SafeColoredVertexRelaxation(side)(base, logits)[0]
    left, midpoint, right = output[4, 2], output[4, 3], output[4, 4]
    cross = torch.linalg.det(torch.stack((midpoint - left, right - left)))
    assert abs(cross.item()) > 1e-4
    assert torch.all(_face_areas(output[None]) > 0)


def test_colored_relaxation_vjp_matches_directional_finite_difference() -> None:
    torch.manual_seed(40177)
    side = 9
    base = _random_base(side, torch.float64).detach()
    layer = SafeColoredVertexRelaxation(side)
    logits = (0.2 * torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64)).requires_grad_()
    cotangent = torch.randn_like(base)
    direction = torch.randn_like(logits)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return (layer(base, value) * cotangent).sum()

    gradient = torch.autograd.grad(objective(logits), logits)[0]
    predicted = (gradient * direction).sum()
    step = 1e-6
    observed = (objective(logits + step * direction) - objective(logits - step * direction)) / (2 * step)
    assert torch.allclose(predicted, observed, rtol=2e-4, atol=2e-5)


def test_image_encoder_to_local_latent_has_gradient_and_valid_output() -> None:
    torch.manual_seed(5103)
    side = 17
    encoder = ConvexQuadLocalImageEncoder(side, width=8, head_mode="multilevel", body_mode="local")
    decoder = HierarchicalConvexQuadLocalLayer(side)
    pair = torch.randn(2, 2, 32, 32)
    output = decoder(*encoder(pair))
    assert output.shape == (2, side, side, 2)
    assert torch.all(_face_areas(output) > 0)
    (output.square().mean()).backward()
    assert encoder.local_head.weight.grad is not None
    assert torch.isfinite(encoder.local_head.weight.grad).all()
    assert encoder.local_head.weight.grad.abs().sum() > 0
