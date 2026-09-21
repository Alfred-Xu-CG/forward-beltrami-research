from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as sparse_linalg
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.decoder import TutteRectangleDecoder
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target
from qcopt.neural_bijection.tutte import symmetric as symmetric_backend
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def _case(nx: int = 4, ny: int = 3):
    mesh = structured_rectangle(nx, ny)
    layer = MatrixFreeSymmetricTutteLayer(
        mesh,
        relative_tolerance=2e-13,
        absolute_tolerance=1e-14,
        max_iterations=500,
    )
    generator = torch.Generator().manual_seed(1917)
    logits = torch.randn(layer.n_conductances, generator=generator, dtype=torch.float64) * 0.45
    count = len(mesh.boundary_loops[0])
    angles = torch.arange(count, dtype=torch.float64) * (2.0 * torch.pi / count)
    boundary = torch.stack((1.3 * torch.cos(angles), 0.8 * torch.sin(angles)), dim=-1)
    return mesh, layer, logits, boundary


def _dense_reference(mesh, layer, conductances, boundary):
    interior = layer.interior_vertices
    loop = layer.boundary_vertices
    interior_index = {int(vertex): row for row, vertex in enumerate(interior)}
    boundary_index = {int(vertex): row for row, vertex in enumerate(loop)}
    matrix = np.zeros((len(interior), len(interior)), dtype=np.float64)
    rhs = np.zeros((len(interior), 2), dtype=np.float64)
    for edge, value in zip(layer.active_edges, conductances.detach().numpy()):
        first, second = map(int, edge)
        for vertex, neighbor in ((first, second), (second, first)):
            if vertex not in interior_index:
                continue
            row = interior_index[vertex]
            matrix[row, row] += value
            if neighbor in interior_index:
                matrix[row, interior_index[neighbor]] -= value
            else:
                rhs[row] += value * boundary[boundary_index[neighbor]].detach().numpy()
    interior_values = np.linalg.solve(matrix, rhs)
    result = np.empty((mesh.n_vertices, 2), dtype=np.float64)
    result[loop] = boundary.detach().numpy()
    result[interior] = interior_values
    return result, matrix


def test_uniform_conductance_and_identity_boundary_recover_identity() -> None:
    mesh = structured_rectangle(6, 4)
    layer = MatrixFreeSymmetricTutteLayer(mesh, relative_tolerance=1e-13)
    logits = torch.zeros(layer.n_conductances, dtype=torch.float64)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)

    output = layer(logits, boundary)

    torch.testing.assert_close(
        output,
        torch.tensor(mesh.vertices, dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )
    assert layer.last_forward_stats.converged
    assert layer.last_forward_stats.relative_residual < 1e-12


def test_matrix_free_solution_matches_independent_dense_spd_reference() -> None:
    mesh, layer, logits, boundary = _case()
    conductances = layer.conductances(logits)
    expected, matrix = _dense_reference(mesh, layer, conductances, boundary)

    output = layer(logits, boundary)

    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0
    np.testing.assert_allclose(output.detach(), expected, atol=3e-11, rtol=2e-11)


def test_matrix_free_edge_operator_is_self_adjoint() -> None:
    _, layer, logits, _ = _case(5, 4)
    conductances = layer.conductances(logits).unsqueeze(0)
    generator = torch.Generator().manual_seed(17)
    first = torch.randn((1, layer.n_interior, 3), generator=generator, dtype=torch.float64)
    second = torch.randn((1, layer.n_interior, 3), generator=generator, dtype=torch.float64)

    lhs = (layer.apply_interior(conductances, first) * second).sum()
    rhs = (first * layer.apply_interior(conductances, second)).sum()

    torch.testing.assert_close(lhs, rhs, atol=2e-13, rtol=2e-13)


def test_symmetric_logits_and_boundary_pass_gradcheck() -> None:
    _, layer, logits, boundary = _case(3, 2)
    logits.requires_grad_()
    boundary.requires_grad_()

    assert torch.autograd.gradcheck(
        layer,
        (logits, boundary),
        eps=1e-6,
        atol=3e-6,
        rtol=3e-5,
    )


def test_batch_broadcast_and_gradients_match_independent_calls() -> None:
    _, layer, logits, boundary = _case(3, 3)
    batched_logits = torch.stack((logits, 0.7 * logits)).requires_grad_()
    shared_boundary = boundary.clone().requires_grad_()

    batched = layer(batched_logits, shared_boundary)
    separate = torch.stack([layer(item, shared_boundary) for item in batched_logits])
    gradients = torch.autograd.grad(batched.square().sum(), (batched_logits, shared_boundary))
    expected_gradients = torch.autograd.grad(
        separate.square().sum(), (batched_logits, shared_boundary)
    )

    torch.testing.assert_close(batched, separate, atol=2e-11, rtol=2e-11)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)


def test_forward_backward_never_call_sparse_factorization(monkeypatch) -> None:
    _, layer, logits, boundary = _case()

    def forbidden(*args, **kwargs):
        pytest.fail("matrix-free symmetric layer called scipy sparse factorization")

    monkeypatch.setattr(sparse_linalg, "splu", forbidden)
    logits.requires_grad_()
    boundary.requires_grad_()
    layer(logits, boundary).square().sum().backward()

    assert layer.last_forward_stats.iterations > 0
    assert layer.last_adjoint_stats.iterations > 0


def test_saved_autograd_state_does_not_scale_with_krylov_iterations() -> None:
    _, layer, logits, boundary = _case(5, 5)
    saved_shapes: list[tuple[int, ...]] = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = layer(logits.requires_grad_(), boundary.requires_grad_())
        output.square().mean().backward()

    state_limit = max(
        layer.n_conductances,
        layer.n_interior * 2,
        len(layer.boundary_vertices) * 2,
        layer.system.n_vertices * 2,
    )
    assert saved_shapes
    assert all(np.prod(shape, dtype=int) <= state_limit for shape in saved_shapes)
    assert all(len(shape) <= 3 for shape in saved_shapes)


def test_zero_rhs_column_does_not_force_true_residual_matvec_each_iteration(monkeypatch) -> None:
    mesh = structured_rectangle(10, 10)
    layer = MatrixFreeSymmetricTutteLayer(
        mesh,
        relative_tolerance=1.0e-12,
        absolute_tolerance=0.0,
        max_iterations=1000,
    )
    generator = torch.Generator().manual_seed(721)
    conductances = torch.exp(
        0.7 * torch.randn((1, layer.n_conductances), generator=generator, dtype=torch.float64)
    )
    rhs = torch.randn((1, layer.n_interior, 1), generator=generator, dtype=torch.float64)
    original_apply = layer._apply_batched
    calls = 0

    def counted_apply(values, vectors):
        nonlocal calls
        calls += 1
        return original_apply(values, vectors)

    monkeypatch.setattr(layer, "_apply_batched", counted_apply)
    expected, _ = symmetric_backend._conjugate_gradient(layer, conductances, rhs)
    baseline_calls = calls
    calls = 0
    augmented, _ = symmetric_backend._conjugate_gradient(
        layer,
        conductances,
        torch.cat((rhs, torch.zeros_like(rhs)), dim=2),
    )

    torch.testing.assert_close(augmented[..., :1], expected)
    assert calls == baseline_calls


def test_nonconvergence_is_fail_closed() -> None:
    mesh = structured_rectangle(8, 8)
    layer = MatrixFreeSymmetricTutteLayer(
        mesh,
        relative_tolerance=1e-14,
        absolute_tolerance=0.0,
        max_iterations=1,
    )
    logits = torch.linspace(-3.0, 3.0, layer.n_conductances, dtype=torch.float64)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="converge"):
        layer(logits, boundary)


def test_float64_output_inplace_mutation_cannot_change_saved_primal_gradient() -> None:
    _, layer, logits, boundary = _case(3, 2)
    cotangent = torch.arange(layer.system.n_vertices * 2, dtype=torch.float64).reshape(-1, 2).cos()

    clean_logits = logits.clone().requires_grad_()
    clean_boundary = boundary.clone().requires_grad_()
    clean = layer(clean_logits, clean_boundary)
    clean_gradients = torch.autograd.grad((2.0 * clean * cotangent).sum(), (clean_logits, clean_boundary))

    mutated_logits = logits.clone().requires_grad_()
    mutated_boundary = boundary.clone().requires_grad_()
    mutated = layer(mutated_logits, mutated_boundary)
    mutated.mul_(2.0)
    mutated_gradients = torch.autograd.grad((mutated * cotangent).sum(), (mutated_logits, mutated_boundary))

    for actual, expected in zip(mutated_gradients, clean_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize(
    "dtype,scale,tolerance",
    ((torch.float64, 1.0e200, 2.0e-12), (torch.float32, 1.0e25, 2.0e-5)),
)
def test_global_conductance_scale_cannot_overflow_cg_convergence_test(
    dtype: torch.dtype,
    scale: float,
    tolerance: float,
) -> None:
    mesh = structured_rectangle(2, 2)
    layer = MatrixFreeSymmetricTutteLayer(mesh, max_iterations=50)
    logits = torch.linspace(scale, 2.0 * scale, layer.n_conductances, dtype=dtype)
    count = len(mesh.boundary_loops[0])
    angles = torch.arange(count, dtype=dtype) * (2.0 * torch.pi / count)
    boundary = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    conductances = layer.conductances(logits)
    weights = conductances.index_select(0, layer._ib_edges)
    expected_interior = (
        weights[:, None] * boundary.index_select(0, layer._ib_boundary)
    ).sum(dim=0) / weights.sum()

    output = layer(logits, boundary)

    torch.testing.assert_close(
        output[layer.interior_vertices[0]],
        expected_interior,
        atol=tolerance,
        rtol=tolerance,
    )
    assert layer.last_forward_stats.iterations > 0
    assert layer.last_forward_stats.converged


@pytest.mark.parametrize(
    "dtype,loss_scale,tolerance",
    ((torch.float64, 1.0e200, 3.0e-12), (torch.float32, 1.0e25, 3.0e-4)),
)
def test_large_finite_adjoint_rhs_is_scaled_without_zero_gradient(
    dtype: torch.dtype,
    loss_scale: float,
    tolerance: float,
) -> None:
    mesh = structured_rectangle(2, 2)
    layer = MatrixFreeSymmetricTutteLayer(mesh, max_iterations=50)
    logits = torch.linspace(-0.3, 0.7, layer.n_conductances, dtype=dtype)
    count = len(mesh.boundary_loops[0])
    angles = torch.arange(count, dtype=dtype) * (2.0 * torch.pi / count)
    boundary = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    baseline_logits = logits.clone().requires_grad_()
    baseline_boundary = boundary.clone().requires_grad_()
    baseline_output = layer(baseline_logits, baseline_boundary)
    baseline = torch.autograd.grad(
        baseline_output[layer.interior_vertices].sum(),
        (baseline_logits, baseline_boundary),
    )

    large_logits = logits.clone().requires_grad_()
    large_boundary = boundary.clone().requires_grad_()
    large_output = layer(large_logits, large_boundary)
    large = torch.autograd.grad(
        large_output[layer.interior_vertices].sum() * loss_scale,
        (large_logits, large_boundary),
    )

    for actual, expected in zip(large, baseline):
        torch.testing.assert_close(
            actual / loss_scale,
            expected,
            atol=tolerance,
            rtol=tolerance,
        )
    assert layer.last_adjoint_stats.converged
    assert layer.last_adjoint_stats.iterations > 0


def test_boundary_only_mesh_needs_no_conductance_reduction_or_cg() -> None:
    mesh = structured_rectangle(1, 1)
    layer = MatrixFreeSymmetricTutteLayer(mesh)
    logits = torch.empty(0, dtype=torch.float64, requires_grad=True)
    boundary = torch.tensor(
        mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64, requires_grad=True
    )

    output = layer(logits, boundary)
    output.sum().backward()

    torch.testing.assert_close(output, torch.tensor(mesh.vertices, dtype=torch.float64))
    torch.testing.assert_close(boundary.grad, torch.ones_like(boundary))
    assert logits.grad is not None and logits.grad.numel() == 0
    assert layer.last_forward_stats.iterations == 0
    assert layer.last_adjoint_stats.iterations == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_float32_cuda_medium_dense_loss_has_reliable_primal_and_adjoint_residuals() -> None:
    mesh = structured_rectangle(16, 16)
    target = build_directed_target(
        mesh,
        image_height=256,
        image_width=256,
        seed=20260922,
        strength=0.25,
        height=0.9,
    )
    # Preserve the original failure threshold as a targeted reliable-residual
    # regression even though the production float32 default is more tolerant.
    layer = MatrixFreeSymmetricTutteLayer(mesh, relative_tolerance=2.0e-6).cuda()
    decoder = TutteRectangleDecoder(mesh, layer, image_height=256, image_width=256).cuda()
    decoder.prepare(device="cuda", dtype=torch.float32)
    logits = torch.zeros(layer.n_conductances, device="cuda", requires_grad=True)
    boundary_logits = torch.zeros(
        decoder.boundary.n_segments, device="cuda", requires_grad=True
    )
    raw_modulus = torch.tensor(
        decoder.boundary.raw_modulus_for_height(1.0),
        device="cuda",
        requires_grad=True,
    )

    decoded = decoder(logits, boundary_logits, raw_modulus)
    loss = (decoded.dense - target.dense.float().cuda()).square().mean()
    loss.backward()

    assert layer.last_forward_stats.converged
    assert layer.last_forward_stats.relative_residual <= 2.0e-6
    assert layer.last_adjoint_stats.converged
    assert layer.last_adjoint_stats.relative_residual <= 2.0e-6
    assert torch.isfinite(logits.grad).all()
