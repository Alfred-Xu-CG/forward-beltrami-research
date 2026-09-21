"""Matrix-free directed solves: numerical, adjoint, and storage contracts."""

import importlib
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer


def module():
    name = "qcopt.neural_bijection.tutte.iterative"
    assert importlib.util.find_spec(name) is not None, "matrix-free backend is missing"
    return importlib.import_module(name)


def make_case(nx=4, ny=3, **kwargs):
    mesh = structured_rectangle(nx, ny)
    layer = module().MatrixFreeDirectedTutteLayer(mesh, **kwargs)
    rng = np.random.default_rng(709)
    z = torch.tensor(rng.normal(scale=0.7, size=(layer.system.n_rows, layer.system.max_degree)), dtype=torch.float64)
    angle = np.arange(len(mesh.boundary_loops[0])) * 2 * np.pi / len(mesh.boundary_loops[0])
    b = torch.tensor(np.column_stack((1.3 * np.cos(angle), 0.9 * np.sin(angle))), dtype=torch.float64)
    return mesh, layer, z, b


def test_matvec_matches_dense_and_transpose_adjoint_identity():
    _, layer, z, _ = make_case()
    s = layer.system
    p = torch.softmax(z, -1).unsqueeze(0).repeat(2, 1, 1)
    matrix = torch.eye(s.n_rows, dtype=z.dtype).repeat(2, 1, 1)
    for i in range(s.n_rows):
        for slot in np.flatnonzero(s.valid_mask[i] & ~s.neighbor_is_boundary[i]):
            matrix[:, i, s.neighbors[i, slot]] -= p[:, i, slot]
    assert (matrix - matrix.transpose(1, 2)).norm() > 0.1
    x = torch.arange(2 * s.n_rows * 2, dtype=z.dtype).sin().reshape(2, s.n_rows, 2)
    y = torch.cos(x * 3)
    torch.testing.assert_close(layer.matvec(p, x), matrix @ x)
    torch.testing.assert_close(layer.matvec(p, y, transpose=True), matrix.transpose(1, 2) @ y)
    torch.testing.assert_close((y * layer.matvec(p, x)).sum(), (layer.matvec(p, y, transpose=True) * x).sum())


def test_direct_output_and_probability_boundary_vjp_parity():
    mesh, layer, z, b = make_case(rtol=1e-12, atol=1e-14)
    z.requires_grad_(); b.requires_grad_()
    reference = DirectTutteLayer(mesh)
    out = layer(z, b)
    exact = reference(z, b)
    torch.testing.assert_close(out, exact, atol=2e-11, rtol=2e-11)
    cotangent = torch.arange(out.numel(), dtype=z.dtype).cos().reshape_as(out)
    observed = torch.autograd.grad((out * cotangent).sum(), (z, b))
    expected = torch.autograd.grad((exact * cotangent).sum(), (z, b))
    for actual, target in zip(observed, expected):
        torch.testing.assert_close(actual, target, atol=3e-10, rtol=3e-9)


def test_nonsymmetric_gradcheck_and_directional_finite_difference():
    _, layer, z, b = make_case(3, 2, rtol=1e-13, atol=1e-14)
    z.requires_grad_(); b.requires_grad_()
    assert torch.autograd.gradcheck(layer, (z, b), eps=1e-6, atol=2e-6, rtol=2e-5)
    out = layer(z, b)
    cotangent = torch.arange(out.numel(), dtype=z.dtype).sin().reshape_as(out)
    gz, gb = torch.autograd.grad((out * cotangent).sum(), (z, b))
    dz, db = torch.cos(z), torch.sin(b)
    eps = 1e-6
    fd = ((layer(z + eps * dz, b + eps * db) - layer(z - eps * dz, b - eps * db)) * cotangent).sum() / (2 * eps)
    torch.testing.assert_close(fd, (gz * dz).sum() + (gb * db).sum(), atol=2e-7, rtol=2e-6)


def test_batch_broadcast_gradients():
    mesh, layer, z, b = make_case(rtol=1e-12, atol=1e-14)
    z = torch.stack((z, z * 0.4)).requires_grad_()
    b.requires_grad_()
    y = layer(z, b)
    target = DirectTutteLayer(mesh)(z, b)
    torch.testing.assert_close(y, target, atol=3e-11, rtol=3e-11)
    actual = torch.autograd.grad(y.square().sum(), (z, b))
    expected = torch.autograd.grad(target.square().sum(), (z, b))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)
    torch.testing.assert_close(layer(z[0].detach(), torch.stack((b.detach(), b.detach() * 2))),
                               torch.stack((layer(z[0].detach(), b.detach()), layer(z[0].detach(), b.detach() * 2))))


def test_residuals_iterations_and_per_call_diagnostic_ownership():
    _, layer, z, b = make_case(rtol=1e-11, atol=0)
    z.requires_grad_(); b.requires_grad_()
    out1 = layer(z, b)
    report1 = layer.last_diagnostics
    out2 = layer(z * 0.7, b)
    report2 = layer.last_diagnostics
    assert report1 is not report2
    assert report1.adjoint is None
    out1.square().sum().backward()
    assert layer.last_diagnostics is report2 and report2.adjoint is None
    for report in (report1.forward, report1.adjoint):
        assert report.iterations.shape == (1, 2)
        assert torch.all(report.iterations > 0)
        assert torch.all(report.iterations <= layer.max_iter)
        assert report.relative_residual.max() <= 1e-11
        assert not report.relative_residual.requires_grad
    out2.sum().backward()
    assert report2.adjoint is not None


def test_no_sparse_assembly_or_factorization_and_no_saved_krylov_history(monkeypatch):
    _, layer, z, b = make_case(6, 5, rtol=1e-11, atol=1e-14)
    backend = module()
    solver = backend._bicgstab
    execution_grad_modes = []
    def observed(*args, **kwargs):
        execution_grad_modes.append(torch.is_grad_enabled())
        return solver(*args, **kwargs)
    monkeypatch.setattr(backend, "_bicgstab", observed)
    def forbidden(*args, **kwargs):
        pytest.fail("matrix-free backend attempted sparse assembly/factorization")
    for name in ("csr_matrix", "csc_matrix", "coo_matrix", "lil_matrix"):
        monkeypatch.setattr(sparse, name, forbidden)
    for name in ("splu", "spsolve", "factorized"):
        monkeypatch.setattr(sla, name, forbidden)
    z = z.unsqueeze(0).requires_grad_()
    b = b.unsqueeze(0).requires_grad_()
    output = layer(z, b)
    saved = output.grad_fn.saved_tensors
    assert [tuple(t.shape) for t in saved] == [tuple(z.shape), tuple(output.shape)]
    assert sum(t.numel() for t in saved) == z.numel() + output.numel()
    output.square().sum().backward()
    assert execution_grad_modes == [False, False]


def test_inplace_output_does_not_mutate_backward_primal():
    _, layer, z, b = make_case(rtol=1e-12, atol=1e-14)
    z = z.unsqueeze(0).requires_grad_(); b = b.unsqueeze(0).requires_grad_()
    expected = torch.autograd.grad((2 * layer(z, b)).sum(), (z, b))
    out = layer(z, b)
    out.mul_(2)
    actual = torch.autograd.grad(out.sum(), (z, b))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=2e-10, rtol=2e-9)


def test_nonconvergence_is_not_accepted():
    _, layer, z, b = make_case(7, 6, max_iter=1, rtol=1e-13, atol=0)
    with pytest.raises(module().KrylovConvergenceError, match="converg"):
        layer(z, b)
    assert layer.last_diagnostics.forward is not None


def test_breakdown_is_explicit():
    backend = module()
    with pytest.raises(backend.KrylovConvergenceError, match="breakdown"):
        backend._bicgstab(lambda x: torch.zeros_like(x), torch.ones((2, 4, 2), dtype=torch.float64),
                          rtol=1e-12, atol=0, max_iter=5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e300])
def test_nonfinite_rhs_or_residual_scale_cannot_look_converged(value):
    backend = module()
    rhs = torch.full((1, 3, 2), value, dtype=torch.float64)
    with pytest.raises(backend.KrylovConvergenceError, match="nonfinite"):
        backend._bicgstab(lambda x: x, rhs, rtol=1e-12, atol=0, max_iter=5)


def test_underflowed_norm_is_not_zero_rhs_convergence():
    backend = module()
    rhs = torch.full((1, 3, 2), 1e-300, dtype=torch.float64)
    with pytest.raises(backend.KrylovConvergenceError, match="underflow") as caught:
        backend._bicgstab(lambda x: x, rhs, rtol=1e-12, atol=0, max_iter=5)
    assert not caught.value.report.converged.any()


@pytest.mark.parametrize("dtype,scale", [
    (torch.float64, 1e-160), (torch.float64, 1e-161),
    (torch.float32, 1e-20), (torch.float32, 1e-22),
    (torch.float64, 1e160), (torch.float64, 1e161),
    (torch.float32, 1e20), (torch.float32, 1e22),
])
def test_scaled_true_residual_or_explicit_failure(dtype, scale):
    backend = module()
    matrix = torch.tensor([[1, -.2, 0], [-.3, 1, -.1], [0, -.25, 1]], dtype=dtype)
    rhs = (scale * torch.tensor([1, .7, 1.3], dtype=dtype)).reshape(1, 3, 1)
    rtol = 1e-11 if dtype == torch.float64 else 5e-6
    try:
        x, report = backend._bicgstab(lambda x: matrix @ x, rhs, rtol=rtol, atol=0, max_iter=30)
    except backend.KrylovConvergenceError as error:
        assert not error.report.converged.all()
        return  # An explicit numerical range rejection is permitted, not false success.
    scaled_rhs = rhs.double() / scale
    scaled_residual = scaled_rhs - matrix.double() @ (x.double() / scale)
    independent_relative = scaled_residual.norm() / scaled_rhs.norm()
    assert independent_relative <= rtol * 1.1
    assert report.relative_residual.max() <= rtol


@pytest.mark.parametrize("dtype,scale", [(torch.float64, 1e-153), (torch.float32, 1e-18)])
def test_dot_guard_remains_active_after_initial_iteration(dtype, scale):
    backend = module()
    matrix = torch.tensor([[1, -.2, 0], [-.3, 1, -.1], [0, -.25, 1]], dtype=dtype)
    rhs = (scale * torch.tensor([1, .7, 1.3], dtype=dtype)).reshape(1, 3, 1)
    with pytest.raises(backend.KrylovConvergenceError, match="underflow") as caught:
        backend._bicgstab(lambda x: matrix @ x, rhs,
                          rtol=1e-11 if dtype == torch.float64 else 5e-6, atol=0, max_iter=30)
    # The initial scale is representable: rejection occurs in a later Krylov
    # recurrence, demonstrating this is not merely an initial-RHS guard.
    assert caught.value.report.iterations.min() >= 1
    assert not caught.value.report.converged.any()
    assert caught.value.report.relative_residual.min() > 0.01


@pytest.mark.parametrize("dtype,scale", [
    (torch.float64, 1e-150), (torch.float64, 1e150),
    (torch.float32, 1e-15), (torch.float32, 1e15),
])
def test_representable_extreme_scales_still_solve(dtype, scale):
    backend = module()
    matrix = torch.tensor([[1, -.2, 0], [-.3, 1, -.1], [0, -.25, 1]], dtype=dtype)
    rhs = (scale * torch.tensor([1, .7, 1.3], dtype=dtype)).reshape(1, 3, 1)
    rtol = 1e-11 if dtype == torch.float64 else 5e-6
    x, report = backend._bicgstab(lambda x: matrix @ x, rhs, rtol=rtol, atol=0, max_iter=30)
    scaled_rhs = rhs.double() / scale
    safe_relative = (scaled_rhs - matrix.double() @ (x.double() / scale)).norm() / scaled_rhs.norm()
    assert safe_relative <= rtol
    assert report.converged.all() and report.relative_residual.max() <= rtol


def test_adjoint_nonconvergence_propagates_to_caller():
    mesh, layer, z, _ = make_case(3, 3, max_iter=2, rtol=1e-12, atol=0)
    z = torch.zeros_like(z, requires_grad=True)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]] - 0.5, requires_grad=True)
    output = layer(z, boundary)  # Symmetric affine RHS needs only two iterations.
    assert layer.last_diagnostics.forward.converged.all()
    cotangent = torch.arange(output.numel(), dtype=z.dtype).sin().reshape_as(output)
    with pytest.raises(module().KrylovConvergenceError, match="converg"):
        (output * cotangent).sum().backward()
    assert not layer.last_diagnostics.adjoint.converged.all()


def test_raw_probability_vjp_matches_differentiable_dense_solve():
    _, layer, z, boundary = make_case(3, 3, rtol=1e-13, atol=1e-14)
    backend, s = module(), layer.system
    p = torch.softmax(z, -1).unsqueeze(0).detach().requires_grad_()
    b = boundary.unsqueeze(0).requires_grad_()
    output = backend._ImplicitMatrixFree.apply(
        p, b, layer._operator(), dict(rtol=1e-13, atol=1e-14, max_iter=100), backend.SolveDiagnostics())
    matrix = torch.eye(s.n_rows, dtype=p.dtype)
    rhs = torch.zeros((s.n_rows, 2), dtype=p.dtype)
    for row in range(s.n_rows):
        for slot in np.flatnonzero(s.valid_mask[row]):
            neighbor = s.neighbors[row, slot]
            if s.neighbor_is_boundary[row, slot]:
                rhs[row] = rhs[row] + p[0, row, slot] * b[0, neighbor]
            else:
                matrix[row, neighbor] = matrix[row, neighbor] - p[0, row, slot]
    exact = b.new_empty((1, s.n_vertices, 2))
    exact[:, s.loop.copy()] = b
    exact[:, s.interior.copy()] = torch.linalg.solve(matrix, rhs)
    weights = torch.arange(output.numel(), dtype=p.dtype).cos().reshape_as(output)
    actual = torch.autograd.grad((output * weights).sum(), (p, b))
    expected = torch.autograd.grad((exact * weights).sum(), (p, b))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-9)


def test_zero_rhs_columns_and_empty_interior():
    backend = module()
    rhs = torch.ones((3, 7, 2), dtype=torch.float64)
    rhs[:, :, 1] = 0
    x, report = backend._bicgstab(lambda x: 2 * x, rhs, rtol=1e-12, atol=0, max_iter=4)
    torch.testing.assert_close(x, rhs / 2)
    assert torch.all(report.iterations[:, 0] == 1)
    assert torch.all(report.iterations[:, 1] == 0)
    assert torch.all(report.relative_residual == 0)
    mesh = structured_rectangle(1, 1)
    layer = backend.MatrixFreeDirectedTutteLayer(mesh)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], requires_grad=True)
    out = layer(torch.empty((0, 0), dtype=torch.float64), boundary)
    torch.testing.assert_close(out, torch.tensor(mesh.vertices))
    out.sum().backward()
    torch.testing.assert_close(boundary.grad, torch.ones_like(boundary))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_returned_dtype_positive_topology_and_identity(dtype):
    mesh, layer, z, _ = make_case(5, 4)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=dtype)
    out = layer(torch.zeros_like(z, dtype=dtype), boundary)
    torch.testing.assert_close(out, torch.tensor(mesh.vertices, dtype=dtype), atol=2e-5 if dtype == torch.float32 else 1e-9, rtol=0)
    assert out.dtype == dtype
    tri = out[mesh.faces.copy()]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    assert torch.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0)


def test_float32_collapse_is_rejected():
    mesh, layer, z, _ = make_case(2, 2)
    z[:] = 0; z[:, 0] = 21
    b = torch.tensor(mesh.vertices[mesh.boundary_loops[0]] + 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="positive.*area|area.*positive"):
        layer(z.float(), b)


@pytest.mark.parametrize("case", ["nan", "underflow", "clockwise", "shape", "dtype", "batch"])
def test_invalid_inputs(case):
    _, layer, z, b = make_case()
    if case == "nan": z[0, 0] = float("nan")
    elif case == "underflow": z[0, 0] = -10000
    elif case == "clockwise": b = b.flip(0)
    elif case == "shape": z = z[:, :-1]
    elif case == "dtype": b = b.float()
    elif case == "batch": z, b = z.repeat(2, 1, 1), b.repeat(3, 1, 1)
    with pytest.raises((ValueError, TypeError)):
        layer(z, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_output_and_gradients(dtype):
    mesh, layer, z, b = make_case()
    layer = layer.to("cuda")
    z = z.to(device="cuda", dtype=dtype).requires_grad_()
    b = b.to(device="cuda", dtype=dtype).requires_grad_()
    out = layer(z, b)
    expected = DirectTutteLayer(mesh)(z.detach().cpu(), b.detach().cpu())
    tolerance = 3e-5 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(out.cpu(), expected, atol=tolerance, rtol=tolerance)
    out.square().sum().backward()
    assert z.grad.device.type == "cuda" and torch.isfinite(z.grad).all()
    assert b.grad.device.type == "cuda" and torch.isfinite(b.grad).all()
