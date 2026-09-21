"""Independent tests for the factor-reusing CPU reference layer."""

import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse.linalg as sla
import torch

from qcopt.mesh import structured_rectangle


@pytest.fixture
def direct():
    def construct(mesh):
        name = "qcopt.neural_bijection.tutte.direct"
        assert importlib.util.find_spec(name) is not None, "CPU direct layer is missing"
        return importlib.import_module(name).DirectTutteLayer(mesh)
    return SimpleNamespace(DirectTutteLayer=construct)


def inputs(direct, nx=3, ny=3):
    mesh = structured_rectangle(nx, ny)
    layer = direct.DirectTutteLayer(mesh)
    rng = np.random.default_rng(412)
    logits = torch.tensor(rng.normal(size=(layer.system.n_rows, layer.system.max_degree)), dtype=torch.float64)
    # Strictly convex ellipse avoids boundary-domain issues in finite differences.
    t = np.arange(len(mesh.boundary_loops[0])) * 2 * np.pi / len(mesh.boundary_loops[0])
    boundary = torch.tensor(np.column_stack((1.4 * np.cos(t), 0.8 * np.sin(t))), dtype=torch.float64)
    return mesh, layer, logits, boundary


def dense_reference(mesh, logits, boundary):
    loop = mesh.boundary_loops[0]
    interior = np.setdiff1d(np.arange(mesh.n_vertices), loop)
    adjacency = [set() for _ in range(mesh.n_vertices)]
    for face in mesh.faces:
        for vertex in face:
            adjacency[vertex].update(set(face) - {vertex})
    a = np.eye(mesh.n_vertices)
    b = np.zeros((mesh.n_vertices, 2))
    b[loop] = boundary.detach().numpy()
    for row, vertex in enumerate(interior):
        neighbors = sorted(adjacency[vertex])
        p = torch.softmax(logits[row, :len(neighbors)], 0).detach().numpy()
        a[vertex, neighbors] -= p
    return np.linalg.solve(a, b), a[np.ix_(interior, interior)]


def test_direct_matches_independent_nonsymmetric_dense_system(direct):
    mesh, layer, logits, boundary = inputs(direct)
    expected, matrix = dense_reference(mesh, logits, boundary)
    assert np.linalg.norm(matrix - matrix.T) > 0.1
    actual = layer(logits, boundary)
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-13)
    torch.testing.assert_close(actual[mesh.boundary_loops[0].copy()], boundary)


def test_identity_boundary_permutation_and_float32(direct):
    mesh, layer, logits, _ = inputs(direct, 4, 3)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float32)
    actual = layer(torch.zeros_like(logits, dtype=torch.float32), boundary)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    torch.testing.assert_close(actual, torch.tensor(mesh.vertices, dtype=torch.float32), atol=2e-7, rtol=0)


def test_batched_and_broadcast_outputs_and_gradients(direct):
    _, layer, logits, boundary = inputs(direct)
    z = torch.stack((logits, logits * 0.4)).requires_grad_()
    b = boundary.clone().requires_grad_()
    result = layer(z, b)
    expected = torch.stack([layer(z[i], b) for i in range(2)])
    torch.testing.assert_close(result, expected)
    grads = torch.autograd.grad(result.square().sum(), (z, b), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), (z, b))
    for g, eg in zip(grads, expected_grads):
        torch.testing.assert_close(g, eg)
    torch.testing.assert_close(layer(logits, torch.stack((boundary, boundary * 2))),
                               torch.stack((layer(logits, boundary), layer(logits, boundary * 2))))


def test_nonsymmetric_logits_and_boundary_gradcheck(direct):
    _, layer, logits, boundary = inputs(direct, 3, 2)
    assert torch.autograd.gradcheck(layer, (logits.requires_grad_(), boundary.requires_grad_()),
                                    eps=1e-6, atol=2e-6, rtol=2e-5)


def test_directional_finite_difference(direct):
    _, layer, logits, boundary = inputs(direct)
    z, b = logits.requires_grad_(), boundary.requires_grad_()
    cotangent = torch.arange(layer.system.n_vertices * 2, dtype=z.dtype).reshape(-1, 2).sin()
    dz, db = torch.cos(z), torch.sin(b)
    gz, gb = torch.autograd.grad((layer(z, b) * cotangent).sum(), (z, b))
    eps = 1e-6
    fd = ((layer(z + eps * dz, b + eps * db) - layer(z - eps * dz, b - eps * db)) * cotangent).sum() / (2 * eps)
    torch.testing.assert_close(fd, (gz * dz).sum() + (gb * db).sum(), atol=1e-7, rtol=1e-6)


def test_one_factor_per_sample_and_reused_transpose_multirhs(direct, monkeypatch):
    _, layer, logits, boundary = inputs(direct)
    original = sla.splu
    factorizations, solves = [], []

    def record(a, *args, **kwargs):
        lu = original(a, *args, **kwargs)
        number = len(factorizations)
        factorizations.append(a.shape)

        class ObservedLU:
            def solve(self, rhs, trans="N"):
                solves.append((number, rhs.shape, trans))
                return lu.solve(rhs, trans=trans)

        return ObservedLU()

    monkeypatch.setattr(sla, "splu", record)
    z = torch.stack((logits, logits * 0.7)).requires_grad_()
    b = boundary.requires_grad_()
    layer(z, b).square().sum().backward()
    assert len(factorizations) == 2
    assert solves == [(0, (layer.system.n_rows, 2), "N"), (1, (layer.system.n_rows, 2), "N"),
                      (0, (layer.system.n_rows, 2), "T"), (1, (layer.system.n_rows, 2), "T")]


@pytest.mark.parametrize("case", ["nan_logits", "nan_boundary", "underflow", "clockwise", "repeated", "shape", "dtype", "batch"])
def test_invalid_inputs_rejected(direct, case):
    _, layer, z, b = inputs(direct)
    if case == "nan_logits": z[0, 0] = float("nan")
    elif case == "nan_boundary": b[0, 0] = float("nan")
    elif case == "underflow": z[0, 0] = -10000
    elif case == "clockwise": b = b.flip(0)
    elif case == "repeated": b[1] = b[0]
    elif case == "shape": z = z[:, :-1]
    elif case == "dtype": b = b.float()
    elif case == "batch": z, b = z.repeat(2, 1, 1), b.repeat(3, 1, 1)
    with pytest.raises((ValueError, TypeError)):
        layer(z, b)


def test_positive_area_certificate_rejects_collapsed_output(direct):
    mesh, layer, logits, _ = inputs(direct, 3, 3)
    # Positive but numerically concentrated rows collapse nearby interior vertices.
    logits[:] = -100
    logits[:, 0] = 100
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive.*area|area.*positive"):
        layer(logits, boundary)


def test_no_interior_vertices(direct, monkeypatch):
    mesh = structured_rectangle(1, 1)
    layer = direct.DirectTutteLayer(mesh)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64, requires_grad=True)
    def unexpected(*args, **kwargs):
        pytest.fail("no factorization is needed for an empty interior")
    monkeypatch.setattr(sla, "splu", unexpected)
    actual = layer(torch.empty((0, 0), dtype=torch.float64), boundary)
    torch.testing.assert_close(actual, torch.tensor(mesh.vertices, dtype=torch.float64))
    actual.sum().backward()
    torch.testing.assert_close(boundary.grad, torch.ones_like(boundary))


def test_inplace_output_scaling_does_not_corrupt_saved_primal(direct):
    _, layer, logits, boundary = inputs(direct)
    z = logits.unsqueeze(0).requires_grad_()
    b = boundary.unsqueeze(0).requires_grad_()
    baseline = layer(z, b)
    expected = torch.autograd.grad((baseline * 2).sum(), (z, b))
    inplace = layer(z, b)
    inplace.mul_(2)
    actual = torch.autograd.grad(inplace.sum(), (z, b))
    for g, reference in zip(actual, expected):
        torch.testing.assert_close(g, reference, atol=1e-12, rtol=1e-12)


def test_float32_cast_collapse_rejected_despite_valid_double_solution(direct):
    mesh, layer, logits, _ = inputs(direct, 2, 2)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]] + 1, dtype=torch.float64)
    logits[:] = 0
    logits[:, 0] = 21
    double_map = layer(logits, boundary)
    triangles = double_map[mesh.faces.copy()]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    assert torch.all(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] > 0)
    rounded = double_map.float()[mesh.faces.copy()]
    e1, e2 = rounded[:, 1] - rounded[:, 0], rounded[:, 2] - rounded[:, 0]
    assert torch.any(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] == 0)
    with pytest.raises(ValueError, match="positive.*area|area.*positive"):
        layer(logits.float(), boundary.float())
