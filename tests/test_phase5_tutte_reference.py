from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest
import torch

from qcopt.forward import tutte_directed_implicit as legacy
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer


def _case(dtype=torch.float64, nx=3, ny=3):
    name = 'qcopt.neural_bijection.tutte.reference'
    assert importlib.util.find_spec(name) is not None, 'legacy reference adapter is missing'
    cls = importlib.import_module(name).LegacyReferenceTutteLayer
    mesh = structured_rectangle(nx, ny)
    layer = cls(mesh)
    generator = torch.Generator().manual_seed(3107)
    z = .2 * torch.randn(layer.system.n_rows, layer.system.max_degree,
                         generator=generator, dtype=dtype)
    b = torch.tensor(mesh.vertices[layer.system.loop], dtype=dtype)
    return mesh, layer, z, b


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reference_calls_legacy_and_preserves_output_dtype(dtype):
    mesh, layer, z, b = _case(dtype)
    actual = layer(z, b)
    expected = legacy.directed_tutte_embedding_torch_implicit(mesh, b, z, layer.system)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('shared', ['boundary', 'latent'])
def test_reference_singleton_batch_broadcast_vjp_matches_improved_direct(shared):
    mesh, layer, z, b = _case()
    z = (torch.stack((z, .7*z)) if shared == 'boundary' else z.unsqueeze(0)).requires_grad_()
    b = (b.unsqueeze(0) if shared == 'boundary' else torch.stack((b, .9*b))).requires_grad_()
    actual = layer(z, b)
    cotangent = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape).cos()
    gradients = torch.autograd.grad((actual*cotangent).sum(), (z, b))
    zr, br = z.detach().requires_grad_(), b.detach().requires_grad_()
    expected = DirectTutteLayer(mesh)(zr, br)
    references = torch.autograd.grad((expected*cotangent).sum(), (zr, br))
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    for value, reference in zip(gradients, references):
        torch.testing.assert_close(value, reference, atol=3e-12, rtol=3e-12)


def test_reference_per_sample_factors_a_and_transpose_in_forward_only(monkeypatch):
    _, layer, z, b = _case()
    original = legacy.factorized
    matrices, solves = [], []

    def spy(matrix):
        matrices.append(matrix.toarray().copy())
        solve = original(matrix)
        def counted(rhs):
            solves.append(1)
            return solve(rhs)
        return counted

    monkeypatch.setattr(legacy, 'factorized', spy)
    z = torch.stack((z, .7*z, 1.2*z)).requires_grad_()
    b = b.requires_grad_()
    output = layer(z, b)
    assert len(matrices) == 6 and len(solves) == 3
    for k in range(0, 6, 2):
        np.testing.assert_array_equal(matrices[k].T, matrices[k+1])
    output.square().sum().backward()
    assert len(matrices) == 6 and len(solves) == 6
    assert z.grad is not None and b.grad is not None


def test_reference_rejects_non_cpu_wrong_dtype_and_empty_batch():
    _, layer, z, b = _case()
    with pytest.raises(ValueError, match='CPU'):
        layer(z.to('meta'), b.to('meta'))
    with pytest.raises(TypeError, match='dtype'):
        layer(z.float(), b)
    with pytest.raises(ValueError, match='batch'):
        layer(z.unsqueeze(0)[:0], b)
    with pytest.raises(TypeError, match='float32|float64'):
        layer(z.half(), b.half())


def test_reference_boundary_only_has_no_factorizations(monkeypatch):
    mesh, layer, z, b = _case(nx=1, ny=1)
    def forbidden(*args):
        pytest.fail('boundary-only reference should not factor a matrix')
    monkeypatch.setattr(legacy, 'factorized', forbidden)
    z, b = z.requires_grad_(), b.requires_grad_()
    output = layer(z, b)
    output.sum().backward()
    torch.testing.assert_close(output, torch.tensor(mesh.vertices))
    torch.testing.assert_close(b.grad, torch.ones_like(b))
    assert z.grad.numel() == 0


def test_reference_rejects_faces_collapsed_by_float32_return_cast():
    mesh, layer, z, b = _case(torch.float32, nx=2, ny=2)
    z.zero_()
    z[0, -1] = 20.0
    # Legacy validates the positive hidden double map before casting. The
    # adapter must not pass its rounded boundary-coincident interior vertex.
    old_output = legacy.directed_tutte_embedding_torch_implicit(mesh, b, z, layer.system)
    torch.testing.assert_close(old_output[layer.system.interior[0]], torch.ones(2))
    with pytest.raises(ValueError, match='positive face'):
        layer(z, b)


def test_reference_rejects_supported_probability_lost_during_legacy_normalization():
    _, layer, z, b = _case(torch.float64, nx=3, ny=3)
    z.zero_()
    row, slot = np.argwhere(layer.system.valid_mask)[0]
    z[row, slot] = -744.0
    realized = layer.system._probabilities(z.numpy())
    assert realized[row, slot] == 0.0

    with pytest.raises(ValueError, match='probabilit.*strictly positive'):
        layer(z, b)


def test_reference_vjp_matches_central_difference():
    _, layer, z, b = _case(nx=2, ny=2)
    # Strict convexity leaves room for independent boundary-coordinate probes.
    angles = torch.arange(len(b), dtype=z.dtype) * (2*torch.pi/len(b))
    b = torch.stack((angles.cos(), angles.sin()), dim=-1)
    assert torch.autograd.gradcheck(layer, (z.requires_grad_(), b.requires_grad_()),
                                    eps=1e-6, atol=2e-7, rtol=2e-5)
