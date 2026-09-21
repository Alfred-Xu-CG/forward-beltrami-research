from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch
from scipy.sparse import linalg as sparse_linalg

from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.decoder import TutteRectangleDecoder
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer


def _api():
    name = 'qcopt.neural_bijection.tutte.composition'
    assert importlib.util.find_spec(name) is not None, 'composition module is not implemented'
    return importlib.import_module(name)


def _layers(count, dtype=torch.float64):
    mesh = structured_rectangle(4, 3)
    decoders = [TutteRectangleDecoder(mesh, DirectTutteLayer(mesh), image_height=13,
                image_width=17) for _ in range(count)]
    module = _api().SquareTutteComposition(decoders)
    module.prepare(device='cpu', dtype=dtype)
    g = torch.Generator().manual_seed(534)
    latents = [(.12 * torch.randn(d.solver.system.n_rows, d.solver.system.max_degree,
                generator=g, dtype=dtype)).requires_grad_() for d in decoders]
    boundaries = [(.10 * torch.randn(d.boundary.n_segments, generator=g,
                    dtype=dtype)).requires_grad_() for d in decoders]
    return module, latents, boundaries


@pytest.mark.parametrize('count', [1, 2, 4])
def test_affine_coordinate_composition_matches_analytic_order(count):
    # Algebra fixtures: contractions into the square, NOT onto-square Tutte maps.
    api = _api()
    mesh = structured_rectangle(5, 4)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=13, width=17)
    source = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    matrix = torch.tensor([[.65, .10], [.04, .72]], dtype=torch.float64)
    offset = torch.tensor([.08, .07], dtype=torch.float64)
    controls = [source @ matrix.T + (i + 1) * offset / count for i in range(count)]
    result = api.compose_control_maps([table] * count, controls)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 13, dtype=torch.float64),
                           torch.linspace(0, 1, 17, dtype=torch.float64), indexing='ij')
    expected = torch.stack((xx, yy), dim=-1)
    for i in range(count):
        expected = expected @ matrix.T + (i + 1) * offset / count
    torch.testing.assert_close(result, expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_real_layers_preserve_unit_height_controls_and_no_composite_p1_claim(count, dtype):
    module, latents, boundaries = _layers(count, dtype)
    result = module(latents, boundaries)
    assert len(result.controls) == len(result.boundaries) == count
    assert result.dense.shape == (13, 17, 2)
    assert not hasattr(result, 'global_injectivity_certificate')
    for boundary, control in zip(result.boundaries, result.controls):
        assert boundary[..., 1].max().item() == 1.0
        assert boundary[..., 1].min().item() == 0.0
        assert bool((control >= -1e-6).all()) and bool((control <= 1 + 1e-6).all())
        assert audit_injectivity(structured_rectangle(4, 3), control.detach().double().numpy()).certified


@pytest.mark.parametrize('batch', [1, 4, 8])
def test_mixed_layer_batch_broadcast_matches_independent_samples(batch):
    module, latents, boundaries = _layers(2)
    batched = latents[1].detach().unsqueeze(0).repeat(batch, 1, 1).requires_grad_()
    inputs = [latents[0], batched]
    actual = module(inputs, boundaries).dense
    expected = torch.stack([module([latents[0], batched[i]], boundaries).dense
                            for i in range(batch)])
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=0)
    variables = [latents[0], batched, *boundaries]
    ga = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), variables)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)


def test_joint_interior_boundary_directional_derivative():
    module, latents, boundaries = _layers(2)
    variables = latents + boundaries
    g = torch.Generator().manual_seed(879)
    directions = [torch.randn(v.shape, generator=g, dtype=v.dtype) for v in variables]
    weight = torch.randn((13, 17, 2), generator=g, dtype=torch.float64)
    def objective(values):
        return (module(values[:2], values[2:]).dense * weight).sum()
    loss = objective(variables)
    grads = torch.autograd.grad(loss, variables)
    analytic = sum((a * b).sum() for a, b in zip(grads, directions))
    h = 1e-7
    finite = (objective([v + h*d for v, d in zip(variables, directions)]) -
              objective([v - h*d for v, d in zip(variables, directions)])) / (2*h)
    torch.testing.assert_close(analytic, finite, atol=3e-7, rtol=3e-6)


def test_exact_solve_calls_one_per_layer_sample_and_transpose_reuse(monkeypatch):
    module, latents, boundaries = _layers(4)
    latents = [v.detach().unsqueeze(0).repeat(2, 1, 1).requires_grad_() for v in latents]
    original = sparse_linalg.splu
    factors = []
    def record(*args, **kwargs):
        factor = original(*args, **kwargs)
        factors.append(factor)
        return factor
    monkeypatch.setattr(sparse_linalg, 'splu', record)
    result = module(latents, boundaries)
    assert len(factors) == 8
    result.dense.square().sum().backward()
    assert len(factors) == 8
    assert all(v.grad is not None for v in latents + boundaries)


@pytest.mark.parametrize('bad', [float('nan'), 1.1, -.1])
def test_control_domain_violations_rejected_even_on_final_layer(bad):
    mesh = structured_rectangle(3, 3)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=5, width=5)
    control = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    control[0, 0] = bad
    with pytest.raises(ValueError, match='finite|unit square'):
        _api().compose_control_maps([table], [control])


def test_invalid_probability_failure_propagates():
    module, latents, boundaries = _layers(2)
    latents[1] = torch.full_like(latents[1], -10000)
    latents[1][:, 0] = 10000
    with pytest.raises(ValueError, match='positive'):
        module(latents, boundaries)


def test_layer_counts_and_shapes_rejected():
    module, latents, boundaries = _layers(2)
    with pytest.raises(ValueError, match='layer'):
        module(latents[:1], boundaries)
    with pytest.raises(ValueError, match='layer'):
        _api().SquareTutteComposition([])
    with pytest.raises(ValueError, match='layer'):
        _api().compose_control_maps([], [])


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('batch', [1, 4, 8])
def test_mandatory_query_resolution_identity(count, batch):
    mesh = structured_rectangle(4, 4)
    decoders = [TutteRectangleDecoder(mesh, DirectTutteLayer(mesh), image_height=256,
                image_width=256) for _ in range(count)]
    module = _api().SquareTutteComposition(decoders)
    module.prepare(device='cpu', dtype=torch.float32)
    latents = [torch.zeros(batch, d.solver.system.n_rows, d.solver.system.max_degree)
               for d in decoders]
    boundaries = [torch.zeros(d.boundary.n_segments) for d in decoders]
    actual = module(latents, boundaries).dense
    line = torch.linspace(0, 1, 256)
    yy, xx = torch.meshgrid(line, line, indexing='ij')
    expected = torch.stack((xx, yy), dim=-1).expand(batch, -1, -1, -1)
    torch.testing.assert_close(actual, expected, atol=8e-7, rtol=0)


@pytest.mark.parametrize('backend', ['directed', 'symmetric'])
def test_composition_accepts_matrix_free_solver_contract(backend):
    from qcopt.neural_bijection.tutte.iterative import MatrixFreeDirectedTutteLayer
    from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer
    mesh = structured_rectangle(3, 3)
    decoders, latents, boundaries = [], [], []
    for _ in range(2):
        solver = (MatrixFreeDirectedTutteLayer(mesh) if backend == 'directed'
                  else MatrixFreeSymmetricTutteLayer(mesh))
        decoder = TutteRectangleDecoder(mesh, solver, image_height=13, image_width=17)
        decoders.append(decoder)
        shape = ((solver.system.n_rows, solver.system.max_degree) if backend == 'directed'
                 else (solver.n_conductances,))
        latents.append(torch.zeros(shape, dtype=torch.float64, requires_grad=True))
        boundaries.append(torch.zeros(decoder.boundary.n_segments, dtype=torch.float64,
                                      requires_grad=True))
    result = _api().SquareTutteComposition(decoders)(latents, boundaries)
    result.dense.square().sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in latents + boundaries)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_dynamic_indices_and_boundary_buffers():
    from qcopt.neural_bijection.tutte.iterative import MatrixFreeDirectedTutteLayer
    mesh = structured_rectangle(3, 3)
    ds = [TutteRectangleDecoder(mesh, MatrixFreeDirectedTutteLayer(mesh), image_height=13,
                               image_width=17) for _ in range(2)]
    module = _api().SquareTutteComposition(ds).to('cuda')
    module.prepare(device='cuda', dtype=torch.float64)
    latents = [torch.zeros(d.solver.system.n_rows, d.solver.system.max_degree,
                          dtype=torch.float64, device='cuda', requires_grad=True) for d in ds]
    boundaries = [torch.zeros(d.boundary.n_segments, dtype=torch.float64, device='cuda',
                             requires_grad=True) for d in ds]
    result = module(latents, boundaries)
    assert result.dense.device.type == 'cuda'
    result.dense.square().sum().backward()
    assert all(v.grad is not None for v in latents + boundaries)
