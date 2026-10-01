"""Exact nested ARAP mean, with rounded materialization explicitly distinct."""
import pytest
import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.coordinated_nested_priors import (
    ExactNestedP1Priors, exact_nested_p1_priors,
)
from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from tools.coordinated_real_case import corner_symmetric_dirichlet


def legal_vertices(dtype=torch.float64):
    axis = torch.arange(5, dtype=torch.float32)/4
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    reference = torch.stack((xx, yy), -1).to(dtype)
    vertices = reference[None].repeat(2, 1, 1, 1)
    generator = torch.Generator().manual_seed(953)
    vertices += .012*torch.randn(vertices.shape, generator=generator, dtype=dtype)
    return vertices


def literal_triangle_svd_energy(vertices, diagonal):
    """Independent source-edge matrix solve, actual faces and NumPy SVD."""
    values = vertices.detach().numpy()
    batch, rows, columns, _ = values.shape
    source = np.array([[0., 0.], [1./(columns-1), 0.],
                       [1./(columns-1), 1./(rows-1)], [0., 1./(rows-1)]])
    faces = ((0, 1, 2), (0, 2, 3)) if diagonal == "ac" else ((0, 1, 3), (1, 2, 3))
    energies = []
    for n in range(batch):
        for row in range(rows-1):
            for column in range(columns-1):
                mapped = values[n, [row, row, row+1, row+1],
                                [column, column+1, column+1, column]]
                for a, b, c in faces:
                    source_edges = np.stack((source[b]-source[a], source[c]-source[a]), axis=1)
                    mapped_edges = np.stack((mapped[b]-mapped[a], mapped[c]-mapped[a]), axis=1)
                    jacobian = np.linalg.solve(source_edges.T, mapped_edges.T).T
                    assert np.linalg.det(jacobian) > 0
                    u, _, vt = np.linalg.svd(jacobian)
                    rotation = u @ vt
                    energies.append(.5*np.sum((jacobian-rotation)**2))
    return np.mean(energies)


@pytest.mark.parametrize("diagonal", ["ac", "bd"])
@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_b2_nonlinear_value_shape_full_vertex_gradient(diagonal, factor):
    vertices = legal_vertices().requires_grad_()
    side = factor*(vertices.shape[1]-1)+1
    fine = refine_p1_vertices(vertices, factor, diagonal)
    expected = (p1_arap_energy(fine, diagonal), corner_symmetric_dirichlet(fine))
    actual = exact_nested_p1_priors(vertices, side, diagonal, strain_model="p1_arap")
    cached = ExactNestedP1Priors(5, side, diagonal=diagonal, strain_model="p1_arap")(vertices)
    for full, reduced, module_value in zip(expected, actual, cached):
        torch.testing.assert_close(reduced, full, rtol=2e-11, atol=2e-13)
        torch.testing.assert_close(reduced, module_value, rtol=0, atol=0)
        full_gradient, = torch.autograd.grad(full, vertices, retain_graph=True)
        reduced_gradient, = torch.autograd.grad(reduced, vertices, retain_graph=True)
        torch.testing.assert_close(reduced_gradient, full_gradient, rtol=2e-10, atol=2e-10)
    assert abs(float(actual[0])-literal_triangle_svd_energy(vertices, diagonal)) < 2e-14


@pytest.mark.parametrize("diagonal", ["ac", "bd"])
def test_full_vertex_directional_fd_and_shape_unchanged(diagonal):
    vertices = legal_vertices().requires_grad_()
    arap, shape = exact_nested_p1_priors(vertices, 33, diagonal, strain_model="p1_arap")
    _, old_shape = exact_nested_p1_priors(vertices, 33, diagonal)
    torch.testing.assert_close(shape, old_shape, rtol=0, atol=0)
    gradient, = torch.autograd.grad(3*arap+1e-4*shape, vertices)
    tangent = torch.randn(vertices.shape, dtype=vertices.dtype, generator=torch.Generator().manual_seed(5))
    h = 1e-6
    plus = exact_nested_p1_priors(vertices.detach()+h*tangent, 33, diagonal, strain_model="p1_arap")
    minus = exact_nested_p1_priors(vertices.detach()-h*tangent, 33, diagonal, strain_model="p1_arap")
    fd = (3*(plus[0]-minus[0])+1e-4*(plus[1]-minus[1]))/(2*h)
    torch.testing.assert_close(fd, (gradient*tangent).sum(), rtol=2e-8, atol=2e-9)
    assert gradient[:, 0].abs().sum() > 0 and gradient[:, :, -1].abs().sum() > 0


@pytest.mark.parametrize("diagonal", ["ac", "bd"])
def test_float32_rounded_fine_discrepancy_reported(diagonal, record_property):
    vertices = legal_vertices(torch.float32).requires_grad_()
    fine = refine_p1_vertices(vertices, 8, diagonal)
    actual = exact_nested_p1_priors(vertices, 33, diagonal, strain_model="p1_arap")
    expected = (p1_arap_energy(fine, diagonal), corner_symmetric_dirichlet(fine))
    for name, full, reduced in zip(("arap", "shape"), expected, actual):
        fg, = torch.autograd.grad(full, vertices, retain_graph=True)
        rg, = torch.autograd.grad(reduced, vertices, retain_graph=True)
        value_error = float((reduced-full).abs()/full.abs())
        gradient_error = float(torch.linalg.vector_norm(rg-fg)/torch.linalg.vector_norm(fg))
        max_absolute = float((rg-fg).abs().max())
        record_property(name+"_relative_value_rounding_error", value_error)
        record_property(name+"_relative_gradient_rounding_error", gradient_error)
        record_property(name+"_maximum_absolute_gradient_rounding_error", max_absolute)
        print(dict(diagonal=diagonal, term=name, relative_value=value_error,
                   relative_gradient=gradient_error, maximum_absolute_gradient=max_absolute))
        assert value_error < 1e-4 and gradient_error < 1e-4


@pytest.mark.parametrize("diagonal", ["ac", "bd"])
def test_default_and_explicit_displacement_path_bit_identical(diagonal):
    vertices = legal_vertices().requires_grad_()
    default = exact_nested_p1_priors(vertices, 17, diagonal)
    explicit = exact_nested_p1_priors(vertices, 17, diagonal, strain_model="displacement_gradient")
    for left, right in zip(default, explicit):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        lg, = torch.autograd.grad(left, vertices, retain_graph=True)
        rg, = torch.autograd.grad(right, vertices, retain_graph=True)
        torch.testing.assert_close(lg, rg, rtol=0, atol=0)


@pytest.mark.parametrize("model", ["wrong", None, True, 1])
def test_invalid_model_rejected_by_both_apis(model):
    vertices = legal_vertices()
    with pytest.raises(ValueError, match="strain_model"):
        ExactNestedP1Priors(5, 17, strain_model=model)
    with pytest.raises(ValueError, match="strain_model"):
        exact_nested_p1_priors(vertices, 17, strain_model=model)


@pytest.mark.parametrize("case", ["dtype", "non_square", "batch", "smaller", "nondyadic", "diagonal", "buffer_dtype", "trainable_reference"])
def test_arap_retains_existing_guards(case):
    vertices, side, diagonal = legal_vertices(), 17, "ac"
    if case == "dtype": vertices = vertices.half()
    elif case == "non_square": vertices = vertices[:, :, :-1]
    elif case == "batch": vertices = vertices[:0]
    elif case == "smaller": side = 3
    elif case == "nondyadic": side = 13
    elif case == "diagonal": diagonal = "bad"
    with pytest.raises(ValueError):
        if case == "buffer_dtype":
            ExactNestedP1Priors(5, 17, dtype=torch.float32, strain_model="p1_arap")(vertices)
        elif case == "trainable_reference":
            module = ExactNestedP1Priors(5, 17, strain_model="p1_arap")
            module.source_reference.requires_grad_()
            module(vertices)
        else:
            exact_nested_p1_priors(vertices, side, diagonal, strain_model="p1_arap")
