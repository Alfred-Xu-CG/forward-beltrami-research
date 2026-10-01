import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_nested_refinement import FrozenNestedP1Refinement
from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_cached_refinement_function_values_and_adjoint(diagonal,dtype):
    torch.manual_seed(28)
    vertices=torch.randn(2,5,5,2,dtype=dtype,requires_grad=True)
    cached=FrozenNestedP1Refinement(5,13,diagonal=diagonal,dtype=dtype)
    expected=refine_p1_vertices(vertices,3,diagonal)
    actual=cached(vertices)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(actual[:,::3,::3],vertices)
    weights=torch.randn_like(actual)
    a=torch.autograd.grad(actual,vertices,weights,retain_graph=True)[0]
    b=torch.autograd.grad(expected,vertices,weights,retain_graph=True)[0]
    # Cached weighted-scatter groups sums differently from dynamic gather AD.
    # Values are bit-identical; floating adjoint accumulation need not be.
    torch.testing.assert_close(a,b,rtol=2e-6 if dtype==torch.float32 else 2e-14,
        atol=2e-6 if dtype==torch.float32 else 2e-14)
    q=torch.rand(1,21,2,dtype=dtype)
    torch.testing.assert_close(p1_map_at_queries(actual,q,diagonal),
        p1_map_at_queries(vertices,q,diagonal),rtol=3e-6 if dtype==torch.float32 else 1e-13,
        atol=3e-6 if dtype==torch.float32 else 1e-13)
    tangent=torch.randn_like(vertices);eps=1e-6 if dtype==torch.float64 else .001
    fd=((cached(vertices.detach()+eps*tangent)-cached(vertices.detach()-eps*tangent))*weights).sum()/(2*eps)
    torch.testing.assert_close(fd,(a*tangent).sum(),rtol=2e-3 if dtype==torch.float32 else 1e-7,
        atol=2e-3 if dtype==torch.float32 else 1e-7)


@pytest.mark.parametrize("sizes",[(5,8),(5,3),(True,9),(2,5)])
def test_reject_non_nested_sizes(sizes):
    with pytest.raises(ValueError,match="nested"):
        FrozenNestedP1Refinement(*sizes)


def test_identity_materialization_has_no_buffer_or_copy():
    layer=FrozenNestedP1Refinement(5,5)
    vertices=torch.ones(1,5,5,2,requires_grad=True)
    assert layer(vertices) is vertices and layer.resident_bytes==0
