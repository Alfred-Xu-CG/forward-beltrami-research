"""Independent fixed-source triangle tests, not just comparison to old sampler."""
import importlib.util

import torch
import numpy as np
import pytest


def evaluator_type():
    name = "qcopt.neural_bijection.dense.coordinated_fixed_sampling"
    assert importlib.util.find_spec(name) is not None, "fixed-query evaluator is not implemented"
    from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
    return FrozenP1Evaluator


def independent_matrix(rows, columns, queries, diagonal):
    source = np.array([(x/(columns-1), y/(rows-1)) for y in range(rows) for x in range(columns)])
    matrix = np.zeros((len(queries), rows*columns))
    for k, (x,y) in enumerate(queries):
        column = min(int(np.floor(x*(columns-1))), columns-2)
        row = min(int(np.floor(y*(rows-1))), rows-2)
        a = row*columns+column
        b,c,d = a+1,a+columns+1,a+columns
        xi,zeta = x*(columns-1)-column,y*(rows-1)-row
        if diagonal == "ac":
            ids = [a,b,c] if zeta<=xi else [a,c,d]
        else:
            ids = [a,b,d] if xi+zeta<=1 else [b,c,d]
        matrix[k,ids] = np.linalg.solve(np.vstack((source[ids].T,np.ones(3))),[x,y,1])
    return matrix


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_non_square_batched_values_vertex_vjp_and_fd(diagonal):
    cls = evaluator_type()
    rng=np.random.default_rng(612)
    q=np.concatenate((rng.uniform(0,1,(27,2)),[[0,0],[1,1],[1,0],[0,1],[.375,.25],[.125,.25],[.5,1],[1,.5]]))
    queries=torch.tensor(q,dtype=torch.float64)[None]
    vertices=torch.tensor(rng.normal(size=(2,3,5,2)),dtype=torch.float64,requires_grad=True)
    module=cls(3,5,queries,diagonal)
    matrix=independent_matrix(3,5,q,diagonal)
    actual=module(vertices)
    expected=np.einsum('qv,bvc->bqc',matrix,vertices.detach().numpy().reshape(2,15,2))
    np.testing.assert_allclose(actual.detach(),expected,rtol=0,atol=2e-15)
    upstream=torch.tensor(rng.normal(size=actual.shape),dtype=torch.float64)
    gradient,=torch.autograd.grad(actual,vertices,upstream)
    expected_gradient=np.einsum('qv,bqc->bvc',matrix,upstream.numpy()).reshape(2,3,5,2)
    np.testing.assert_allclose(gradient,expected_gradient,rtol=0,atol=3e-15)
    direction=torch.tensor(rng.normal(size=vertices.shape),dtype=torch.float64)
    h=1e-6
    fd=((module(vertices.detach()+h*direction)-module(vertices.detach()-h*direction))*upstream).sum()/(2*h)
    assert abs(float(fd-(gradient*direction).sum()))<2e-8
    # Caller mutation cannot change frozen geometry; same queries support B=1/4.
    before=actual.detach().clone();queries.zero_()
    torch.testing.assert_close(module(vertices),before,rtol=0,atol=0)
    assert module(vertices[:1]).shape==(1,len(q),2)
    assert module(vertices.detach()[:1].expand(4,-1,-1,-1)).shape==(4,len(q),2)
    assert set(dict(module.named_buffers()))=={"vertex_ids","weights"}


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_image_query_shape_dtype_transfer_and_existing_sampler(diagonal):
    from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries
    cls=evaluator_type()
    queries=torch.rand(1,4,7,2,dtype=torch.float64)
    vertices=torch.randn(2,3,5,2,dtype=torch.float64)
    module=cls(3,5,queries,diagonal)
    torch.testing.assert_close(module(vertices),p1_map_at_queries(vertices,queries,diagonal),rtol=0,atol=5e-16)
    converted=module.to(dtype=torch.float32)
    assert converted.vertex_ids.dtype==torch.long
    assert converted.weights.dtype==torch.float32
    torch.testing.assert_close(converted(vertices.float()),p1_map_at_queries(vertices.float(),queries.float(),diagonal),rtol=0,atol=8e-7)
    with pytest.raises(ValueError,match="dtype"):
        converted(vertices)
    with pytest.raises(ValueError,match="vertices"):
        converted(torch.zeros(1,4,5,2))


@pytest.mark.parametrize("kind",["grad","batch","nan","outside","integer","shape","empty","diagonal","rows"])
def test_explicit_preconditions(kind):
    cls=evaluator_type();q=torch.tensor([[[.2,.3]]],dtype=torch.float64)
    rows,columns,diagonal=3,5,"ac"
    if kind=="grad":q.requires_grad_()
    elif kind=="batch":q=q.expand(2,-1,-1)
    elif kind=="nan":q[0,0,0]=float('nan')
    elif kind=="outside":q[0,0,0]=1.01
    elif kind=="integer":q=q.long()
    elif kind=="shape":q=q[0]
    elif kind=="empty":q=q[:,:0]
    elif kind=="diagonal":diagonal="bad"
    elif kind=="rows":rows=True
    with pytest.raises(ValueError):cls(rows,columns,q,diagonal)
