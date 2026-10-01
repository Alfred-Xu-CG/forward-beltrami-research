import importlib.util

import pytest
import torch
import torch.nn.functional as F


def api():
    name="qcopt.neural_bijection.dense.coordinated_proposal_filter"
    assert importlib.util.find_spec(name) is not None,"proposal filter not implemented"
    from qcopt.neural_bijection.dense.coordinated_proposal_filter import dirichlet_lazy_proposal_filter
    return dirichlet_lazy_proposal_filter


def independent_matrix(rows,columns):
    matrix=torch.zeros(rows*columns,rows*columns,dtype=torch.float64)
    for row in range(rows):
        for column in range(columns):
            index=row*columns+column
            matrix[index,index]=.5
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr,cc=row+dr,column+dc
                if 0<=rr<rows and 0<=cc<columns:
                    matrix[index,rr*columns+cc]=.125
    return matrix


@pytest.mark.parametrize("shape",[(1,1),(1,5),(4,1),(3,5),(4,6)])
def test_independent_spd_spectrum_and_dense_values(shape):
    rows,columns=shape;matrix=independent_matrix(rows,columns)
    expected=torch.tensor([.5+.25*torch.cos(torch.tensor(torch.pi*k/(rows+1),dtype=torch.float64))
        +.25*torch.cos(torch.tensor(torch.pi*l/(columns+1),dtype=torch.float64))
        for k in range(1,rows+1) for l in range(1,columns+1)])
    torch.testing.assert_close(torch.linalg.eigvalsh(matrix),expected.sort().values,rtol=2e-14,atol=2e-14)
    assert torch.linalg.eigvalsh(matrix).min()>0
    torch.testing.assert_close(matrix,matrix.T,rtol=0,atol=0)
    raw=torch.randn(2,3,rows,columns,generator=torch.Generator().manual_seed(522),dtype=torch.float64)
    for steps in (0,1,4):
        actual=api()(raw,steps=steps)
        reference=(raw.flatten(2)@torch.linalg.matrix_power(matrix,steps).T).reshape_as(raw)
        torch.testing.assert_close(actual,reference,rtol=2e-14,atol=2e-14)
        padded=F.pad(actual,(1,1,1,1))
        assert torch.count_nonzero(padded[:,:,0])==torch.count_nonzero(padded[:,:,-1])==0
        assert torch.count_nonzero(padded[:,:,:,0])==torch.count_nonzero(padded[:,:,:,-1])==0


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("steps",[0,1,4])
def test_symmetric_full_vjp_and_directional_fd(dtype,steps):
    generator=torch.Generator().manual_seed(42)
    raw=torch.randn(2,2,3,5,generator=generator,dtype=dtype,requires_grad=True)
    upstream=torch.randn(raw.shape,generator=generator,dtype=dtype)
    tangent=torch.randn(raw.shape,generator=generator,dtype=dtype)
    result=api()(raw,steps=steps)
    gradient,=torch.autograd.grad(result,raw,upstream)
    matrix=torch.linalg.matrix_power(independent_matrix(3,5),steps).to(dtype)
    dense_gradient=(upstream.flatten(2)@matrix).reshape_as(raw)
    torch.testing.assert_close(gradient,dense_gradient,rtol=1e-6 if dtype==torch.float32 else 2e-14,
        atol=1e-7 if dtype==torch.float32 else 2e-14)
    torch.testing.assert_close(gradient,api()(upstream,steps=steps),rtol=1e-6 if dtype==torch.float32 else 2e-14,
        atol=1e-7 if dtype==torch.float32 else 2e-14)
    h=1e-3 if dtype==torch.float32 else 1e-6
    fd=((api()(raw.detach()+h*tangent,steps=steps)-api()(raw.detach()-h*tangent,steps=steps))*upstream).sum()/(2*h)
    torch.testing.assert_close(fd,(gradient*tangent).sum(),rtol=5e-4 if dtype==torch.float32 else 2e-8,
        atol=5e-4 if dtype==torch.float32 else 2e-8)
    assert result.shape==raw.shape and result.dtype==raw.dtype and result.device==raw.device
    if steps==0:assert result is raw


def test_zero_and_zero_ghost_not_constant_preserving():
    filter=api()
    assert torch.count_nonzero(filter(torch.zeros(2,3,4,5),steps=4))==0
    actual=filter(torch.ones(1,1,3,5,dtype=torch.float64),steps=1)
    assert actual[0,0,0,0]==.75 and actual[0,0,0,2]==.875 and actual[0,0,1,2]==1
    assert filter(torch.ones(1,1,1,1),steps=4).item()==.5**4


def test_noncontiguous_input_and_default_four_passes():
    raw=torch.randn(2,3,5,4,dtype=torch.float64).transpose(-1,-2)
    assert not raw.is_contiguous()
    matrix=torch.linalg.matrix_power(independent_matrix(4,5),4)
    expected=(raw.flatten(2)@matrix.T).reshape_as(raw)
    torch.testing.assert_close(api()(raw),expected,rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize("case",["not_tensor","shape","empty_batch","empty_channel","empty_rows","empty_columns",
    "integer","half","nan","inf","negative_steps","float_steps","bool_steps","tensor_steps","grad_steps"])
def test_guards(case):
    raw=torch.zeros(1,1,2,3,dtype=torch.float64);steps=4
    if case=="not_tensor":raw=[1.]
    elif case=="shape":raw=raw[0]
    elif case=="empty_batch":raw=raw[:0]
    elif case=="empty_channel":raw=raw[:,:0]
    elif case=="empty_rows":raw=raw[:,:,:0]
    elif case=="empty_columns":raw=raw[:,:,:,:0]
    elif case=="integer":raw=raw.long()
    elif case=="half":raw=raw.half()
    elif case=="nan":raw[0,0,0,0]=float("nan")
    elif case=="inf":raw[0,0,0,0]=float("inf")
    elif case=="negative_steps":steps=-1
    elif case=="float_steps":steps=4.
    elif case=="bool_steps":steps=True
    elif case=="tensor_steps":steps=torch.tensor(4)
    elif case=="grad_steps":steps=torch.tensor(4.,requires_grad=True)
    with pytest.raises(ValueError):api()(raw,steps=steps)
