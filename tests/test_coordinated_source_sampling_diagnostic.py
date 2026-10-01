import math

import pytest
import torch

from tools.coordinated_source_sampling_diagnostic import chain_spectrum,make_evaluator,sampling_matrix


@pytest.mark.parametrize("length",[1,3,15])
def test_chain_closed_spectrum_matches_independent_adjacent_sums(length):
    s=torch.zeros(length+1,length,dtype=torch.float64)
    for node in range(length):
        s[node,node]=s[node+1,node]=.5
    eigenvalues=torch.linalg.eigvalsh(s.T@s)
    theory=chain_spectrum(length)
    assert float(eigenvalues[0])==pytest.approx(theory["minimum_gram_eigenvalue"],abs=2e-15)
    assert float(eigenvalues[-1])==pytest.approx(theory["maximum_gram_eigenvalue"],abs=2e-15)
    assert float(eigenvalues[0])>0


@pytest.mark.parametrize("side",[5,9,17])
def test_actual_ac_pixelcenters_equal_diagonal_endpoint_average(side):
    v=torch.randn(2,side,side,2,dtype=torch.float64,generator=torch.Generator().manual_seed(side))
    evaluator=make_evaluator(side,side-1)
    torch.testing.assert_close(evaluator(v),.5*(v[:,:-1,:-1]+v[:,1:,1:]),rtol=0,atol=0)
    assert ((evaluator.weights>0).sum(1)==2).all()


def test_full_boundary_eliminated_matrix_and_oversampling_not_nullmode():
    side=9
    s=sampling_matrix(side,side-1)
    doubled=sampling_matrix(side,2*(side-1))
    eig=torch.linalg.eigvalsh(s.T@s)
    other=torch.linalg.eigvalsh(doubled.T@doubled)
    assert float(eig[0])==pytest.approx(chain_spectrum(side-2)["minimum_gram_eigenvalue"],abs=1e-14)
    assert float(other[0])>float(eig[0])
    assert float(other[-1]/other[0])<float(eig[-1]/eig[0])
    # Independent matrix application, not only eigenvalue comparisons.
    amplitude=torch.randn(side-2,side-2,dtype=s.dtype,generator=torch.Generator().manual_seed(41))
    vertices=torch.zeros(1,side,side,2,dtype=s.dtype)
    vertices[0,1:-1,1:-1,0]=amplitude
    actual=make_evaluator(side,side-1)(vertices)[0,:,:,0].flatten()
    torch.testing.assert_close(s@amplitude.flatten(),actual,rtol=0,atol=0)


def test_large_chain_reports_gram_not_sampling_condition():
    result=chain_spectrum(1023)
    assert result["gram_condition"]==pytest.approx(424971.1792,rel=1e-8)
    assert result["sampling_condition"]**2==pytest.approx(result["gram_condition"])
    assert not result["exact_null_mode"]
    with pytest.raises(ValueError):sampling_matrix(1025,1024)
