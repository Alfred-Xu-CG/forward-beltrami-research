"""Regional scales: independent corners, reference units, seams and true VJP."""

import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_patches import CoordinatedPatchQ1Pass
from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update


def grid(rows, columns=None, dtype=torch.float64):
    columns = rows if columns is None else columns
    y,x = torch.meshgrid(torch.arange(rows,dtype=dtype)/(rows-1),
                         torch.arange(columns,dtype=dtype)/(columns-1),indexing="ij")
    return torch.stack((x,y),dim=-1)[None]


def generic_corners(vertices):
    v=vertices.detach().double().numpy()
    a,b,c,d=v[:,:-1,:-1],v[:,:-1,1:],v[:,1:,1:],v[:,1:,:-1]
    return np.stack([np.linalg.det(np.stack((q-p,r-p),axis=-1))
                     for p,q,r in ((a,b,d),(a,b,c),(d,b,c),(a,c,d))],axis=-1)


@pytest.mark.parametrize("patch_cells",[8,16])
@pytest.mark.parametrize("mode",["radial","analytic"])
def test_distant_patch_not_limited_by_one_thin_region(patch_cells,mode):
    reference=grid(65)
    # Independent monotone x-coordinate map: a thin column remains strictly
    # ordered, and its lost width is recovered immediately in the next cell.
    widths=torch.full((64,),1/64,dtype=reference.dtype)
    widths[14]=.002/64
    widths[15]=(2-.002)/64
    coordinates=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
    anchor=reference.clone()
    anchor[...,0]=coordinates[None,None,:]
    assert np.min(generic_corners(anchor)/generic_corners(reference))>.001
    raw=torch.full(anchor.shape[:-1],.05,dtype=anchor.dtype)
    regional=CoordinatedPatchQ1Pass(65,patch_cells=patch_cells,mode=mode)
    proposal=regional.windowed_proposal(raw)
    global_result=CoordinatedQ1Update(mode=mode)(anchor,proposal,reference=reference)
    result=regional(anchor,raw,reference=reference)
    # Same physical tapered proposal, compared far from the compressed column.
    remote_column=48+patch_cells//2
    remote_row=32+patch_cells//2
    ratio=result.amplitude[0,remote_row,remote_column]/global_result.amplitude[0,remote_row,remote_column]
    assert ratio>100 if patch_cells==8 else ratio>30
    assert np.min(generic_corners(result.vertices)/generic_corners(reference))>.001
    assert result.patch_scales.max()>20*result.patch_scales.min()
    for actual,expected in ((result.vertices[:,0],anchor[:,0]),(result.vertices[:,-1],anchor[:,-1]),
                            (result.vertices[:,:,0],anchor[:,:,0]),(result.vertices[:,:,-1],anchor[:,:,-1])):
        assert torch.equal(actual,expected)


@pytest.mark.parametrize("patch_cells",[8,16])
def test_actual_fine_reference_not_a_local_unit_square(patch_cells):
    reference=grid(65)*torch.tensor((3.,2.),dtype=torch.float64)
    layer=CoordinatedPatchQ1Pass(65,patch_cells=patch_cells,minimum_jacobian=.3)
    raw=torch.zeros(reference.shape[:-1],dtype=reference.dtype,requires_grad=True)
    result=layer(reference,raw,reference=reference)
    assert torch.equal(result.vertices,reference)
    torch.testing.assert_close(result.normalized_margin_min,torch.tensor([.7],dtype=reference.dtype))
    result.vertices.square().sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("offset",[(0,0),(4,4),(3,6)])
def test_nondivisible_rectangular_layout_reports_and_preserves_tails(offset):
    reference=grid(20,27,dtype=torch.float32)
    layer=CoordinatedPatchQ1Pass(20,27,8,offset_row=offset[0],offset_column=offset[1])
    raw=torch.ones(reference.shape[:-1],dtype=reference.dtype)*10
    result=layer(reference,raw,reference=reference)
    independent_starts=[(r,c) for r in range(offset[0],19-8+1,8)
                         for c in range(offset[1],26-8+1,8)]
    assert layer.patch_count==len(independent_starts)
    assert result.covered_cells==len(independent_starts)*64
    assert result.uncovered_cells==19*26-result.covered_cells
    active=np.zeros((20,27),dtype=bool)
    cell_counts=np.zeros((19,26),dtype=int)
    for row,column in independent_starts:
        active[row+1:row+8,column+1:column+8]=True
        cell_counts[row:row+8,column:column+8]+=1
    assert cell_counts.max()==1
    unchanged=torch.from_numpy(~active)
    assert torch.equal(result.vertices[:,unchanged],reference[:,unchanged])
    assert np.min(generic_corners(result.vertices)/generic_corners(reference))>.001


def test_window_is_smooth_over_patch_width_with_exact_zero_edges():
    layer=CoordinatedPatchQ1Pass(17,patch_cells=16)
    assert layer.window[0].count_nonzero()==0 and layer.window[-1].count_nonzero()==0
    assert layer.window[:,0].count_nonzero()==0 and layer.window[:,-1].count_nonzero()==0
    torch.testing.assert_close(layer.window[1,8],torch.sin(torch.tensor(torch.pi/16)).double().square(),
                               atol=1e-8,rtol=1e-7)
    assert layer.window[1,8]<.05 and layer.window[8,8]==1
    assert layer.window[4,8]==pytest.approx(.5)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_regional_proposal_and_geometry_vjp_unique_active_patches(mode):
    torch.manual_seed(297)
    reference=grid(13,17)
    anchor=reference.clone()
    anchor[:,1:-1,1:-1]+=torch.randn(1,11,15,2,dtype=anchor.dtype)*.002
    anchor.requires_grad_()
    raw=(torch.randn(anchor.shape[:-1],dtype=anchor.dtype)*.2).requires_grad_()
    layer=CoordinatedPatchQ1Pass(13,17,4,direction=(.6,.8),mode=mode)
    # Explicitly exclude max ties from this finite-difference fixture.
    patch=anchor.reshape(1,-1,2)[:,layer.patch_ids].reshape(-1,5,5,2)
    refpatch=reference.reshape(1,-1,2)[:,layer.patch_ids].reshape(-1,5,5,2)
    proposal=layer._proposal_patches(raw).reshape(-1,5,5)
    slack,delta,_=layer.operator._constraints(patch,proposal,refpatch)
    active=((-delta).clamp_min(0)/slack).sort(dim=1,descending=True).values
    assert bool((active[:,0]-active[:,1]>1e-6).all())
    assert bool((active[:,0]>1).all())
    weights=torch.randn_like(anchor)
    result=layer(anchor,raw,reference=reference)
    gradients=torch.autograd.grad((result.vertices*weights).sum(),(anchor,raw))
    for index,variable in enumerate((anchor,raw)):
        tangent=torch.randn_like(variable)
        step=1e-6
        plus=[anchor.detach(),raw.detach()]
        minus=[anchor.detach(),raw.detach()]
        plus[index]=variable.detach()+step*tangent
        minus[index]=variable.detach()-step*tangent
        fd=((layer(*plus,reference=reference).vertices*weights).sum()
            -(layer(*minus,reference=reference).vertices*weights).sum())/(2*step)
        ad=(gradients[index]*tangent).sum()
        torch.testing.assert_close(ad,fd,atol=2e-7,rtol=2e-6)


def test_empty_invalid_layout_and_invalid_uncovered_anchor_are_errors():
    with pytest.raises(ValueError,match="no complete patch"):
        CoordinatedPatchQ1Pass(9,patch_cells=8,offset_row=4)
    with pytest.raises(ValueError,match="offsets"):
        CoordinatedPatchQ1Pass(17,patch_cells=8,offset_column=8)
    reference=grid(20,27)
    layer=CoordinatedPatchQ1Pass(20,27,8)
    invalid=reference.clone()
    invalid[:,-1,-1]=torch.tensor((-1.,-1.))
    with pytest.raises(ValueError,match="full-grid"):
        layer(invalid,torch.zeros(reference.shape[:-1],dtype=reference.dtype),reference=reference)


def test_batched_analytic_trials_and_zero_proposal():
    reference=grid(17).expand(2,-1,-1,-1).clone()
    layer=CoordinatedPatchQ1Pass(17,patch_cells=8,mode="analytic")
    raw=torch.zeros(reference.shape[:-1],dtype=reference.dtype,requires_grad=True)
    trial=torch.full((2,layer.patch_count),.2,dtype=reference.dtype)
    result=layer(reference,raw,alpha_trial=trial)
    assert torch.equal(result.vertices,reference)
    torch.testing.assert_close(result.patch_scales,trial)
    result.vertices.square().sum().backward()
    assert torch.isfinite(raw.grad).all()
