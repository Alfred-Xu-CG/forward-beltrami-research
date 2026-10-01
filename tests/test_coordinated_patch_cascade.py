"""Independent corners, coverage, and full-chain geometry/proposal derivatives."""
import pytest
import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_patch_cascade import CoordinatedPatchCascade
from qcopt.neural_bijection.dense.coordinated_patches import CoordinatedPatchQ1Pass
from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update


def grid(rows, columns=None, dtype=torch.float64):
    columns = rows if columns is None else columns
    yy, xx = torch.meshgrid(torch.arange(rows, dtype=dtype)/(rows-1),
                           torch.arange(columns, dtype=dtype)/(columns-1), indexing="ij")
    return torch.stack((xx, yy), -1)[None]


def corners(vertices):
    v = vertices.detach().double().numpy()
    a, b, c, d = v[:, :-1, :-1], v[:, :-1, 1:], v[:, 1:, 1:], v[:, 1:, :-1]
    return np.stack([np.linalg.det(np.stack((q-p, r-p), -1))
                     for p, q, r in ((a,b,d), (a,b,c), (d,b,c), (a,c,d))], -1)


@pytest.mark.parametrize("rows,columns,p", [(17,25,8), (9,13,4), (21,29,8), (20,27,8), (22,30,8)])
def test_independent_vertex_and_cell_coverage_with_tails(rows, columns, p):
    layer = CoordinatedPatchCascade(rows, columns, p)
    union = np.zeros((rows, columns), dtype=bool)
    for index, (offrow, offcolumn) in enumerate(layer.offsets):
        mask = np.zeros_like(union)
        count = 0
        for row in range(offrow, rows-p, p):
            for column in range(offcolumn, columns-p, p):
                mask[row+1:row+p, column+1:column+p] = True
                count += 1
        actual = np.zeros(rows*columns, dtype=bool)
        actual[layer.passes[index].interior_ids.numpy()] = True
        np.testing.assert_array_equal(actual.reshape(rows, columns), mask)
        assert layer.passes[index].covered_cells == count*p*p
        union |= mask
    if (rows-1)%(p//2) == 0 and (columns-1)%(p//2) == 0:
        assert union[1:-1, 1:-1].all()  # Includes original patch seams.
    else:
        assert not union[1:-1, 1:-1].all()  # Explicit uncovered tails, not a coverage promise.
    expected = np.zeros_like(union)
    er = p//2*((rows-1)//(p//2))
    ec = p//2*((columns-1)//(p//2))
    expected[1:er, 1:ec] = True
    np.testing.assert_array_equal(union, expected)
    reference = grid(rows, columns)
    result = layer(reference, torch.ones(reference.shape[:-1], dtype=reference.dtype)*.003)
    assert result.geometry_pass_count == 4
    assert result.pass_patch_counts == tuple(op.patch_count for op in layer.passes)
    assert result.pass_covered_cells == tuple(op.covered_cells for op in layer.passes)
    assert result.pass_uncovered_cells == tuple(op.uncovered_cells for op in layer.passes)
    assert torch.equal(result.vertices[:, ~torch.from_numpy(union)], reference[:, ~torch.from_numpy(union)])


@pytest.mark.parametrize("mode", ["radial", "analytic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch2_rectangular_zero_boundary_actual_corners(mode, dtype):
    reference = grid(17, 25, dtype)*torch.tensor((2., 3.), dtype=dtype)
    anchor = reference.repeat(2, 1, 1, 1)
    layer = CoordinatedPatchCascade(17, 25, 8, direction=(.6,.8), mode=mode)
    zero = layer(anchor, torch.zeros(anchor.shape[:-1], dtype=dtype), reference=reference)
    assert torch.equal(zero.vertices, anchor)
    proposal = torch.full(anchor.shape[:-1], .04, dtype=dtype)
    result = layer(anchor, proposal, reference=reference, alpha_trial=torch.tensor([.7,.8]))
    margin = corners(result.vertices)/corners(reference)-.001
    np.testing.assert_allclose(result.normalized_margin_min.detach().numpy(),
                               margin.reshape(2,-1).min(1), atol=2e-14, rtol=2e-13)
    assert margin.min() > 0
    for index in (0,-1):
        assert torch.equal(result.vertices[:,index], anchor[:,index])
        assert torch.equal(result.vertices[:,:,index], anchor[:,:,index])
    assert result.patch_scales.shape == (2, sum(result.pass_patch_counts))
    current = anchor
    actual_pass_margins = []
    for op in layer.passes:
        current = op(current, proposal, reference=reference, alpha_trial=torch.tensor([.7,.8]), validate=False).vertices
        actual_pass_margins.append((corners(current)/corners(reference)-.001).reshape(2,-1).min(1))
    np.testing.assert_allclose(result.pass_margin_min.detach().numpy(),
                               np.stack(actual_pass_margins, 1), atol=2e-14, rtol=2e-13)
    trusted = layer(anchor, proposal, reference=reference, alpha_trial=torch.tensor([.7,.8]), validate=False)
    torch.testing.assert_close(trusted.pass_margin_min, result.pass_margin_min, rtol=0, atol=0)


@pytest.mark.parametrize("rows,columns", [(17,25), (20,27), (21,29)])
def test_sum_windows_and_unsaturated_amplitude_not_four_times_raw(rows,columns):
    p = 8
    reference = grid(rows, columns)
    layer = CoordinatedPatchCascade(rows, columns, p, mode="analytic", direction=(.6,.8))
    total = torch.zeros(rows,columns,dtype=reference.dtype)
    for offrow, offcolumn in layer.offsets:
        for row in range(offrow, rows-p, p):
            for column in range(offcolumn, columns-p, p):
                r = torch.arange(p+1,dtype=reference.dtype)
                axis = torch.sin(torch.pi*r/p).square()
                axis[0] = axis[-1] = 0
                total[row:row+p+1,column:column+p+1] += axis[:,None]*axis[None,:]
    assert float(total.max()) <= 1+2e-15
    assert float(total.min()) == 0
    # Complete complementary windows partition unity away from truncation.
    torch.testing.assert_close(total[p//2:p+1,p//2:p+1], torch.ones(p//2+1,p//2+1,dtype=reference.dtype),
                               rtol=0,atol=2e-15)
    raw = torch.full(reference.shape[:-1], .0001, dtype=reference.dtype)
    result = layer(reference,raw)
    assert torch.equal(result.patch_scales,torch.ones_like(result.patch_scales))
    expected = reference+raw[...,None]*total[None,...,None]*torch.tensor([.6,.8],dtype=reference.dtype)
    torch.testing.assert_close(result.vertices,expected,rtol=0,atol=3e-16)


@pytest.mark.parametrize("mode", ["radial", "analytic"])
def test_explicit_sequential_oracle_full_y_and_proposal_fd_unique_active(mode):
    generator = torch.Generator().manual_seed(297)
    reference = grid(13, 17)
    anchor = reference.clone()
    anchor[:,1:-1,1:-1] += .002*torch.randn(1,11,15,2, generator=generator, dtype=anchor.dtype)
    anchor.requires_grad_()
    proposal = (.2*torch.randn(anchor.shape[:-1], generator=generator, dtype=anchor.dtype)).requires_grad_()
    layer = CoordinatedPatchCascade(13, 17, 4, direction=(.6,.8), mode=mode)
    current = anchor
    for offrow, offcolumn in layer.offsets:
        op = CoordinatedPatchQ1Pass(13, 17, 4, direction=(.6,.8), mode=mode,
                                    offset_row=offrow, offset_column=offcolumn)
        patch = current.reshape(1,-1,2)[:,op.patch_ids].reshape(-1,5,5,2)
        ref = reference.reshape(1,-1,2)[:,op.patch_ids].reshape(-1,5,5,2)
        raw = op._proposal_patches(proposal).reshape(-1,5,5)
        slack, delta, _ = op.operator._constraints(patch, raw, ref)
        top = ((-delta).clamp_min(0)/slack).sort(dim=1, descending=True).values
        assert (top[:,0]-top[:,1]>1e-6).all() and (top[:,0]>1).all()
        current = op(current, proposal, reference=reference).vertices
    result = layer(anchor, proposal, reference=reference)
    torch.testing.assert_close(result.vertices, current, rtol=0, atol=0)
    weights = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
    grads = torch.autograd.grad((result.vertices*weights).sum(), (anchor, proposal), retain_graph=True)
    oracle = torch.autograd.grad((current*weights).sum(), (anchor, proposal))
    for index, variable in enumerate((anchor, proposal)):
        torch.testing.assert_close(grads[index], oracle[index], rtol=0, atol=0)
        tangent = torch.randn(variable.shape, generator=generator, dtype=variable.dtype)
        plus, minus = [anchor.detach(), proposal.detach()], [anchor.detach(), proposal.detach()]
        step = 1e-6
        plus[index], minus[index] = variable.detach()+step*tangent, variable.detach()-step*tangent
        fd = ((layer(*plus, reference=reference).vertices-layer(*minus, reference=reference).vertices)*weights).sum()/(2*step)
        torch.testing.assert_close((grads[index]*tangent).sum(), fd, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("mode", ["radial", "analytic"])
def test_thin_region_distant_motion_same_four_window_supports(mode):
    reference = grid(65)
    widths = torch.full((64,), 1/64, dtype=reference.dtype)
    widths[14], widths[15] = .002/64, (2-.002)/64
    anchor = reference.clone()
    anchor[...,0] = torch.cat((widths.new_zeros(1), widths.cumsum(0)))[None,None,:]
    raw = torch.full(anchor.shape[:-1], .05, dtype=reference.dtype)
    layer = CoordinatedPatchCascade(65, patch_cells=8, mode=mode)
    regional = layer(anchor, raw, reference=reference)
    global_current = anchor
    # Same four passes/windowed fields: only patch-wise vs global coupling differs.
    for op in layer.passes:
        global_current = CoordinatedQ1Update(mode=mode)(
            global_current, op.windowed_proposal(raw), reference=reference).vertices
    regional_motion = (regional.vertices-anchor)[0,36,52,0]
    global_motion = (global_current-anchor)[0,36,52,0]
    assert regional_motion > 20*global_motion
    assert (corners(regional.vertices)/corners(reference)).min() > .001


@pytest.mark.parametrize("rows,columns,p", [(17,17,3),(17,17,0),(17,17,True),(16,17,8),(17,16,8),(17.,17,8)])
def test_invalid_patch_or_grid(rows, columns, p):
    with pytest.raises(ValueError):
        CoordinatedPatchCascade(rows, columns, p)


def test_invalid_inputs_and_uncovered_bad_anchor_rejected():
    layer = CoordinatedPatchCascade(20,27,8)
    reference = grid(20,27)
    raw = torch.zeros(reference.shape[:-1], dtype=reference.dtype)
    bad = reference.clone()
    bad[:,-1,-1] = -1
    with pytest.raises(ValueError, match="full-grid"):
        layer(bad, raw, reference=reference)
    with pytest.raises(ValueError, match="scalar"):
        layer(reference, raw, alpha_trial=torch.ones(1, layer.passes[0].patch_count))
    with pytest.raises(ValueError, match="bool"):
        layer(reference, raw, validate=1)


@pytest.mark.parametrize("mode", ["radial","analytic"])
def test_zero_tied_gauge_full_chain_gradients_remain_finite(mode):
    anchor = grid(9,13).repeat(2,1,1,1).requires_grad_()
    proposal = torch.zeros(anchor.shape[:-1],dtype=anchor.dtype,requires_grad=True)
    result = CoordinatedPatchCascade(9,13,4,mode=mode)(anchor,proposal)
    assert torch.equal(result.vertices,anchor)
    assert result.patch_gauges.count_nonzero() == 0
    yg,zg = torch.autograd.grad(result.vertices.square().sum(),(anchor,proposal))
    assert torch.isfinite(yg).all() and torch.isfinite(zg).all()
    torch.testing.assert_close(yg,2*anchor,rtol=0,atol=0)


@pytest.mark.parametrize("case", ["vertices_shape","proposal_shape","proposal_dtype","vertices_dtype"])
def test_input_shape_and_precision_guards(case):
    anchor = grid(9)
    proposal = torch.zeros(anchor.shape[:-1],dtype=anchor.dtype)
    if case == "vertices_shape": anchor = anchor[:,:,:-1]
    elif case == "proposal_shape": proposal = proposal[:,:,:-1]
    elif case == "proposal_dtype": proposal = proposal.float()
    elif case == "vertices_dtype": anchor,proposal = anchor.half(),proposal.half()
    with pytest.raises(ValueError):
        CoordinatedPatchCascade(9,patch_cells=4)(anchor,proposal)
