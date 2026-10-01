"""Independent geometry recomputation and image-evidence convention checks."""
import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update, single_direction_corner_change
from qcopt.neural_bijection.dense.coordinated_patches import CoordinatedPatchQ1Pass
from tools.coordinated_real_case import Evidence


def triangles_numpy(vertices):
    # Generic homogeneous 3x3 determinants: independent of production cross
    # differences and production Q1 determinant helper.
    rows, columns = vertices.shape[:2]
    answer = np.empty((rows-1, columns-1, 4))
    for i in range(rows-1):
        for j in range(columns-1):
            a,b,c,d = vertices[i,j],vertices[i,j+1],vertices[i+1,j+1],vertices[i+1,j]
            for k, points in enumerate(((a,b,d),(a,b,c),(d,b,c),(a,c,d))):
                matrix = np.concatenate((np.asarray(points), np.ones((3,1))), axis=1)
                answer[i,j,k] = np.linalg.det(matrix)
    return answer


@pytest.mark.parametrize("mode", ["radial", "analytic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_separate_triangle_path(mode, dtype):
    gen = np.random.default_rng(403)
    y,x = np.meshgrid(np.linspace(0,1,7), np.linspace(0,1,9), indexing="ij")
    anchor = np.stack((x,y),-1)
    anchor[1:-1,1:-1] += gen.normal(0,.004,(5,7,2))
    proposal = gen.normal(0,.2,(7,9))
    source = torch.tensor(anchor[None],dtype=dtype)
    raw = torch.tensor(proposal[None],dtype=dtype)
    e = np.array([.8,.6])
    predicted = single_direction_corner_change(source.double(), raw.double(), tuple(e))[0].numpy()
    direct = triangles_numpy(source.double()[0].numpy()+raw.double()[0].numpy()[...,None]*e)-triangles_numpy(source.double()[0].numpy())
    np.testing.assert_allclose(predicted,direct,atol=3e-16,rtol=1e-12)
    result = CoordinatedQ1Update(tuple(e),mode=mode,minimum_jacobian=.001)(source,raw)
    corners = triangles_numpy(result.vertices.double()[0].numpy())
    assert corners.min() > .001/48
    assert np.array_equal(result.vertices[0,0].numpy(), source[0,0].numpy())


@pytest.mark.parametrize("loss", ["mind", "local_ncc"])
def test_original_evidence_identity_and_oob_fixed_denominator(loss):
    torch.manual_seed(904)
    image = torch.rand(1,1,32,32)
    y,x = torch.meshgrid(torch.linspace(0,1,9),torch.linspace(0,1,9),indexing="ij")
    vertices = torch.stack((x,y),-1)[None].requires_grad_()
    evidence = Evidence(image,image,torch.eye(2),torch.zeros(2),loss,.05,1.)
    initial,parts = evidence(vertices)
    if loss == "mind":
        assert float(parts["image"]) < 2e-6
    assert float(parts["outside_fraction"]) == 0
    initial.backward()
    assert bool(torch.isfinite(vertices.grad).all())
    denominator = float(evidence.denominator)
    shifted = vertices.detach()+torch.tensor([2.,0.])
    worse, shifted_parts = evidence(shifted)
    assert float(evidence.denominator) == denominator
    assert float(shifted_parts["outside_fraction"]) == 1
    assert float(shifted_parts["oob"]) > 1
    assert float(worse) > float(initial)


def test_independent_tiny_inactive_analytic_gradient():
    y,x = torch.meshgrid(torch.linspace(0,1,4,dtype=torch.float64),
                         torch.linspace(0,1,4,dtype=torch.float64),indexing="ij")
    vertices = torch.stack((x,y),-1)[None].requires_grad_()
    proposal = torch.zeros(1,4,4,dtype=torch.float64)
    proposal[0,1,1] = 1e-305
    proposal.requires_grad_()
    result = CoordinatedQ1Update(mode="analytic")(vertices,proposal)
    assert float(result.scale) == 1
    g_vertices,g_proposal = torch.autograd.grad(result.vertices.sum(),(vertices,proposal))
    assert bool(torch.isfinite(g_vertices).all() and torch.isfinite(g_proposal).all())
    assert float(g_proposal[0,1,1]) == 1


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_regional_separate_reconstruction(mode):
    y,x = torch.meshgrid(torch.linspace(0,1,13,dtype=torch.float64),
                         torch.linspace(0,1,18,dtype=torch.float64),indexing="ij")
    reference = torch.stack((x,y),-1)[None]
    current = reference.clone()
    current[0,5,5,0] -= .997/17
    raw = torch.full((1,13,18),.12,dtype=torch.float64,requires_grad=True)
    layer = CoordinatedPatchQ1Pass(13,18,patch_cells=4,offset_row=1,offset_column=1,mode=mode)
    result = layer(current,raw,reference=reference)
    # Recompute all grid triangles, not just gathered patch subarrays.
    q = triangles_numpy(result.vertices[0].detach().numpy())
    assert q.min()*12*17 > .001
    assert torch.equal(result.vertices[:,0],reference[:,0])
    assert torch.equal(result.vertices[:,:,-1],reference[:,:,-1])
    assert result.covered_cells + result.uncovered_cells == 12*17
    actual = (result.vertices-current)[...,0]
    torch.testing.assert_close(actual,result.amplitude,rtol=0,atol=8e-17)
    result.vertices.square().sum().backward()
    assert bool(torch.isfinite(raw.grad).all())
