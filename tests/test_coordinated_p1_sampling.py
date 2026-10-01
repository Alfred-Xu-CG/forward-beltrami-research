import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries,p1_map_at_pixel_centers
from qcopt.neural_bijection.dense.q1_image_sampling import q1_map_at_pixel_centers


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_affine_values_and_source_endpoints(diagonal):
    y,x=torch.meshgrid(torch.linspace(0,1,5,dtype=torch.float64),torch.linspace(0,1,7,dtype=torch.float64),indexing="ij")
    matrix=torch.tensor([[1.3,.2],[-.1,.8]],dtype=torch.float64)
    offset=torch.tensor([.02,-.03],dtype=torch.float64)
    vertices=torch.stack((x,y),-1)[None]@matrix.T+offset
    points=torch.tensor([[[0.,0.],[1.,1.],[0.,1.],[1.,0.],[.13,.27],[.5,.5]]],dtype=torch.float64)
    torch.testing.assert_close(p1_map_at_queries(vertices,points,diagonal),points@matrix.T+offset,rtol=0,atol=5e-16)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_independent_triangle_barycentric_inverse(diagonal):
    vertices=torch.tensor([[[[.1,.2],[1.2,.1]],[[.05,1.1],[1.1,1.4]]]],dtype=torch.float64)
    query=torch.tensor([[[.8,.1],[.1,.8],[.5,.5],[.3,.6]]],dtype=torch.float64)
    source=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    target=np.stack((vertices[0,0,0],vertices[0,0,1],vertices[0,1,1],vertices[0,1,0]))
    expected=[]
    for point in query[0].numpy():
        if diagonal=="ac":
            ids=[0,1,2] if point[1]<=point[0] else [0,2,3]
        else:
            ids=[0,1,3] if point.sum()<=1 else [1,2,3]
        s=source[ids]
        coordinate=np.linalg.solve(np.stack((s[1]-s[0],s[2]-s[0]),axis=1),point-s[0])
        weights=np.array([1-coordinate.sum(),*coordinate])
        expected.append(weights@target[ids])
    np.testing.assert_allclose(p1_map_at_queries(vertices,query,diagonal)[0].numpy(),expected,atol=3e-16)


def test_nonaffine_p1_not_q1_and_both_diagonal_center_values():
    vertices=torch.tensor([[[[0.,0.],[1.,0.]],[[0.,1.],[1.2,1.1]]]],dtype=torch.float64)
    center=torch.tensor([[[.5,.5]]],dtype=torch.float64)
    ac=p1_map_at_queries(vertices,center,"ac");bd=p1_map_at_queries(vertices,center,"bd")
    torch.testing.assert_close(ac,(vertices[:,0,0]+vertices[:,1,1])[:,None]/2)
    torch.testing.assert_close(bd,(vertices[:,0,1]+vertices[:,1,0])[:,None]/2)
    q1=q1_map_at_pixel_centers(vertices,1,1).reshape(1,1,2)
    assert not torch.equal(ac,q1) and not torch.equal(bd,q1)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_batch_and_vertex_vjp(diagonal):
    torch.manual_seed(802)
    vertices=torch.randn(2,5,7,2,dtype=torch.float64,requires_grad=True)
    query=torch.tensor([[[.13,.27],[.44,.69],[.95,.04]]],dtype=torch.float64)
    direction=torch.randn_like(vertices)*.1
    upstream=torch.randn(2,3,2,dtype=torch.float64)
    value=(p1_map_at_queries(vertices,query,diagonal)*upstream).sum()
    gradient,=torch.autograd.grad(value,vertices)
    step=1e-6
    fd=((p1_map_at_queries(vertices+step*direction,query,diagonal)-p1_map_at_queries(vertices-step*direction,query,diagonal))*upstream).sum()/(2*step)
    torch.testing.assert_close(fd,(gradient*direction).sum(),rtol=2e-8,atol=3e-10)
    assert p1_map_at_pixel_centers(vertices,9,11,diagonal).shape==(2,9,11,2)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_separate_saved_map_scorer(diagonal):
    from tools.coordinated_score_case import p1_at_queries_numpy
    generator=np.random.default_rng(771)
    vertices=torch.from_numpy(generator.normal(size=(1,5,7,2)))
    query=torch.from_numpy(generator.uniform(size=(1,17,2)))
    expected=p1_at_queries_numpy(vertices[0].numpy(),query[0].numpy(),diagonal)
    np.testing.assert_allclose(p1_map_at_queries(vertices,query,diagonal)[0].numpy(),expected,atol=1e-15)


@pytest.mark.parametrize("point",[[float("nan"),.5],[-.01,.5],[1.01,.5]])
def test_public_query_domain_is_enforced(point):
    vertices=torch.zeros(1,3,3,2,dtype=torch.float64)
    with pytest.raises(ValueError,match="finite and inside"):
        p1_map_at_queries(vertices,torch.tensor([[point]],dtype=torch.float64))


def test_affine_range_bound_is_necessary_not_correspondence_success():
    from tools.coordinated_score_case import affine_domain_bound
    a=np.array([[2.,0.],[0.,1.]])
    offset=np.array([.1,-.2])
    expected=np.array([[1.,.3],[2.3,.3],[-.2,-.6]])
    report=affine_domain_bound(expected,a,offset,100)
    assert report["outside_required_landmarks"]==2
    assert report["mean_unavoidable_canvas_px"]==pytest.approx((20+50)/3)
    assert report["max_unavoidable_canvas_px"]==pytest.approx(50)
