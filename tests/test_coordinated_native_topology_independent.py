"""Focused independent audit predicate checks, without registration jobs."""
import numpy as np
from tools.coordinated_native_topology_independent import exact_boundary_check, exact_orientation


def test_boundary_identity_and_crossing():
    identity=np.array([[[0.,0.],[1.,0.]],[[0.,1.],[1.,1.]]])
    assert exact_boundary_check(identity)['simple']
    reversed_result=exact_boundary_check(identity[:,::-1])
    assert reversed_result['simple']
    assert reversed_result['signed_area_sign']==-1
    crossed=identity.copy()
    crossed[0,1]=[1.,1.]
    crossed[1,1]=[1.,0.]
    assert exact_boundary_check(crossed)['nonadjacent_intersections']==1
    assert exact_boundary_check(crossed[:,::-1])['nonadjacent_intersections']==1


def test_adjacent_backtracking_is_not_simple():
    vertices=np.array([[[0.,0.],[2.,0.],[1.,0.]],
                       [[0.,1.],[.5,1.],[1.,1.]]])
    result=exact_boundary_check(vertices)
    assert not result['simple']
    assert result['adjacent_overlaps']==1


def test_nonadjacent_endpoint_touch_is_not_simple():
    vertices=np.array([[[0.,0.],[2.,0.],[2.,2.]],
                       [[1.,1.],[0.,2.],[1.,1.]]])
    result=exact_boundary_check(vertices)
    assert not result['simple']
    assert result['zero_edges']==0
    assert result['nonadjacent_intersections']>0


def test_exact_orientation_sign_and_collinearity():
    assert exact_orientation((0.,0.),(1.,0.),(1.,2.**-100))==1
    assert exact_orientation((0.,0.),(1.,0.),(1.,-2.**-100))==-1
    assert exact_orientation((0.,0.),(1.,1.),(.5,.5))==0
